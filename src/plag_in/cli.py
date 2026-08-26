"""Operator CLI: inspect, recommend, serve, stop, status, verify, receipt, connect.

There is deliberately no `download` command in this MVP (D-001, PRODUCT_SPEC
section 6): recommendation never triggers a network action, and serving
never triggers an implicit download.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from plag_in.config import is_loopback, load_config_file
from plag_in.connect import emit
from plag_in.engines.base import EngineAdapter
from plag_in.engines.libllama import EmbeddedLibLlama, LibLlamaAdapter
from plag_in.engines.llama_server import LlamaServerAdapter
from plag_in.errors import (
    BackendResponseError,
    ConfigurationError,
    LocalityPolicyError,
    ModelIdentityMismatchError,
    PathContainmentError,
    PlagInError,
)
from plag_in.gateway import BackendInfo, GatewayContext, GatewayServer
from plag_in.identity import canonical_digest
from plag_in.input_control import confirm_action, select_number
from plag_in.inventory import discover_gguf, discover_ollama_models
from plag_in.model_paths import resolve_contained_model
from plag_in.onboarding import doctor_report, render_doctor, run_setup_assistant
from plag_in.paths import StateLayout
from plag_in.receipts import ReceiptStore
from plag_in.recommend import RecommendationQuery, load_catalogue, recommend as recommend_fn
from plag_in.registry import Registry, RegistryEntry
from plag_in.resource_safety import (
    CHAT_MEMORY_HIGH_BYTES,
    CHAT_MEMORY_MAX_BYTES,
    CHAT_SWAP_MAX_BYTES,
    resource_preflight,
    verify_chat_cgroup,
)
from plag_in.setup_flow import (
    configure_tested_model,
    default_config_path,
    default_state_path,
)
from plag_in.supervisor import Supervisor

ENGINE_ADAPTERS: dict[str, EngineAdapter] = {
    "llama_server": LlamaServerAdapter(),
    "libllama": LibLlamaAdapter(),
}

_MEMORY_OVERHEAD_FACTOR = 1.2  # rough runtime overhead multiplier; always labelled an estimate
_BOUNDED_CHAT_ENV = "PLAG_IN_BOUNDED_CHAT"


def _memory_estimate_gb(size_bytes: int) -> float:
    return round((size_bytes / (1024**3)) * _MEMORY_OVERHEAD_FACTOR, 3)


def _runtime_profile_summary(config, identity, engine_name: str) -> dict:
    return {
        "engine": "libllama_embedded" if engine_name == "libllama" else "llama_server_direct",
        "ollama_runtime_dependency": False,
        "model_path": identity.path,
        "model_sha256": identity.sha256,
        "model_size_bytes": identity.size_bytes,
        "quantisation": "fixed by selected model bytes; human-readable label unverified",
        "runtime": config.runtime.as_dict(),
        "sampling_note": "temperature and top_p are server fallbacks; explicit client request values take precedence",
    }


def _runtime_receipt_profile(config, engine_name: str) -> dict:
    if engine_name == "libllama":
        return config.runtime.as_embedded_request_record()
    return config.runtime.as_direct_record()


def _require_runtime_confirmation(args, summary: dict) -> None:
    if args.confirm_runtime_profile:
        return
    if not sys.stdin.isatty():
        raise ConfigurationError(
            "runtime profile confirmation required; rerun with --confirm-runtime-profile after reviewing it",
            runtime_profile=summary,
        )
    print(json.dumps({"confirmation_required": True, "runtime_profile": summary}, sort_keys=True))
    answer = input("Load the selected local model with this runtime profile? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise ConfigurationError("runtime profile was not confirmed; no listener or engine was started")


@dataclass
class _EmbeddedSession:
    alias: str
    server: GatewayServer
    embedded: EmbeddedLibLlama
    context: GatewayContext
    receipts: ReceiptStore

    def close(self) -> None:
        self.server.stop()
        self.embedded.close()


def _start_embedded_session(config, alias: str, profile, state_dir: Path) -> _EmbeddedSession:
    engine_cfg = config.engines[profile.engine]
    if engine_cfg.kind != "embedded":
        raise ConfigurationError("private test chat requires an embedded engine")
    identity = resolve_contained_model(
        profile.model_path, list(config.model_roots), config.ollama_manifest_root
    )
    registry = Registry()
    config_digest = canonical_digest(
        {
            "engine": profile.engine,
            "identity": engine_cfg.identity_record(),
            "runtime": config.runtime.as_dict(),
        }
    )
    registry.bind(
        RegistryEntry(
            alias=alias,
            identity=identity,
            engine=profile.engine,
            engine_config_digest=config_digest,
        )
    )
    registry.verify_identity(alias)

    state = StateLayout(state_dir).ensure()
    receipts = ReceiptStore(state.receipts_file, hmac_key_path=state.receipts_hmac_key_file)
    context = GatewayContext(config, registry, receipts, ENGINE_ADAPTERS)
    server = GatewayServer(context)
    embedded = EmbeddedLibLlama(
        library_path=engine_cfg.library,
        library_sha256=engine_cfg.library_sha256,
        ggml_base_library_path=engine_cfg.ggml_base_library,
        ggml_base_library_sha256=engine_cfg.ggml_base_library_sha256,
        ggml_library_path=engine_cfg.ggml_library,
        ggml_library_sha256=engine_cfg.ggml_library_sha256,
        backend_libraries=engine_cfg.backend_libraries,
        upstream_identity=engine_cfg.upstream_identity,
        model_path=identity.path,
        model_sha256=identity.sha256,
        runtime=config.runtime,
    )
    try:
        embedded.load()
        runtime_profile = embedded.runtime_profile
        context.register_backend(
            alias,
            BackendInfo(
                engine=profile.engine,
                engine_version=engine_cfg.upstream_identity,
                inference_mode="embedded",
                host="127.0.0.1",
                port=0,
                weight_digest=identity.sha256,
                config_digest=config_digest,
                template_digest=embedded.template_digest,
                native_identity_digest=runtime_profile.get("native_identity_digest"),
                native_component_digests=runtime_profile.get("native_component_digests"),
                upstream_identity=engine_cfg.upstream_identity,
                requested_runtime_profile=runtime_profile.get("requested"),
                effective_runtime_profile=runtime_profile.get("effective"),
                measurement_status=runtime_profile.get("measurement_status"),
                gpu_offload=runtime_profile.get("gpu_offload"),
                runtime_profile=runtime_profile,
                inference_backend=embedded,
            ),
        )
        server.start()
    except BaseException:
        embedded.close()
        server.stop()
        raise
    return _EmbeddedSession(alias, server, embedded, context, receipts)


def _profile_api_key(config, alias: str) -> str | None:
    if config.security.auth_mode == "none":
        return None
    for key in config.security.api_keys:
        if not key.aliases or alias in key.aliases:
            return key.secret
    raise ConfigurationError(f"no API key permits configured profile: {alias}")


def _select_tested_profile(config, input_fn: Callable[[str], str], output: TextIO):
    tested = [(alias, profile) for alias, profile in config.profiles.items()
              if profile.compatibility_status == "tested"]
    if not tested:
        raise ConfigurationError("configuration contains no compatibility-tested model profile")
    if len(tested) == 1:
        return tested[0]
    print("Tested local profiles", file=output)
    for index, (_, profile) in enumerate(tested, 1):
        print(f"  {index}. {profile.display_name}", file=output)
    selection = select_number(
        input_fn,
        output,
        f"Select 1-{len(tested)}: ",
        minimum=1,
        maximum=len(tested),
    )
    if selection is None:
        raise ConfigurationError("tested profile selection was cancelled; no model was started")
    return tested[selection - 1]


def _model_load_safety(identity, gpu_layers: str) -> tuple[dict, dict]:
    cgroup = verify_chat_cgroup()
    safety = resource_preflight(identity.size_bytes, gpu_layers)
    if safety["status"] != "pass":
        raise ConfigurationError(
            "host resource safety preflight failed; no model was started",
            resource_safety=safety,
        )
    return cgroup, safety


def _local_chat_request(base_url: str, api_key: str | None, alias: str,
                        messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps(
        {"model": alias, "messages": messages, "max_tokens": max_tokens, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body, method="POST"
    )
    request.add_header("Content-Type", "application/json")
    if api_key is not None:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendResponseError(f"local gateway returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise BackendResponseError(f"local gateway request failed: {exc}") from exc
    try:
        payload = json.loads(raw)
        content = payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError) as exc:
        raise BackendResponseError("local gateway returned an invalid chat response") from exc
    if not isinstance(content, str):
        raise BackendResponseError("local gateway returned non-text chat content")
    return {"content": content, "payload": payload}


def run_private_chat(
    config_path: Path,
    input_fn: Callable[[str], str],
    output: TextIO,
    *,
    state_dir: Path | None = None,
    max_tokens: int = 256,
) -> int:
    """Start one confirmed embedded model and stop it when the chat closes."""
    config = load_config_file(config_path)
    alias, profile = _select_tested_profile(config, input_fn, output)
    identity = resolve_contained_model(
        profile.model_path, list(config.model_roots), config.ollama_manifest_root
    )
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    summary = _runtime_profile_summary(config, identity, profile.engine)
    effective_state = (state_dir or default_state_path()).resolve()
    print(
        f"\nPrivate test chat\n"
        f"  Model: {profile.display_name}\n"
        f"  Alias: {alias}\n"
        f"  Model SHA-256: {identity.sha256}\n"
        f"  Bind: {config.bind.host}:{config.bind.port}\n"
        f"  Authentication: {config.security.auth_mode}\n"
        f"  Content logging: {str(config.security.content_logging).lower()}\n"
        f"  Remote provider: absent\n"
        f"  Runtime: {json.dumps(summary['runtime'], sort_keys=True)}\n"
        f"  Receipt state: {effective_state}\n",
        file=output,
    )
    resource_text = (
        f"  Cgroup memory high: {cgroup['memory.high'] // (1024**2)} MiB\n"
        f"  Cgroup memory maximum: {cgroup['memory.max'] // (1024**2)} MiB\n"
        f"  Available RAM: {safety['available_ram_bytes'] // (1024**2)} MiB\n"
        f"  Required available RAM: {safety['required_available_ram_bytes'] // (1024**2)} MiB\n"
    )
    if safety["gpu"] is not None:
        resource_text += (
            f"  Free VRAM: {safety['gpu']['free_mib']} MiB\n"
            f"  Required free VRAM: {safety['required_free_vram_mib']} MiB\n"
        )
    print(resource_text, file=output)
    if not confirm_action(
        input_fn,
        output,
        "Load this model and start the private test chat? [y/N] ",
    ):
        print("No listener or model was started.", file=output)
        return 0

    session = _start_embedded_session(config, alias, profile, effective_state)
    api_key = _profile_api_key(config, alias)
    messages: list[dict] = []
    print(
        f"Local chat ready at {session.server.base_url}. Type /exit to stop the model.",
        file=output,
    )
    try:
        while True:
            try:
                prompt = input_fn("you> ")
            except (EOFError, KeyboardInterrupt):
                print("", file=output)
                break
            if prompt.strip().lower() in {"/exit", "/quit"}:
                break
            if not prompt.strip():
                continue
            messages.append({"role": "user", "content": prompt})
            response = _local_chat_request(
                session.server.base_url, api_key, alias, messages, max_tokens
            )
            answer_text = response["content"]
            messages.append({"role": "assistant", "content": answer_text})
            print(f"model> {answer_text}", file=output)
    finally:
        session.close()
    valid, count = session.receipts.verify_chain()
    print(
        f"Model stopped. Receipt chain valid: {str(valid).lower()}. Requests recorded: {count}.",
        file=output,
    )
    return 0


def run_profile_gateway(
    config_path: Path,
    input_fn: Callable[[str], str],
    output: TextIO,
    *,
    state_dir: Path | None = None,
    connector_format: str = "manual",
) -> int:
    """Run one bounded tested profile for an existing local harness."""
    config = load_config_file(config_path)
    alias, profile = _select_tested_profile(config, input_fn, output)
    identity = resolve_contained_model(
        profile.model_path, list(config.model_roots), config.ollama_manifest_root
    )
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    effective_state = (state_dir or default_state_path()).resolve()
    gpu_text = (
        f"  Free VRAM: {safety['gpu']['free_mib']} MiB\n"
        if safety["gpu"] is not None else "  GPU route: disabled\n"
    )
    print(
        f"\nHarness gateway\n"
        f"  Model: {profile.display_name}\n"
        f"  Alias: {alias}\n"
        f"  Model SHA-256: {identity.sha256}\n"
        f"  Cgroup memory maximum: {cgroup['memory.max'] // (1024**2)} MiB\n"
        f"  Available RAM: {safety['available_ram_bytes'] // (1024**2)} MiB\n"
        f"{gpu_text}"
        f"  Authentication: {config.security.auth_mode}\n"
        f"  Content logging: {str(config.security.content_logging).lower()}\n"
        f"  Connector output: includes the scoped local key in this terminal\n"
        f"  Receipt state: {effective_state}\n",
        file=output,
    )
    if not confirm_action(input_fn, output, "Load this model for an existing harness? [y/N] "):
        print("No listener or model was started.", file=output)
        return 0

    session = _start_embedded_session(config, alias, profile, effective_state)
    api_key = _profile_api_key(config, alias) or ""
    connector = emit(
        connector_format,
        session.server.base_url + "/v1",
        api_key,
        alias,
        reveal_secret=True,
    )
    print(
        "\nGateway ready. Enter these values in the harness running on this computer:\n",
        file=output,
    )
    print(connector, file=output)
    print("Keep this terminal open. Press Ctrl+C here to stop the model.", file=output)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("", file=output)
    finally:
        session.close()
    valid, count = session.receipts.verify_chain()
    print(
        f"Model stopped. Receipt chain valid: {str(valid).lower()}. Requests recorded: {count}.",
        file=output,
    )
    return 0


def _setup_configure(active_config: Path | None, input_fn, output) -> Path | None:
    return configure_tested_model(
        input_fn=input_fn,
        output=output,
        config_path=active_config or default_config_path(),
    )


def _setup_chat(config_path: Path, input_fn, output) -> int:
    try:
        return _launch_bounded_command(["chat", "--config", str(config_path)])
    except PlagInError as exc:
        print(f"Chat did not start: {exc.message}", file=output)
        return 1


def _setup_connect(config_path: Path, input_fn, output) -> int:
    print(
        "\nHarness connector\n"
        "  1. Generic local endpoint details\n"
        "  2. OpenAI-style environment values\n"
        "  3. Generic JSON values\n"
        "  4. Back\n",
        file=output,
    )
    choice = select_number(input_fn, output, "Select 1-4: ", minimum=1, maximum=4)
    formats = {"1": "manual", "2": "openai-env", "3": "json"}
    if choice in {None, 4}:
        return 0
    connector_format = formats.get(str(choice))
    if connector_format is None:
        print("Invalid connector selection. No model was started.", file=output)
        return 1
    try:
        return _launch_bounded_command(
            [
                "gateway",
                "--config",
                str(config_path),
                "--connector-format",
                connector_format,
            ]
        )
    except PlagInError as exc:
        print(f"Gateway did not start: {exc.message}", file=output)
        return 1


def _bounded_command(raw_argv: list[str]) -> list[str]:
    launcher = shutil.which("systemd-run")
    if launcher is None:
        raise ConfigurationError(
            "systemd-run is unavailable; refusing an unbounded interactive model process"
        )
    return [
        launcher,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--property=MemoryHigh={CHAT_MEMORY_HIGH_BYTES}",
        f"--property=MemoryMax={CHAT_MEMORY_MAX_BYTES}",
        f"--property=MemorySwapMax={CHAT_SWAP_MAX_BYTES}",
        "--property=OOMPolicy=kill",
        "--property=CPUWeight=50",
        "--property=IOWeight=50",
        f"--setenv={_BOUNDED_CHAT_ENV}=1",
        sys.executable,
        "-m",
        "plag_in",
        *raw_argv,
    ]


def _launch_bounded_command(raw_argv: list[str]) -> int:
    result = subprocess.run(_bounded_command(raw_argv))
    return result.returncode


def _inspect_engines_report(config) -> dict:
    engines_report = {}
    for name, engine_cfg in config.engines.items():
        if engine_cfg.kind == "embedded":
            engines_report[name] = {
                "kind": "embedded",
                "library_present": Path(engine_cfg.library).is_file(),
                "declared_library_sha256": engine_cfg.library_sha256,
                "ggml_base_library_present": Path(engine_cfg.ggml_base_library).is_file(),
                "declared_ggml_base_library_sha256": engine_cfg.ggml_base_library_sha256,
                "ggml_library_present": Path(engine_cfg.ggml_library).is_file(),
                "declared_ggml_library_sha256": engine_cfg.ggml_library_sha256,
                "declared_backend_sha256": [item.sha256 for item in engine_cfg.backend_libraries],
                "upstream_identity": engine_cfg.upstream_identity,
                # The requested profile only, read from configuration; this
                # never loads the library or the model (requirement E).
                "requested_runtime_profile": config.runtime.as_embedded_request_record(),
            }
        else:
            engines_report[name] = {
                "kind": "executable",
                "executable": engine_cfg.executable,
                "executable_present": Path(engine_cfg.executable).exists(),
            }
    return engines_report


def _inspect_configured_profiles(config) -> list[dict]:
    """Inspect only the model each declared profile points at.

    This is the default inspection scope. It resolves and hashes exactly
    the configured profile models, applying the same containment and
    Ollama manifest-to-blob identity checks as `serve`, and never
    enumerates or hashes unrelated held models. A path that is missing,
    escaped by symlink or otherwise not contained is reported
    `unassessed` with its reason rather than aborting the survey; a held
    Ollama blob whose bytes do not match its manifest digest is an
    integrity failure and fails closed for the whole command.
    """
    profiles_report: list[dict] = []
    for alias, profile in config.profiles.items():
        entry: dict = {
            "alias": alias,
            "display_name": profile.display_name,
            "engine": profile.engine,
            "compatibility_status": profile.compatibility_status,
            "configured_model_path": str(profile.model_path),
        }
        try:
            identity = resolve_contained_model(
                profile.model_path, list(config.model_roots), config.ollama_manifest_root
            )
        except ModelIdentityMismatchError:
            # A held blob that does not match its manifest digest is an
            # integrity failure, not a missing model: fail closed for the
            # whole command rather than recording it as merely unassessed.
            raise
        except PathContainmentError as exc:
            # Missing, escaped or uncontained: explicitly unassessed, not
            # silently dropped and not hashed.
            entry.update(
                status="unassessed",
                reason=exc.error_type,
                detail=exc.message,
            )
            profiles_report.append(entry)
            continue
        entry.update(
            status="inspected",
            resolved_path=identity.path,
            sha256=identity.sha256,
            size_bytes=identity.size_bytes,
            quantisation="unverified",
            estimated_memory_gb=_memory_estimate_gb(identity.size_bytes),
            estimate_label="estimate",
            capability_state="unknown",
        )
        profiles_report.append(entry)
    return profiles_report


def _inspect_all_models(config) -> dict:
    gguf_models = discover_gguf(list(config.model_roots))
    gguf_report = [
        {
            "path": identity.path,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
            "quantisation": "unverified",
            "estimated_memory_gb": _memory_estimate_gb(identity.size_bytes),
            "estimate_label": "estimate",
            "capability_state": "unknown",
        }
        for identity in gguf_models
    ]

    ollama_report = []
    if config.ollama_manifest_root is not None:
        for entry in discover_ollama_models(config.ollama_manifest_root):
            ollama_report.append(
                {
                    "manifest": entry.manifest_relative_path,
                    "declared_digest": entry.declared_digest,
                    "path": entry.identity.path,
                    "sha256": entry.identity.sha256,
                    "size_bytes": entry.identity.size_bytes,
                    "quantisation": "unverified",
                    "estimated_memory_gb": _memory_estimate_gb(entry.identity.size_bytes),
                    "estimate_label": "estimate",
                    "capability_state": "unknown",
                }
            )
    return {"gguf_models": gguf_report, "ollama_models": ollama_report}


def cmd_inspect(args: argparse.Namespace) -> dict:
    config = load_config_file(Path(args.config))
    report = {
        "engines": _inspect_engines_report(config),
        "direct_runtime_profile": config.runtime.as_direct_record(),
    }
    if getattr(args, "all_models", False):
        # Operator-requested full inventory: enumerate and hash every held
        # GGUF and Ollama model under the configured roots (requirement A4).
        report["scope"] = "all_models"
        report.update(_inspect_all_models(config))
    else:
        # Default bounded scope: only the configured profile models are
        # resolved and hashed; unrelated held models are never opened.
        report["scope"] = "configured_profiles"
        report["profiles"] = _inspect_configured_profiles(config)
    return report


def cmd_recommend(args: argparse.Namespace) -> dict:
    catalogue = load_catalogue(Path(args.catalogue))
    query = RecommendationQuery(
        task=args.task,
        available_memory_gb=args.memory_gb,
        preference=args.preference,
        context_tokens=args.context_tokens,
        licence_constraint=args.licence,
    )
    options = recommend_fn(query, catalogue)
    return {"options": options}


def cmd_serve(args: argparse.Namespace) -> dict:
    config = load_config_file(Path(args.config))
    engine_cfg = config.engines.get(args.engine)
    if engine_cfg is None:
        raise ConfigurationError(f"engine not configured: {args.engine}")

    if not is_loopback(args.engine_host):
        raise LocalityPolicyError(
            "refusing a non-loopback --engine-host in this MVP", engine_host=args.engine_host
        )

    identity = resolve_contained_model(
        Path(args.model_path), list(config.model_roots), config.ollama_manifest_root
    )
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    runtime_summary = _runtime_profile_summary(config, identity, args.engine)
    runtime_summary["resource_safety"] = {
        "cgroup_memory_high_bytes": cgroup["memory.high"],
        "cgroup_memory_max_bytes": cgroup["memory.max"],
        "cgroup_swap_max_bytes": cgroup["memory.swap.max"],
        "available_ram_bytes": safety["available_ram_bytes"],
        "required_available_ram_bytes": safety["required_available_ram_bytes"],
        "gpu": safety["gpu"],
        "required_free_vram_mib": safety["required_free_vram_mib"],
    }
    runtime_receipt_profile = _runtime_receipt_profile(config, args.engine)
    _require_runtime_confirmation(args, runtime_summary)

    registry = Registry()
    config_digest = canonical_digest(
        {"engine": args.engine, "identity": engine_cfg.identity_record(), "runtime": config.runtime.as_dict()}
    )
    registry.bind(
        RegistryEntry(
            alias=args.alias,
            identity=identity,
            engine=args.engine,
            engine_config_digest=config_digest,
        )
    )
    registry.verify_identity(args.alias)

    state = StateLayout(Path(args.state_dir)).ensure()
    receipt_store = ReceiptStore(state.receipts_file, hmac_key_path=state.receipts_hmac_key_file)
    context = GatewayContext(config, registry, receipt_store, ENGINE_ADAPTERS)

    # Reserve the gateway listener before the engine starts (D): a bind
    # conflict must leave no engine process and no session record behind.
    # GatewayServer's constructor performs the actual bind and converts a
    # conflict into a typed error; nothing has been started yet if it fails.
    server = GatewayServer(context)

    if engine_cfg.kind == "embedded":
        embedded = EmbeddedLibLlama(
            library_path=engine_cfg.library,
            library_sha256=engine_cfg.library_sha256,
            ggml_base_library_path=engine_cfg.ggml_base_library,
            ggml_base_library_sha256=engine_cfg.ggml_base_library_sha256,
            ggml_library_path=engine_cfg.ggml_library,
            ggml_library_sha256=engine_cfg.ggml_library_sha256,
            backend_libraries=engine_cfg.backend_libraries,
            upstream_identity=engine_cfg.upstream_identity,
            model_path=identity.path,
            model_sha256=identity.sha256,
            runtime=config.runtime,
        )
        try:
            embedded.load()
            runtime_receipt_profile = embedded.runtime_profile
            context.register_backend(
                args.alias,
                BackendInfo(
                    engine=args.engine,
                    engine_version=engine_cfg.upstream_identity,
                    inference_mode="embedded",
                    host="127.0.0.1",
                    port=0,
                    weight_digest=identity.sha256,
                    config_digest=config_digest,
                    # Worker-era identity fields carry no embedded-library
                    # meaning: an embedded backend never populates them
                    # (handoff 07 item C). The library and native-component
                    # identity live in native_identity_digest and
                    # native_component_digests below instead.
                    executable_digest=None,
                    argv_digest=None,
                    template_digest=embedded.template_digest,
                    native_identity_digest=runtime_receipt_profile.get("native_identity_digest"),
                    native_component_digests=runtime_receipt_profile.get("native_component_digests"),
                    upstream_identity=engine_cfg.upstream_identity,
                    requested_runtime_profile=runtime_receipt_profile.get("requested"),
                    effective_runtime_profile=runtime_receipt_profile.get("effective"),
                    measurement_status=runtime_receipt_profile.get("measurement_status"),
                    gpu_offload=runtime_receipt_profile.get("gpu_offload"),
                    runtime_profile=runtime_receipt_profile,
                    inference_backend=embedded,
                ),
            )
            server.start()
        except BaseException:
            embedded.close()
            server.stop()
            raise

        startup_info = {
            "alias": args.alias,
            "engine_pid": None,
            "gateway_base_url": server.base_url,
            "locality_level": context.locality_level(),
            "runtime_profile": runtime_receipt_profile,
        }
        print(json.dumps(startup_info, sort_keys=True), flush=True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            embedded.close()
        return {"alias": args.alias, "state": "stopped"}

    if args.engine_port is None:
        server.stop()
        raise ConfigurationError("--engine-port is required for an executable engine")
    supervisor = Supervisor(state.sessions_dir)
    backend_key_path, backend_api_key = state.load_or_create_backend_api_key(args.alias)

    adapter = ENGINE_ADAPTERS[args.engine]
    argv = adapter.build_argv(
        engine_cfg.executable,
        str(identity.path),
        args.engine_host,
        args.engine_port,
        config.runtime,
        str(backend_key_path),
    )
    try:
        session = supervisor.start(
            alias=args.alias,
            argv=argv,
            allowlisted_executable=engine_cfg.executable,
            host=args.engine_host,
            port=args.engine_port,
            health_path=adapter.health_path(),
            ready_timeout_s=args.ready_timeout,
        )
    except BaseException:
        server.stop()
        raise

    try:
        context.register_backend(
            args.alias,
            BackendInfo(
                engine=args.engine,
                engine_version="unknown",
                inference_mode="worker",
                host=args.engine_host,
                port=args.engine_port,
                weight_digest=identity.sha256,
                config_digest=config_digest,
                executable_digest=session.executable_digest,
                argv_digest=session.argv_digest,
                requested_runtime_profile=runtime_receipt_profile.get("arguments"),
                measurement_status=runtime_receipt_profile.get("measurement_status"),
                gpu_offload=runtime_receipt_profile.get("gpu_offload"),
                runtime_profile=runtime_receipt_profile,
                backend_api_key=backend_api_key,
            ),
        )
        server.start()
    except BaseException:
        supervisor.stop(args.alias)
        server.stop()
        raise

    # `serve` is a foreground command: the gateway thread is a daemon and
    # would die the instant this function returns, orphaning the engine
    # child. Print the startup line immediately, then block until the
    # operator interrupts, shutting both down cleanly on the way out.
    startup_info = {
        "alias": args.alias,
        "engine_pid": session.pid,
        "gateway_base_url": server.base_url,
        "locality_level": context.locality_level(),
        "runtime_profile": runtime_receipt_profile,
    }
    print(json.dumps(startup_info, sort_keys=True), flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        supervisor.stop(args.alias)

    return {"alias": args.alias, "state": "stopped"}


def cmd_status(args: argparse.Namespace) -> dict:
    state = StateLayout(Path(args.state_dir))
    supervisor = Supervisor(state.sessions_dir)
    return supervisor.status(args.alias)


def cmd_stop(args: argparse.Namespace) -> dict:
    state = StateLayout(Path(args.state_dir))
    supervisor = Supervisor(state.sessions_dir)
    return supervisor.stop(args.alias)


def cmd_verify(args: argparse.Namespace) -> dict:
    config = load_config_file(Path(args.config))
    loopback = is_loopback(config.bind.host)

    # Configuration inspection alone never measures anything: it stays L0
    # even when the declared bind is loopback. L1 requires an actually
    # running, identity-checked loopback session (CODEX_REVIEW_MVP_2026-08-25.md F1).
    running_loopback_session = False
    if args.state_dir and args.alias:
        supervisor = Supervisor(StateLayout(Path(args.state_dir)).sessions_dir)
        status = supervisor.status(args.alias)
        record = supervisor.load_record(args.alias)
        running_loopback_session = (
            status["state"] == "running" and record is not None and is_loopback(record.host)
        )

    if running_loopback_session:
        level = "L1"
        claim = "local route observed"
    else:
        level = "L0"
        claim = "locality configured, not enforced"

    return {
        "declared_bind_host": config.bind.host,
        "measured_loopback_bind": loopback,
        "measured_running_loopback_session": running_loopback_session,
        "locality_enforcement_level": level,
        "permitted_claim": claim,
        "note": "L2 and L3 require operating-system policy and traffic observation not performed by this MVP",
        "auth_mode": config.security.auth_mode,
    }


def cmd_receipt(args: argparse.Namespace) -> dict:
    state = StateLayout(Path(args.state_dir))
    store = ReceiptStore(state.receipts_file, hmac_key_path=state.receipts_hmac_key_file)
    if args.check_chain:
        valid, count = store.verify_chain()
        return {"valid": valid, "record_count": count}
    return store.get(args.request_id)


def cmd_connect(args: argparse.Namespace) -> dict:
    api_key = args.api_key if args.api_key is not None else secrets.token_hex(16)
    text = emit(args.format, args.base_url, api_key, args.model, reveal_secret=args.reveal_secret)
    return {"format": args.format, "output": text}


def cmd_chat(args: argparse.Namespace) -> None:
    run_private_chat(
        Path(args.config),
        input,
        sys.stdout,
        state_dir=Path(args.state_dir),
        max_tokens=args.max_tokens,
    )
    return None


def cmd_gateway(args: argparse.Namespace) -> None:
    run_profile_gateway(
        Path(args.config),
        input,
        sys.stdout,
        state_dir=Path(args.state_dir),
        connector_format=args.connector_format,
    )
    return None


def cmd_doctor(args: argparse.Namespace) -> dict:
    config_path = Path(args.config) if args.config is not None else None
    return doctor_report(config_path)


def _add_common_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plag-in")
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--config", default=None)

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--config", default=None)
    p_doctor.add_argument("--format", default="human", choices=["human", "json"])
    p_doctor.set_defaults(func=cmd_doctor)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--config", required=True)
    p_inspect.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "enumerate and hash every held GGUF and Ollama model under the "
            "configured roots; the default inspects only configured profiles"
        ),
    )
    p_inspect.set_defaults(func=cmd_inspect)

    p_recommend = sub.add_parser("recommend")
    p_recommend.add_argument("--catalogue", required=True)
    p_recommend.add_argument("--task", required=True)
    p_recommend.add_argument("--memory-gb", type=float, required=True)
    p_recommend.add_argument("--preference", default="quality", choices=["speed", "quality"])
    p_recommend.add_argument("--context-tokens", type=int, default=0)
    p_recommend.add_argument("--licence", default=None)
    p_recommend.set_defaults(func=cmd_recommend)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--config", required=True)
    p_serve.add_argument("--alias", required=True)
    p_serve.add_argument("--model-path", required=True)
    p_serve.add_argument("--engine", default="llama_server")
    p_serve.add_argument("--engine-host", default="127.0.0.1")
    p_serve.add_argument("--engine-port", type=int, default=None)
    p_serve.add_argument("--ready-timeout", type=float, default=5.0)
    p_serve.add_argument(
        "--confirm-runtime-profile",
        action="store_true",
        help="confirm the displayed runtime values before loading the selected model",
    )
    _add_common_state_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--config", default=str(default_config_path()))
    p_chat.add_argument("--state-dir", default=str(default_state_path()))
    p_chat.add_argument("--max-tokens", type=int, default=256)
    p_chat.set_defaults(func=cmd_chat)

    p_gateway = sub.add_parser("gateway")
    p_gateway.add_argument("--config", default=str(default_config_path()))
    p_gateway.add_argument("--state-dir", default=str(default_state_path()))
    p_gateway.add_argument(
        "--connector-format",
        default="manual",
        choices=["manual", "openai-env", "json"],
    )
    p_gateway.set_defaults(func=cmd_gateway)

    p_status = sub.add_parser("status")
    p_status.add_argument("--alias", required=True)
    _add_common_state_args(p_status)
    p_status.set_defaults(func=cmd_status)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--alias", required=True)
    _add_common_state_args(p_stop)
    p_stop.set_defaults(func=cmd_stop)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--config", required=True)
    p_verify.add_argument("--state-dir", default=None)
    p_verify.add_argument("--alias", default=None)
    p_verify.set_defaults(func=cmd_verify)

    p_receipt = sub.add_parser("receipt")
    p_receipt.add_argument("--request-id", default=None)
    p_receipt.add_argument("--check-chain", action="store_true")
    _add_common_state_args(p_receipt)
    p_receipt.set_defaults(func=cmd_receipt)

    p_connect = sub.add_parser("connect")
    p_connect.add_argument("--base-url", required=True)
    p_connect.add_argument("--api-key", default=None)
    p_connect.add_argument("--model", required=True)
    p_connect.add_argument(
        "--format",
        dest="format",
        default="env",
        choices=["env", "manual", "openai-env", "json", "curl"],
    )
    p_connect.add_argument("--reveal-secret", action="store_true")
    p_connect.set_defaults(func=cmd_connect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        if sys.stdin.isatty() and sys.stdout.isatty():
            configured = default_config_path()
            return run_setup_assistant(
                configured if configured.is_file() else None,
                configure_fn=_setup_configure,
                chat_fn=_setup_chat,
                connect_fn=_setup_connect,
            )
        parser.print_help(file=sys.stderr)
        return 2

    args = parser.parse_args(raw_argv)
    if args.command == "setup":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(
                json.dumps(
                    {
                        "error": {
                            "type": "interactive_terminal_required",
                            "message": "plag-in setup requires an interactive terminal",
                        }
                    }
                ),
                file=sys.stderr,
            )
            return 2
        config_path = Path(args.config) if args.config is not None else None
        if config_path is None and default_config_path().is_file():
            config_path = default_config_path()
        return run_setup_assistant(
            config_path,
            configure_fn=_setup_configure,
            chat_fn=_setup_chat,
            connect_fn=_setup_connect,
        )
    try:
        if args.command in {"chat", "gateway", "serve"} and os.environ.get(_BOUNDED_CHAT_ENV) != "1":
            return _launch_bounded_command(raw_argv)
        result = args.func(args)
    except PlagInError as exc:
        print(json.dumps(exc.to_dict()), file=sys.stderr)
        return 1
    if result is None:
        return 0
    if args.command == "doctor" and args.format == "human":
        print(render_doctor(result))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
