"""Operator CLI: inspect, recommend, serve, stop, status, verify, receipt, connect.

There is deliberately no `download` command in this MVP (D-001, PRODUCT_SPEC
section 6): recommendation never triggers a network action, and serving
never triggers an implicit download.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from plag_in.active_store import ActiveReceiptStore
from plag_in.compatibility_records import find_compatibility_record
from plag_in.config import is_loopback, load_config_file, with_trial_profile
from plag_in.conformance import check_conformance
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
from plag_in.gguf_metadata import UNASSESSED, read_gguf_metadata
from plag_in.identity import canonical_digest
from plag_in.input_control import (
    confirm_action,
    select_menu,
    select_number,
    terminal_menu_available,
)
from plag_in.inventory import (
    AVAILABILITY_LOCAL_COMPLETE,
    HeldModel,
    discover_gguf,
    discover_held_models,
    discover_ollama_models,
)
from plag_in.model_paths import (
    resolve_contained_model,
    verify_model_unchanged,
    witness_model_file,
)
from plag_in.onboarding import doctor_report, render_doctor, run_setup_assistant
from plag_in.paths import StateLayout
from plag_in.receipts import (
    CURRENT_STORE_FILENAME,
    RECEIPT_SCHEMA_VERSION,
    ReceiptStore,
    initialise_current_store,
    rollback_current_store,
)
from plag_in.recommend import RecommendationQuery, load_catalogue, recommend as recommend_fn
from plag_in.registry import Registry, RegistryEntry
from plag_in.resource_safety import (
    CHAT_MEMORY_HIGH_BYTES,
    CHAT_MEMORY_MAX_BYTES,
    CHAT_SWAP_MAX_BYTES,
    evaluate_model_admission,
    resource_preflight,
    verify_chat_cgroup,
)
from plag_in.setup_flow import (
    configure_tested_model,
    default_config_path,
    default_state_path,
    detect_tested_candidate,
)
from plag_in.supervisor import Supervisor

ENGINE_ADAPTERS: dict[str, EngineAdapter] = {
    "llama_server": LlamaServerAdapter(),
    "libllama": LibLlamaAdapter(),
}

_MEMORY_OVERHEAD_FACTOR = 1.2  # rough runtime overhead multiplier; always labelled an estimate
_BOUNDED_CHAT_ENV = "PLAG_IN_BOUNDED_CHAT"
_TRIAL_ALIAS_PREFIX = "trial-"
_UNSAFE_ALIAS_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# What the private test chat tells the model about itself, built only from
# values verified on this computer. It is scoped to the private test-chat
# conversation and never enters a request from another harness. It carries
# no provider name: naming one would state an origin PLAG IN has not
# verified from the model's own local metadata.
_IDENTITY_INSTRUCTION_TEMPLATE = (
    "You are being run locally by PLAG IN under the local alias {alias}. "
    "The local model file has SHA-256 {sha256}. Its GGUF metadata declares "
    "architecture {architecture} and context length {context_length}. "
    "If you are asked which model you are, report exactly this local identity. "
    "Do not claim a different model, product or provider identity."
)


def _trial_alias_for(tag: str) -> str:
    """Derive a valid alias from a local tag without inventing identity."""
    cleaned = _UNSAFE_ALIAS_CHARS.sub("-", tag).strip("-") or "model"
    return (_TRIAL_ALIAS_PREFIX + cleaned)[:64]


@dataclass(frozen=True)
class _InventoryRow:
    """One selectable line of the held-model inventory."""

    label: str
    startable: bool
    reason: str
    compatibility_status: str
    alias: str | None = None
    profile: object | None = None
    entry: HeldModel | None = None


def _held_size_text(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "size unassessed"
    return f"{size_bytes / (1024**3):.2f} GiB"


def _inventory_rows(config, manifest_root: Path | None) -> list[_InventoryRow]:
    """Build the full held inventory with configured profiles first.

    Sizes and availability for held entries come from manifests and file
    lengths, so listing them hashes no weight file. A profile whose stored
    label is `tested` is the one exception: the menu offers it as a tested
    start, so the bytes behind that offer are resolved and hashed here and
    the state shown comes from that derivation, never from the stored label
    (independent review of the general model admission repair, F1-R1). A
    profile whose bytes are absent or uncontained stays on the list carrying
    the reason it cannot be started. Nothing is dropped.
    """
    rows: list[_InventoryRow] = []
    configured_paths: dict[str, tuple[str, object]] = {
        str(profile.model_path): (alias, profile) for alias, profile in config.profiles.items()
    }

    for alias, profile in sorted(config.profiles.items()):
        if profile.compatibility_status != "tested":
            continue
        try:
            identity = resolve_contained_model(
                profile.model_path, list(config.model_roots), config.ollama_manifest_root
            )
        except PlagInError as exc:
            rows.append(
                _InventoryRow(
                    label=(
                        f"{profile.display_name} ({alias}) | configured profile | "
                        f"not startable: {exc.error_type}"
                    ),
                    startable=False,
                    reason=exc.error_type,
                    compatibility_status="unverified",
                    alias=alias,
                    profile=profile,
                )
            )
            continue
        status, disclosure = derive_compatibility_state(profile, identity)
        rows.append(
            _InventoryRow(
                label=(
                    f"{profile.display_name} ({alias}) | configured profile | "
                    f"available: local_complete | compatibility: "
                    f"{_compatibility_text(status, disclosure)}"
                ),
                startable=True,
                reason=disclosure.get("reason", ""),
                compatibility_status=status,
                alias=alias,
                profile=profile,
            )
        )

    held = discover_held_models(manifest_root, list(config.model_roots))
    complete, incomplete = [], []
    for entry in held:
        if entry.blob_path and entry.blob_path in configured_paths:
            alias, profile = configured_paths[entry.blob_path]
            if profile.compatibility_status == "tested":
                continue
        (complete if entry.availability == AVAILABILITY_LOCAL_COMPLETE else incomplete).append(entry)

    for entry in complete:
        rows.append(
            _InventoryRow(
                label=(
                    f"{entry.tag} | {_held_size_text(entry.held_size_bytes)} | "
                    f"available: {entry.availability} | compatibility: unverified"
                ),
                startable=True,
                reason="",
                compatibility_status="unverified",
                alias=_trial_alias_for(entry.tag),
                entry=entry,
            )
        )
    for entry in incomplete:
        rows.append(
            _InventoryRow(
                label=(
                    f"{entry.tag} | {_held_size_text(entry.held_size_bytes)} | "
                    f"available: {entry.availability} | not startable: {entry.reason}"
                ),
                startable=False,
                reason=entry.reason,
                compatibility_status="unverified",
                entry=entry,
            )
        )
    return rows


def _select_inventory_row(
    rows: list[_InventoryRow],
    input_fn: Callable[[str], str],
    output: TextIO,
    heading: str,
) -> _InventoryRow | None:
    """Select one inventory line with Up, Down, Enter and q."""
    if not rows:
        return None
    labels = [row.label for row in rows]
    print(heading, file=output)
    if terminal_menu_available(input_fn, output):
        selection = select_menu(
            labels,
            output,
            "Use Up and Down, then press Enter. Press q to return.",
        )
    else:
        for index, label in enumerate(labels, 1):
            print(f"  {index}. {label}", file=output)
        selection = select_number(
            input_fn, output, f"Select 1-{len(labels)}: ", minimum=1, maximum=len(labels)
        )
    if selection is None:
        return None
    return rows[selection - 1]


def _manifest_root_for(config) -> Path | None:
    if config.ollama_manifest_root is not None:
        return config.ollama_manifest_root
    default_root = Path.home() / ".ollama" / "models"
    return default_root if default_root.is_dir() else None


def _resolve_trial_entry(config, trial_model: str) -> HeldModel:
    """Find one held entry by tag or by held path, without starting anything."""
    held = discover_held_models(_manifest_root_for(config), list(config.model_roots))
    for entry in held:
        if trial_model in (entry.tag, entry.blob_path):
            return entry
    raise ConfigurationError(
        "the requested trial model is not present in the local inventory",
        trial_model=trial_model,
    )


def _assess_trial_admission(config, entry: HeldModel, engine_name: str) -> dict:
    """Decide whether one held model may receive a bounded unverified trial.

    Availability, static compatibility and host resources are assessed as
    separate axes and each refusal keeps its own reason, so an absent blob
    is never reported as an unsupported model and a resource refusal is
    never reported as an incompatible one (D-021).
    """
    record: dict = {
        "tag": entry.tag,
        "source": entry.source,
        "availability": entry.availability,
        "compatibility": "unverified",
        "admission": "refused",
        "reason": entry.reason,
        "identity": None,
        "metadata": None,
        "resources": None,
        "requested_runtime": config.runtime.as_dict(),
    }
    if entry.availability != AVAILABILITY_LOCAL_COMPLETE:
        record["reason"] = entry.reason or "model bytes are not held on this computer"
        return record

    try:
        identity = resolve_contained_model(
            Path(entry.blob_path), list(config.model_roots), config.ollama_manifest_root
        )
    except (PathContainmentError, ModelIdentityMismatchError) as exc:
        record["reason"] = exc.error_type
        record["detail"] = exc.message
        return record
    record["identity"] = {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

    try:
        metadata = read_gguf_metadata(identity.path)
    except ConfigurationError as exc:
        record["compatibility"] = "unsupported"
        record["reason"] = exc.fields.get("reason", "gguf_inspection_failed")
        record["detail"] = exc.message
        return record
    record["metadata"] = metadata.as_dict()
    if not metadata.chat_template_present:
        # A chat request needs a chat template. Without one the embedded
        # engine has nothing to apply, and this is a property of the file,
        # not of the host.
        record["compatibility"] = "unsupported"
        record["reason"] = "gguf_declares_no_chat_template"
        return record

    resources = evaluate_model_admission(identity.size_bytes, config.runtime.gpu_layers)
    record["resources"] = resources
    if resources["outcome"] != "admitted":
        record["reason"] = resources["outcome"]
        return record

    record["admission"] = "approved"
    record["reason"] = ""
    record["engine"] = engine_name
    return record


def _print_trial_disclosure(record: dict, alias: str, output: TextIO) -> None:
    """Show every value the operator is being asked to confirm."""
    metadata = record["metadata"]
    identity = record["identity"]
    resources = record["resources"]
    preflight = resources["preflight"]
    unassessed = ", ".join(metadata["unassessed_fields"]) or "none"
    print(
        f"\nUnverified local model trial\n"
        f"  Local tag: {record['tag']}\n"
        f"  Trial alias: {alias}\n"
        f"  Model path: {identity['path']}\n"
        f"  Complete SHA-256: {identity['sha256']}\n"
        f"  Held size: {_held_size_text(identity['size_bytes'])}\n"
        f"  GGUF version: {metadata['gguf_version']}\n"
        f"  Architecture: {metadata['architecture']}\n"
        f"  Declared context length: {metadata['context_length']}\n"
        f"  Chat template present: {str(metadata['chat_template_present']).lower()}\n"
        f"  Quantisation identifier: {metadata['file_type_id']} "
        f"(label: {metadata['file_type_label']})\n"
        f"  Unassessed metadata: {unassessed}\n"
        f"  Requested runtime: {json.dumps(record['requested_runtime'], sort_keys=True)}\n"
        f"  Cgroup memory maximum: {resources['cgroup']['memory.max'] // (1024**2)} MiB\n"
        f"  Available RAM: {preflight['available_ram_bytes'] // (1024**2)} MiB\n"
        f"  Required available RAM: "
        f"{preflight['required_available_ram_bytes'] // (1024**2)} MiB\n"
        f"  Resource policy: {resources['policy_validation_status']}\n"
        "  This is an unverified local trial. It does not promote this model to tested,\n"
        "  and it writes no persistent configuration.",
        file=output,
    )
    if preflight["gpu"] is not None:
        print(
            f"  Free VRAM: {preflight['gpu']['free_mib']} MiB\n"
            f"  Required free VRAM: {preflight['required_free_vram_mib']} MiB",
            file=output,
        )


def _identity_instruction(alias: str, record: dict) -> str:
    """Build the private test-chat identity instruction from verified metadata."""
    metadata = record["metadata"] or {}
    return _IDENTITY_INSTRUCTION_TEMPLATE.format(
        alias=alias,
        sha256=(record["identity"] or {}).get("sha256", UNASSESSED),
        architecture=metadata.get("architecture", UNASSESSED),
        context_length=metadata.get("context_length", UNASSESSED),
    )


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


def derive_compatibility_state(profile, identity) -> tuple[str, dict]:
    """The one derivation of compatibility from resolved model bytes.

    Every route that has resolved a model calls this and carries what it
    returns. No surface downstream of resolution reads
    `profile.compatibility_status`: the configuration label selects a
    candidate, and the bytes decide what is reported, registered and
    recorded (independent review of the general model admission repair,
    F1-R1). A profile whose stored label is `tested` but whose bytes are not
    the recorded bytes is reported `unverified` with both digests, and a
    record withdrawn from the reviewed source demotes the profile rather
    than letting a stored label outlive its evidence.
    """
    def _evidence(status: str, reason: str, record_id, recorded, **extra) -> dict:
        # One shape on every outcome. A surface that reports compatibility
        # reports what was compared, so a reader never has to infer whether
        # a digest is absent because it matched or because no record exists
        # (independent review of repair 2, F1-R3).
        return {
            "status": status,
            "reason": reason,
            "compatibility_record": record_id,
            "recorded_model_sha256": recorded,
            "observed_model_sha256": identity.sha256,
            **extra,
        }

    if profile is None:
        return "unverified", _evidence(
            "unverified",
            "no_configured_profile_for_this_model_path_and_engine",
            None,
            None,
        )
    if profile.compatibility_status != "tested":
        return "unverified", _evidence(
            "unverified", "configured_profile_is_not_tested", None, None
        )

    record = (
        find_compatibility_record(profile.compatibility_record)
        if profile.compatibility_record
        else None
    )
    if record is None:
        return "unverified", _evidence(
            "unverified",
            "reviewed_compatibility_record_absent",
            profile.compatibility_record,
            None,
        )
    if identity.sha256 != record.model_sha256:
        return "unverified", _evidence(
            "unverified",
            "model_digest_does_not_match_reviewed_record",
            record.record_id,
            record.model_sha256,
        )
    return "tested", _evidence(
        "tested",
        "",
        record.record_id,
        record.model_sha256,
        decision_id=record.decision_id,
        trial_report=record.trial_report,
        trial_report_sha256=record.trial_report_sha256,
        native_abi_profile_id=record.native_abi_profile_id,
        record_binding_digest=record.binding_digest(),
    )


def _compatibility_text(status: str, disclosure: dict) -> str:
    """The compatibility line an operator reads before confirming a load."""
    if disclosure.get("reason"):
        return f"{status} ({disclosure['reason']})"
    return status


def _compatibility_disclosure_lines(disclosure: dict) -> str:
    """The digests a compatibility claim rests on, for operator output."""
    recorded = disclosure.get("recorded_model_sha256") or "none recorded"
    return (
        f"  Compatibility record: {disclosure.get('compatibility_record') or 'none'}\n"
        f"  Recorded model SHA-256: {recorded}\n"
        f"  Observed model SHA-256: {disclosure.get('observed_model_sha256')}\n"
    )


def _serve_compatibility_state(config, identity, engine_name: str) -> tuple[str, dict]:
    """Derive the state for a model path handed straight to `serve`.

    Serving a path directly is `unverified` unless that exact path and
    engine are a configured profile, and a request that succeeds never
    promotes it (D-021).
    """
    profile = next(
        (
            candidate
            for candidate in config.profiles.values()
            if str(candidate.model_path) == identity.path and candidate.engine == engine_name
        ),
        None,
    )
    return derive_compatibility_state(profile, identity)


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
    receipts: ActiveReceiptStore

    def close(self) -> None:
        self.server.stop()
        self.embedded.close()


def _start_embedded_session(
    config,
    alias: str,
    profile,
    state_dir: Path,
    *,
    identity=None,
) -> _EmbeddedSession:
    """Start one embedded session for an already-selected profile.

    The registration sink derives its own compatibility state from the
    profile and the identity it is about to register. There is deliberately
    no parameter for that state: a sink that accepted one could be handed a
    value conflicting with its own resolved identity, which would recreate
    the asserted-`tested` defect one layer inside the product rather than in
    configuration (independent review of repair 2, F1-R2).

    A caller that has already resolved the model passes the identity it
    resolved, so the weights are read once. Re-deriving from an identity in
    hand reads nothing further.
    """
    engine_cfg = config.engines[profile.engine]
    if engine_cfg.kind != "embedded":
        raise ConfigurationError("private test chat requires an embedded engine")
    if identity is None:
        identity = resolve_contained_model(
            profile.model_path, list(config.model_roots), config.ollama_manifest_root
        )
    compatibility, compatibility_evidence = derive_compatibility_state(profile, identity)
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
    # New receipts go to whichever store is active at the moment of the write:
    # the original store until a current-schema store has been initialised
    # beside it, that store from activation onwards. Resolving once here bound
    # a running session to the store it started with (second pass, D2).
    receipts = ActiveReceiptStore(state)
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
                compatibility_status=compatibility,
                compatibility_evidence=compatibility_evidence,
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
    if len(tested) == 1 and not terminal_menu_available(input_fn, output):
        return tested[0]
    labels = [f"{profile.display_name} ({alias})" for alias, profile in tested]
    if terminal_menu_available(input_fn, output):
        selection = select_menu(
            labels,
            output,
            "Tested local profiles. Use Up and Down, then press Enter.",
        )
    else:
        print("Tested local profiles", file=output)
        for index, label in enumerate(labels, 1):
            print(f"  {index}. {label}", file=output)
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


def _legacy_receipt_store_requires_current(state: StateLayout) -> bool:
    """Return whether the original store contains a pre-current record."""
    snapshot = state.active_store_snapshot()
    if not snapshot.is_legacy:
        return False
    try:
        info = os.lstat(snapshot.store)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ConfigurationError(
            "the legacy receipt store could not be inspected",
            path=str(snapshot.store),
            reason=str(exc),
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ConfigurationError(
            "the legacy receipt store is not a regular file",
            path=str(snapshot.store),
        )
    if info.st_size == 0:
        return False
    try:
        fd = os.open(str(snapshot.store), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ConfigurationError(
            "the legacy receipt store could not be opened",
            path=str(snapshot.store),
            reason=str(exc),
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ConfigurationError(
                "the legacy receipt store is not a regular file",
                path=str(snapshot.store),
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            for raw in handle:
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return True
                if not isinstance(record, dict):
                    return True
                if record.get("schema_version") != RECEIPT_SCHEMA_VERSION:
                    return True
    finally:
        if fd >= 0:
            os.close(fd)
    return False


def _initialise_receipt_state(state: StateLayout, destination: Path | None = None) -> dict:
    destination = destination or state.base / CURRENT_STORE_FILENAME
    destination = state.validate_active_store_path(destination)
    created = initialise_current_store(destination)
    try:
        pointer = state.write_receipt_pointer(destination, created["schema_version"])
    except BaseException:
        rollback_current_store(created)
        raise
    created.pop("created_identities", None)
    return {
        "initialised": created,
        "active_pointer": pointer,
        "receipt_stores": state.receipt_store_roles(),
        "legacy_store_modified": False,
    }


def _prepare_receipt_state(
    state_dir: Path,
    input_fn: Callable[[str], str],
    output: TextIO,
) -> bool:
    """Activate a separate current store before loading a model when required."""
    state = StateLayout(state_dir)
    if not _legacy_receipt_store_requires_current(state):
        return True
    destination = state.base / CURRENT_STORE_FILENAME
    affected = (
        destination,
        destination.with_name(destination.name + ".hmac_key"),
        destination.with_name(destination.name + ".checkpoint"),
        destination.with_name(destination.name + ".lock"),
        state.base / "receipts.init.lock",
        state.receipt_pointer_file,
    )
    print(
        "Receipt migration required before inference.\n"
        f"  Historical store preserved: {state.legacy_receipts_file}\n"
        "  New current-schema state:\n"
        + "\n".join(f"    {path}" for path in affected),
        file=output,
    )
    if not confirm_action(
        input_fn,
        output,
        "Create and activate the separate current-schema receipt store? [y/N] ",
    ):
        print("No listener or model was started.", file=output)
        return False
    result = _initialise_receipt_state(state)
    print(
        f"Current-schema receipt store active: {result['active_pointer']['current_store']}\n"
        "Historical receipt bytes were not modified.",
        file=output,
    )
    return True


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


def _prepare_trial(
    config,
    trial_model: str,
    trial_alias: str | None,
    input_fn: Callable[[str], str],
    output: TextIO,
):
    """Admit or refuse one unverified held model, without persisting anything.

    Returns `(config, alias, profile, record)` on approval. A refusal
    returns `None` after printing the measured reason, and starts nothing.
    """
    entry = _resolve_trial_entry(config, trial_model)
    engine_name = next(
        (name for name, engine in config.engines.items() if engine.kind == "embedded"), None
    )
    if engine_name is None:
        raise ConfigurationError("an unverified local trial requires a configured embedded engine")

    record = _assess_trial_admission(config, entry, engine_name)
    if record["admission"] != "approved":
        print(
            f"\nUnverified local model trial refused\n"
            f"  Local tag: {record['tag']}\n"
            f"  Availability: {record['availability']}\n"
            f"  Compatibility: {record['compatibility']}\n"
            f"  Reason: {record['reason']}\n"
            f"  Detail: {record.get('detail', 'none')}\n"
            f"  Measurements: {json.dumps(record['resources'], sort_keys=True)}\n"
            "  No listener, engine or model was started.",
            file=output,
        )
        return None

    alias = trial_alias or _trial_alias_for(entry.tag)
    trial_config = with_trial_profile(
        config,
        alias,
        Path(record["identity"]["path"]),
        engine_name,
        entry.tag,
    )
    _print_trial_disclosure(record, alias, output)
    if not confirm_action(
        input_fn,
        output,
        "Load this unverified local model for a bounded trial? [y/N] ",
    ):
        print("No listener or model was started.", file=output)
        return None
    return trial_config, alias, trial_config.profiles[alias], record


def run_private_chat(
    config_path: Path,
    input_fn: Callable[[str], str],
    output: TextIO,
    *,
    state_dir: Path | None = None,
    max_tokens: int = 256,
    trial_model: str | None = None,
    trial_alias: str | None = None,
) -> int:
    """Start one confirmed embedded model and stop it when the chat closes."""
    config = load_config_file(config_path)
    trial_record = None
    if trial_model is not None:
        prepared = _prepare_trial(config, trial_model, trial_alias, input_fn, output)
        if prepared is None:
            return 0
        config, alias, profile, trial_record = prepared
    else:
        alias, profile = _select_tested_profile(config, input_fn, output)
    identity = resolve_contained_model(
        profile.model_path, list(config.model_roots), config.ollama_manifest_root
    )
    witness = witness_model_file(identity.path)
    compatibility, compatibility_disclosure = derive_compatibility_state(profile, identity)
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    summary = _runtime_profile_summary(config, identity, profile.engine)
    # Never resolved. `Path.resolve()` follows every symbolic link in the
    # path, so the confined walk received the link's target and had nothing
    # left to refuse: a state path declared through a link was silently
    # replaced by whatever it named (independent review of the C1 repair,
    # C1-R1). A path the operator declares is used exactly as declared.
    effective_state = Path(state_dir) if state_dir is not None else default_state_path()
    print(
        f"\nPrivate test chat\n"
        f"  Model: {profile.display_name}\n"
        f"  Alias: {alias}\n"
        f"  Compatibility: {_compatibility_text(compatibility, compatibility_disclosure)}\n"
        f"{_compatibility_disclosure_lines(compatibility_disclosure)}"
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
    if not _prepare_receipt_state(effective_state, input_fn, output):
        return 0
    if not confirm_action(
        input_fn,
        output,
        "Load this model and start the private test chat? [y/N] ",
    ):
        print("No listener or model was started.", file=output)
        return 0

    # Inspection and load are separate steps. This refuses the case where the
    # path still names a file but no longer names the file whose metadata and
    # resource decision the operator just confirmed. The complete byte digest
    # is re-verified again by the embedded engine before any native call.
    verify_model_unchanged(witness)
    session = _start_embedded_session(
        config,
        alias,
        profile,
        effective_state,
        identity=identity,
    )
    api_key = _profile_api_key(config, alias)
    messages: list[dict] = []
    if trial_record is not None:
        # Scoped to this conversation. It is not injected into a request
        # from any other harness, and what the model answers about itself
        # remains its own claim, not identity evidence.
        messages.append(
            {"role": "system", "content": _identity_instruction(alias, trial_record)}
        )
    print(
        f"Local chat ready at {session.server.base_url}. Type /exit to stop the model.\n"
        "A model's description of itself is not identity evidence. The verified local\n"
        f"identity is alias {alias}, SHA-256 {identity.sha256}.",
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
    integrity_valid, count = session.receipts.verify_chain()
    print(
        f"Model stopped. Receipt chain integrity valid: {str(integrity_valid).lower()} "
        f"(scope: chain, checkpoint and canonical storage; not specification conformance). "
        f"Requests recorded: {count}.",
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
    trial_model: str | None = None,
    trial_alias: str | None = None,
) -> int:
    """Run one bounded confirmed profile for an existing local harness."""
    config = load_config_file(config_path)
    if trial_model is not None:
        prepared = _prepare_trial(config, trial_model, trial_alias, input_fn, output)
        if prepared is None:
            return 0
        config, alias, profile, _record = prepared
    else:
        alias, profile = _select_tested_profile(config, input_fn, output)
    identity = resolve_contained_model(
        profile.model_path, list(config.model_roots), config.ollama_manifest_root
    )
    witness = witness_model_file(identity.path)
    compatibility, compatibility_disclosure = derive_compatibility_state(profile, identity)
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    # Never resolved. `Path.resolve()` follows every symbolic link in the
    # path, so the confined walk received the link's target and had nothing
    # left to refuse: a state path declared through a link was silently
    # replaced by whatever it named (independent review of the C1 repair,
    # C1-R1). A path the operator declares is used exactly as declared.
    effective_state = Path(state_dir) if state_dir is not None else default_state_path()
    gpu_text = (
        f"  Free VRAM: {safety['gpu']['free_mib']} MiB\n"
        if safety["gpu"] is not None else "  GPU route: disabled\n"
    )
    print(
        f"\nHarness gateway\n"
        f"  Model: {profile.display_name}\n"
        f"  Alias: {alias}\n"
        f"  Compatibility: {_compatibility_text(compatibility, compatibility_disclosure)}\n"
        f"{_compatibility_disclosure_lines(compatibility_disclosure)}"
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
    if not _prepare_receipt_state(effective_state, input_fn, output):
        return 0
    if not confirm_action(input_fn, output, "Load this model for an existing harness? [y/N] "):
        print("No listener or model was started.", file=output)
        return 0

    # Refuses a file replaced between inspection and load, as in the private
    # test chat. No identity instruction is added here: the private test chat
    # scopes one to its own conversation, and harness traffic is unchanged.
    verify_model_unchanged(witness)
    session = _start_embedded_session(
        config,
        alias,
        profile,
        effective_state,
        identity=identity,
    )
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
    integrity_valid, count = session.receipts.verify_chain()
    print(
        f"Model stopped. Receipt chain integrity valid: {str(integrity_valid).lower()} "
        f"(scope: chain, checkpoint and canonical storage; not specification conformance). "
        f"Requests recorded: {count}.",
        file=output,
    )
    return 0


_INVENTORY_HEADING = (
    "\nLocal model inventory\n"
    "  Every discovered entry is listed. Names and sizes come from manifests and\n"
    "  file lengths, so listing does not hash a weight file.\n"
    "  Availability is whether the bytes are held here. Compatibility is whether\n"
    "  PLAG IN has an exact reviewed profile for them. They are separate.\n"
    "  Only an exact compatibility-tested model is configured automatically. Any\n"
    "  other held model may receive a bounded unverified trial, which writes no\n"
    "  configuration and promotes nothing."
)


def _setup_configure(active_config: Path | None, input_fn, output) -> Path | None:
    manifest_root = Path.home() / ".ollama" / "models"
    configured = None
    if active_config is not None and active_config.is_file():
        configured = load_config_file(active_config)
        if configured.ollama_manifest_root is not None:
            manifest_root = configured.ollama_manifest_root

    rows = (
        _inventory_rows(configured, manifest_root)
        if configured is not None
        else [
            _InventoryRow(
                label=(
                    f"{entry.tag} | {_held_size_text(entry.held_size_bytes)} | "
                    f"available: {entry.availability} | compatibility: unverified"
                    if entry.availability == AVAILABILITY_LOCAL_COMPLETE
                    else f"{entry.tag} | available: {entry.availability} | "
                    f"not startable: {entry.reason}"
                ),
                startable=entry.availability == AVAILABILITY_LOCAL_COMPLETE,
                reason=entry.reason,
                compatibility_status="unverified",
                alias=_trial_alias_for(entry.tag),
                entry=entry,
            )
            for entry in discover_held_models(manifest_root, [])
        ]
    )

    candidate = detect_tested_candidate(manifest_root=manifest_root)
    if rows:
        selected = _select_inventory_row(rows, input_fn, output, _INVENTORY_HEADING)
        if selected is None:
            return active_config
        if not selected.startable:
            print(
                f"This entry cannot be started: {selected.reason}.\n"
                "Its bytes are not held complete on this computer, which is not the "
                "same as the model being unsupported. No configuration was written.",
                file=output,
            )
            return active_config
        if selected.compatibility_status != "tested":
            tested_digest = (
                f"sha256:{candidate['model']['declared_sha256']}"
                if candidate.get("available") else None
            )
            entry = selected.entry
            if entry is None or entry.declared_digest != tested_digest:
                print(
                    "Selected model is held locally but has not passed the exact PLAG IN "
                    "compatibility trial. No configuration was written.\n"
                    "  Run a bounded unverified trial instead, from Start a private test "
                    "chat, or with:\n"
                    f"    plag-in chat --trial-model {entry.tag if entry else '<tag>'}\n"
                    "  A trial changes no configuration and promotes no model to tested.",
                    file=output,
                )
                return active_config
    return configure_tested_model(
        input_fn=input_fn,
        output=output,
        config_path=active_config or default_config_path(),
        candidate=candidate,
    )


def _setup_chat(config_path: Path, input_fn, output) -> int:
    try:
        config = load_config_file(config_path)
    except PlagInError as exc:
        print(f"Chat did not start: {exc.message}", file=output)
        return 1
    rows = [
        row
        for row in _inventory_rows(config, _manifest_root_for(config))
        if row.startable
    ]
    argv = ["chat", "--config", str(config_path)]
    if rows:
        selected = _select_inventory_row(rows, input_fn, output, _INVENTORY_HEADING)
        if selected is None:
            print("No model was started.", file=output)
            return 0
        if selected.compatibility_status != "tested" and selected.entry is not None:
            argv += ["--trial-model", selected.entry.tag, "--trial-alias", selected.alias]
    try:
        return _launch_bounded_command(argv)
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
        # The configured label is reported as what it is, a configuration
        # value. `compatibility_status` is the derived state, and until the
        # bytes are resolved nothing has been established about them
        # (independent review of the general model admission repair, F1-R1).
        entry: dict = {
            "alias": alias,
            "display_name": profile.display_name,
            "engine": profile.engine,
            "configured_compatibility_status": profile.compatibility_status,
            "compatibility_status": "unverified",
            "compatibility_reason": "model_bytes_not_resolved",
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
        derived, disclosure = derive_compatibility_state(profile, identity)
        entry.update(
            status="inspected",
            resolved_path=identity.path,
            sha256=identity.sha256,
            size_bytes=identity.size_bytes,
            quantisation="unverified",
            estimated_memory_gb=_memory_estimate_gb(identity.size_bytes),
            estimate_label="estimate",
            capability_state="unknown",
            compatibility_status=derived,
            compatibility_reason=disclosure.get("reason", ""),
            compatibility_record=disclosure.get("compatibility_record"),
            compatibility_evidence=disclosure,
            recorded_model_sha256=disclosure.get("recorded_model_sha256"),
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
    compatibility_status, compatibility_disclosure = _serve_compatibility_state(
        config, identity, args.engine
    )
    cgroup, safety = _model_load_safety(identity, config.runtime.gpu_layers)
    runtime_summary = _runtime_profile_summary(config, identity, args.engine)
    runtime_summary["compatibility"] = compatibility_disclosure
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
    # Follows the pointer for the life of the process (second pass, D2): a
    # store activated while this gateway serves takes effect on the next
    # receipt, with no restart and no documented stop-before-initialise step.
    receipt_store = ActiveReceiptStore(state)
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
                    compatibility_status=compatibility_status,
                    compatibility_evidence=compatibility_disclosure,
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
                compatibility_status=compatibility_status,
                compatibility_evidence=compatibility_disclosure,
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
    status = dict(supervisor.status(args.alias))
    # Both receipt stores and their roles, always: an operator must be able to
    # see that a legacy store still exists and is no longer written to, rather
    # than infer it from a line that is not there.
    status["receipt_stores"] = state.receipt_store_roles()
    return status


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

    if args.initialise_current_store:
        destination = Path(args.store_path) if args.store_path else state.base / CURRENT_STORE_FILENAME
        return _initialise_receipt_state(state, destination)

    # One pointer read for all three paths: a check that read the pointer once
    # per path could assess a store against another store's key (second pass, D3).
    active = state.active_store_snapshot()
    store_path = active.store
    key_path = active.hmac_key
    checkpoint_path = active.checkpoint

    if args.check_conformance:
        # The current store only: a legacy record is evidence under its own
        # historical rules and is not conformance evidence for this schema.
        result = check_conformance(store_path, key_path, checkpoint_path)
        result["store_path"] = str(store_path)
        result["receipt_stores"] = state.receipt_store_roles()
        return result

    store = ReceiptStore(store_path, hmac_key_path=key_path)
    if args.check_chain:
        # Storage integrity, not specification conformance: this establishes
        # that the HMAC chain and the authenticated checkpoint agree over the
        # stored records. It is deliberately not reported as `valid`, because a
        # caller reading that word would take it for conformance, which this
        # scope does not assess. Run `--check-conformance` for that.
        integrity_valid, count = store.verify_chain()
        return {
            "integrity_valid": integrity_valid,
            "record_count": count,
            "verification_scope": "chain_checkpoint_storage",
            "store_path": str(store_path),
            "receipt_stores": state.receipt_store_roles(),
        }
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
        trial_model=args.trial_model,
        trial_alias=args.trial_alias,
    )
    return None


def cmd_gateway(args: argparse.Namespace) -> None:
    run_profile_gateway(
        Path(args.config),
        input,
        sys.stdout,
        state_dir=Path(args.state_dir),
        connector_format=args.connector_format,
        trial_model=args.trial_model,
        trial_alias=args.trial_alias,
    )
    return None


def cmd_doctor(args: argparse.Namespace) -> dict:
    config_path = Path(args.config) if args.config is not None else None
    return doctor_report(config_path)


def _add_common_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", required=True)


def _add_trial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-model",
        default=None,
        help=(
            "run one bounded unverified trial of a held local model, named by its "
            "local tag or held path; nothing is written to the configuration and "
            "no model is promoted to tested"
        ),
    )
    parser.add_argument(
        "--trial-alias",
        default=None,
        help="alias for the trial profile; derived from the local tag when absent",
    )


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
    _add_trial_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_gateway = sub.add_parser("gateway")
    p_gateway.add_argument("--config", default=str(default_config_path()))
    p_gateway.add_argument("--state-dir", default=str(default_state_path()))
    p_gateway.add_argument(
        "--connector-format",
        default="manual",
        choices=["manual", "openai-env", "json"],
    )
    _add_trial_args(p_gateway)
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

    p_receipt = sub.add_parser(
        "receipt",
        description=(
            "Read or check inference receipts. A receipt associates a request "
            "identifier with an exact weight digest and optional declared source "
            "and runtime metadata under a local HMAC. It is emitted alongside a "
            "response. It does not cryptographically bind response content: the "
            "schema carries no response digest and no authenticated transport "
            "transcript."
        ),
    )
    p_receipt.add_argument(
        "--request-id",
        default=None,
        help="print the metadata receipt for one request identifier",
    )
    p_receipt.add_argument(
        "--check-chain",
        action="store_true",
        help=(
            "check storage integrity only: the HMAC chain, the authenticated "
            "checkpoint and canonical storage. This is not a conformance result"
        ),
    )
    p_receipt.add_argument(
        "--check-conformance",
        action="store_true",
        help=(
            "run every mandatory Inference Receipt Specification rule against the "
            "current store and report the conformance level assessed"
        ),
    )
    p_receipt.add_argument(
        "--initialise-current-store",
        action="store_true",
        help=(
            "create a fresh current-schema receipt store beside an existing one; "
            "the existing store and its checkpoint are left byte-identical"
        ),
    )
    p_receipt.add_argument(
        "--store-path",
        default=None,
        help=(
            f"destination for --initialise-current-store; must be {CURRENT_STORE_FILENAME} "
            "directly beneath the state directory, and is refused if it already exists, "
            "is a symbolic link, or names any other path"
        ),
    )
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
