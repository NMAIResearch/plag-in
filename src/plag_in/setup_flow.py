"""Confirmed Linux setup for the compatibility-tested local profile.

The setup path recognises one exact native bundle and one held model with a
completed PLAG IN trial. It never treats other installed models or libraries
as compatible from their names alone.
"""
from __future__ import annotations

import json
import os
import platform
import secrets
import tempfile
from pathlib import Path
from typing import Callable, TextIO

from plag_in.compatibility_records import REGISTERED_COMPATIBILITY_RECORDS
from plag_in.identity import hash_file
from plag_in.input_control import confirm_action
from plag_in.native_profiles import REGISTERED_NATIVE_ABI_PROFILES


_TESTED_ABI = REGISTERED_NATIVE_ABI_PROFILES[0]
# The setup path configures exactly what a reviewed record admits, so the
# model digest is read from that record rather than restated here. Two
# copies of one digest can drift; one cannot.
_TESTED_RECORD = REGISTERED_COMPATIBILITY_RECORDS[0]
_TESTED_BUNDLE = {
    "upstream_identity": _TESTED_ABI.upstream_identity,
    "files": {
        "library": (
            "libllama.so.0.0.1",
            _TESTED_ABI.library_sha256,
        ),
        "ggml_base_library": (
            "libggml-base.so.0.19.0",
            _TESTED_ABI.ggml_base_library_sha256,
        ),
        "ggml_library": (
            "libggml.so.0.19.0",
            _TESTED_ABI.ggml_library_sha256,
        ),
        "cuda_backend": (
            "cuda_v12/libggml-cuda.so",
            _TESTED_ABI.backend_library_sha256[0],
        ),
        "cpu_backend": (
            "libggml-cpu-alderlake.so",
            _TESTED_ABI.backend_library_sha256[1],
        ),
    },
    "model_manifest": "manifests/registry.ollama.ai/library/qwen2.5/3b",
    "model_digest": _TESTED_RECORD.model_sha256,
    "compatibility_record": _TESTED_RECORD.record_id,
    "alias": "qwen25-3b",
    "display_name": "Qwen2.5 3B",
}


def default_config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "plag-in" / "config.json"


def default_state_path() -> Path:
    """The state path the product chooses when the operator declares none.

    A pathname is not authority over what it reaches. An environment variable,
    or the home anchor the operating system reports, selects a pathname; it does
    not establish that a link target substituted anywhere inside that pathname
    is one the product may write to. So the default is assembled from spellings
    and every component of the result is proved afterwards by the confined walk,
    which refuses a link at any of them.

    One exception is drawn as narrowly as the host requires. The home anchor
    itself, and nothing below it, is canonicalised, because a host may reach the
    user's home through a compatibility link: on this one `/home` is a link to
    `/var/home`, so the product's own default would otherwise be refused by the
    product's own rule with nothing the operator could do about it. The anchor
    is what the operating system reports as the user's home, and translating it
    changes no decision the operator made. The `.local`, `state` and `plag-in`
    components are appended to the translated anchor as spellings and are never
    resolved.

    A non-empty `XDG_STATE_HOME` is used without symbolic-link resolution: no
    component of it is followed, and the walk refuses any link inside it. The
    spelling itself is still subject to ordinary path normalisation, so a
    trailing separator or a `..` component is normalised rather than preserved
    byte for byte. Canonicalising the whole root followed a link at `.local`, at
    `state` or at any component of that variable and handed the walk the
    selected target, which has no linked component left to refuse (independent
    review of the C1-R1 repair, C1-R2).

    An empty `XDG_STATE_HOME` selects the home fallback rather than the
    variable. Taking an empty value as supplied would select a bare relative
    `plag-in` beneath whatever the working directory happens to be, which is not
    a state location the product should choose for itself (independent review of
    the C1-R2 repair, R2-F1).
    """
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "plag-in"
    home_anchor = Path(os.path.realpath(str(Path.home())))
    return home_anchor / ".local" / "state" / "plag-in"


def detect_tested_candidate(
    runtime_root: Path = Path("/usr/local/lib/ollama"),
    manifest_root: Path | None = None,
    *,
    bundle: dict | None = None,
) -> dict:
    """Match the held files to the exact completed CPU and GPU trial identity."""
    spec = bundle or _TESTED_BUNDLE
    manifest_root = manifest_root or (Path.home() / ".ollama" / "models")
    result = {
        "available": False,
        "platform_status": "matched" if platform.system() == "Linux" else "unsupported",
        "native_files": {},
        "model": {"status": "unavailable"},
        "alias": spec["alias"],
        "display_name": spec["display_name"],
        "compatibility_record": spec.get("compatibility_record"),
    }
    if result["platform_status"] != "matched":
        return result

    native_ok = True
    for role, (relative_path, expected_digest) in spec["files"].items():
        path = (runtime_root / relative_path).resolve()
        present = path.is_file()
        observed_digest = hash_file(path) if present else None
        matched = present and observed_digest == expected_digest
        result["native_files"][role] = {
            "path": str(path),
            "expected_sha256": expected_digest,
            "observed_sha256": observed_digest,
            "matched": matched,
        }
        native_ok = native_ok and matched

    manifest_path = (manifest_root / spec["model_manifest"]).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return result
    model_layer = next(
        (
            layer
            for layer in manifest.get("layers", [])
            if layer.get("mediaType") == "application/vnd.ollama.image.model"
        ),
        None,
    )
    if model_layer is None:
        return result
    declared_digest = str(model_layer.get("digest", ""))
    if declared_digest != f"sha256:{spec['model_digest']}":
        result["model"] = {"status": "manifest_digest_mismatch"}
        return result
    model_path = (manifest_root / "blobs" / f"sha256-{spec['model_digest']}").resolve()
    model_present = model_path.is_file()
    result["model"] = {
        "status": "held" if model_present else "blob_missing",
        "path": str(model_path),
        "declared_sha256": spec["model_digest"],
        "size_bytes": model_path.stat().st_size if model_present else None,
        "complete_hash_status": "deferred_until_model_start",
    }
    result["available"] = native_ok and model_present
    result["runtime_root"] = str(runtime_root.resolve())
    result["manifest_root"] = str(manifest_root.resolve())
    result["upstream_identity"] = spec["upstream_identity"]
    return result


def build_config(candidate: dict, api_secret: str) -> dict:
    """Build the local tested profile after candidate identity matching."""
    if not candidate.get("available"):
        raise ValueError("tested local candidate is unavailable")
    if not candidate.get("compatibility_record"):
        raise ValueError(
            "tested local candidate carries no reviewed compatibility record"
        )
    native = candidate["native_files"]
    alias = candidate["alias"]
    return {
        "model_roots": [],
        "ollama_manifest_root": candidate["manifest_root"],
        "engines": {
            "libllama": {
                "library": native["library"]["path"],
                "library_sha256": native["library"]["observed_sha256"],
                "ggml_base_library": native["ggml_base_library"]["path"],
                "ggml_base_library_sha256": native["ggml_base_library"]["observed_sha256"],
                "ggml_library": native["ggml_library"]["path"],
                "ggml_library_sha256": native["ggml_library"]["observed_sha256"],
                "backend_libraries": [
                    {
                        "path": native["cuda_backend"]["path"],
                        "sha256": native["cuda_backend"]["observed_sha256"],
                    },
                    {
                        "path": native["cpu_backend"]["path"],
                        "sha256": native["cpu_backend"]["observed_sha256"],
                    },
                ],
                "upstream_identity": candidate["upstream_identity"],
            }
        },
        "profiles": {
            alias: {
                "model_path": candidate["model"]["path"],
                "engine": "libllama",
                "display_name": candidate["display_name"],
                "compatibility_status": "tested",
                "compatibility_record": candidate["compatibility_record"],
            }
        },
        "bind": {"host": "127.0.0.1", "port": 19080},
        "security": {
            "auth_mode": "api_key",
            "api_keys": [
                {
                    "id": "local-operator",
                    "secret": api_secret,
                    "aliases": [alias],
                    "origin": "loopback",
                }
            ],
            "content_logging": False,
        },
        "runtime": {
            "context_size": 2048,
            "gpu_layers": "all",
            "threads": 8,
            "batch_size": 512,
            "ubatch_size": 128,
            "parallel": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def write_config(path: Path, data: dict) -> None:
    """Write one confirmed configuration atomically with private permissions."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def configure_tested_model(
    *,
    input_fn: Callable[[str], str],
    output: TextIO,
    config_path: Path | None = None,
    candidate: dict | None = None,
) -> Path | None:
    """Preview and write the exact tested profile after terminal confirmation."""
    candidate = candidate or detect_tested_candidate()
    if not candidate.get("available"):
        print(
            "No exact compatibility-tested local model and native bundle matched this computer.\n"
            "No configuration was written. Other held models remain unverified.",
            file=output,
        )
        return None

    size_gib = candidate["model"]["size_bytes"] / (1024**3)
    target = (config_path or default_config_path()).resolve()
    print(
        f"Detected tested profile: {candidate['display_name']}\n"
        f"  Held model size: {size_gib:.2f} GiB\n"
        "  Inference: embedded libllama with matched CUDA and CPU backends\n"
        "  Bind: loopback only\n"
        "  Authentication: generated local API key\n"
        "  Content logging: off\n"
        "  Remote fallback: absent\n"
        "  Runtime: context 2048, GPU layers requested all, threads 8, batch 512\n"
        "  Exact GPU layer count: unassessed\n"
        f"  Configuration target: {target}\n"
        f"  Existing target: {'yes' if target.exists() else 'no'}",
        file=output,
    )
    prompt = "Replace this configuration? [y/N] " if target.exists() else "Write this configuration? [y/N] "
    if not confirm_action(input_fn, output, prompt):
        print("No configuration was written.", file=output)
        return None

    data = build_config(candidate, secrets.token_hex(32))
    write_config(target, data)
    print(f"Configuration written: {target}", file=output)
    print("Next: choose Start a private test chat from the setup menu.", file=output)
    return target
