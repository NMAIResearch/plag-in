"""Model path containment for `serve`.

A path handed to `serve` must resolve inside a configured model root, or
match a blob bound by a held Ollama manifest and digest. A symlink that
resolves outside the permitted set is rejected the same way a direct path
outside it would be (CODEX_REVIEW_MVP_2026-08-25.md F7).
"""
from __future__ import annotations

import json
from pathlib import Path

from plag_in.errors import ModelIdentityMismatchError, PathContainmentError
from plag_in.identity import ModelIdentity


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_held_ollama_blob(requested: Path, store_root: Path) -> ModelIdentity | None:
    real_root = Path(store_root).resolve()
    manifests_dir = real_root / "manifests"
    blobs_dir = real_root / "blobs"
    if not manifests_dir.is_dir() or not blobs_dir.is_dir():
        return None
    if not _within(requested, blobs_dir):
        return None
    filename = requested.name
    if not filename.startswith("sha256-"):
        return None
    declared_hex = filename[len("sha256-"):]
    if len(declared_hex) != 64 or any(char not in "0123456789abcdef" for char in declared_hex):
        return None

    declared = f"sha256:{declared_hex}"
    found = False
    for manifest_path in sorted(manifests_dir.rglob("*")):
        if not manifest_path.is_file():
            continue
        resolved_manifest = manifest_path.resolve()
        if not _within(resolved_manifest, real_root):
            continue
        try:
            data = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for layer in data.get("layers", []):
            if (
                layer.get("mediaType") == "application/vnd.ollama.image.model"
                and layer.get("digest") == declared
            ):
                found = True
                break
        if found:
            break
    if not found or not requested.is_file():
        return None

    identity = ModelIdentity.from_file(requested)
    if identity.sha256 != declared_hex:
        raise ModelIdentityMismatchError(
            "held Ollama model blob does not match its manifest digest",
            expected_sha256=declared_hex,
            actual_sha256=identity.sha256,
        )
    return identity


def resolve_contained_model(
    requested_path: Path,
    model_roots: list[Path],
    ollama_manifest_root: Path | None,
) -> ModelIdentity:
    resolved = Path(requested_path).resolve()

    for root in model_roots:
        real_root = Path(root).resolve()
        if not real_root.is_dir():
            continue
        try:
            resolved.relative_to(real_root)
        except ValueError:
            continue
        if resolved.is_file():
            return ModelIdentity.from_file(resolved)

    if ollama_manifest_root is not None:
        identity = _resolve_held_ollama_blob(resolved, ollama_manifest_root)
        if identity is not None:
            return identity

    raise PathContainmentError(
        "model path is not contained by a configured model root or a held Ollama manifest blob",
        requested_path=str(resolved),
    )
