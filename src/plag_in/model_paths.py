"""Model path containment for `serve`.

A path handed to `serve` must resolve inside a configured model root, or
match a blob bound by a held Ollama manifest and digest. A symlink that
resolves outside the permitted set is rejected the same way a direct path
outside it would be (CODEX_REVIEW_MVP_2026-08-25.md F7).
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from plag_in.errors import ModelIdentityMismatchError, PathContainmentError
from plag_in.identity import ModelIdentity, is_manifest_digest, manifest_digest_hex


@dataclass(frozen=True)
class ModelFileWitness:
    """What file the inspected bytes came from, taken at inspection time.

    Inspection and native load are separate steps, so the file named at the
    first can be replaced before the second. The complete-byte digest is
    re-verified by the embedded engine immediately before any native call;
    this witness closes the narrower gap where the same path names a
    different file whose length happens to match.
    """

    path: str
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int


def witness_model_file(path: Path | str) -> ModelFileWitness:
    """Record the file identity of the model that was inspected."""
    resolved = Path(path)
    try:
        info = os.lstat(resolved)
    except OSError as exc:
        raise ModelIdentityMismatchError(
            "the selected model file could not be inspected",
            requested_path=str(resolved),
            reason=str(exc),
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise ModelIdentityMismatchError(
            "the selected model path is not a regular file",
            requested_path=str(resolved),
        )
    return ModelFileWitness(
        path=str(resolved),
        device=info.st_dev,
        inode=info.st_ino,
        size_bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
    )


def verify_model_unchanged(witness: ModelFileWitness) -> None:
    """Refuse when the inspected file is no longer the file at that path."""
    current = witness_model_file(witness.path)
    if current != witness:
        raise ModelIdentityMismatchError(
            "the selected model file changed after it was inspected; no model was started",
            requested_path=witness.path,
            inspected_inode=witness.inode,
            observed_inode=current.inode,
            inspected_size_bytes=witness.size_bytes,
            observed_size_bytes=current.size_bytes,
        )


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
    declared = f"sha256:{filename[len('sha256-'):]}"
    if not is_manifest_digest(declared):
        return None
    declared_hex = manifest_digest_hex(declared)

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
