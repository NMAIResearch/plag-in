"""Read-only discovery of engines, GGUF roots and Ollama manifests.

Discovery never modifies a source and never depends on the Ollama daemon.
Files outside a declared root, including after symlink resolution, are
ignored rather than reported.

Availability and compatibility are separate axes (D-021). Availability is
what this module reports: whether the complete model bytes are held on
this computer. It says nothing about whether the bytes can be loaded. A
manifest whose bytes are not held is `local_incomplete` or `remote_only`,
never `unsupported`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plag_in.identity import ModelIdentity, blob_filename, is_manifest_digest

AVAILABILITY_LOCAL_COMPLETE = "local_complete"
AVAILABILITY_LOCAL_INCOMPLETE = "local_incomplete"
AVAILABILITY_REMOTE_ONLY = "remote_only"


def _within(resolved: Path, real_root: Path) -> bool:
    try:
        resolved.relative_to(real_root)
        return True
    except ValueError:
        return False


def discover_gguf(roots: list[Path]) -> list[ModelIdentity]:
    """Discover directly named GGUF files under explicitly configured roots."""
    results: list[ModelIdentity] = []
    for root in roots:
        real_root = root.resolve()
        if not real_root.is_dir():
            continue
        for candidate in sorted(real_root.rglob("*.gguf")):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if not _within(resolved, real_root):
                continue
            results.append(ModelIdentity.from_file(resolved))
    return results


@dataclass(frozen=True)
class OllamaModelEntry:
    manifest_relative_path: str
    declared_digest: str
    identity: ModelIdentity


@dataclass(frozen=True)
class OllamaModelSummary:
    manifest_relative_path: str
    declared_digest: str
    size_bytes: int | None
    blob_path: str
    held: bool


def discover_ollama_model_summaries(manifest_root: Path) -> list[OllamaModelSummary]:
    """List Ollama model manifests and held sizes without hashing weight files."""
    real_root = manifest_root.resolve()
    manifests_dir = real_root / "manifests"
    blobs_dir = real_root / "blobs"
    results: list[OllamaModelSummary] = []
    if not manifests_dir.is_dir():
        return results

    for manifest_path in sorted(manifests_dir.rglob("*")):
        if not manifest_path.is_file():
            continue
        resolved_manifest = manifest_path.resolve()
        if not _within(resolved_manifest, real_root):
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for layer in data.get("layers", []):
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            digest = layer.get("digest", "")
            if not is_manifest_digest(digest):
                continue
            blob_path = (blobs_dir / blob_filename(digest)).resolve()
            if not _within(blob_path, real_root):
                continue
            size = layer.get("size")
            results.append(
                OllamaModelSummary(
                    manifest_relative_path=str(resolved_manifest.relative_to(real_root)),
                    declared_digest=digest,
                    size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
                    blob_path=str(blob_path),
                    held=blob_path.is_file(),
                )
            )
    return results


@dataclass(frozen=True)
class HeldModel:
    """One discovered local model entry and its availability, not its support.

    `source` is `ollama_manifest` or `gguf_root`. `blob_path` is empty when
    no local bytes are bound to the entry. `reason` is empty when the entry
    is `local_complete`.
    """

    tag: str
    source: str
    availability: str
    reason: str
    blob_path: str
    declared_digest: str
    declared_size_bytes: int | None
    held_size_bytes: int | None


def _ollama_tag(relative_path: str) -> str:
    """Render a manifest path as the tag an operator would recognise."""
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return f"{parts[-2]}:{parts[-1]}"
    return relative_path


def discover_held_models(
    manifest_root: Path | None,
    gguf_roots: list[Path] | None = None,
) -> list[HeldModel]:
    """List every discoverable local entry, including non-startable ones.

    An entry whose bytes are absent stays in the list with the reason it
    cannot be started. Dropping it would report a smaller inventory than
    the computer actually declares.
    """
    results: list[HeldModel] = []
    if manifest_root is not None:
        results.extend(_discover_manifest_entries(Path(manifest_root)))
    for identity in discover_gguf(list(gguf_roots or [])):
        path = Path(identity.path)
        results.append(
            HeldModel(
                tag=path.name,
                source="gguf_root",
                availability=AVAILABILITY_LOCAL_COMPLETE,
                reason="",
                blob_path=identity.path,
                declared_digest=f"sha256:{identity.sha256}",
                declared_size_bytes=identity.size_bytes,
                held_size_bytes=identity.size_bytes,
            )
        )
    return sorted(results, key=lambda entry: (entry.source, entry.tag))


def _discover_manifest_entries(manifest_root: Path) -> list[HeldModel]:
    real_root = manifest_root.resolve()
    manifests_dir = real_root / "manifests"
    blobs_dir = real_root / "blobs"
    results: list[HeldModel] = []
    if not manifests_dir.is_dir():
        return results

    for manifest_path in sorted(manifests_dir.rglob("*")):
        if not manifest_path.is_file():
            continue
        resolved_manifest = manifest_path.resolve()
        if not _within(resolved_manifest, real_root):
            continue
        relative = str(resolved_manifest.relative_to(manifests_dir))
        tag = _ollama_tag(relative)
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            results.append(
                HeldModel(
                    tag=tag,
                    source="ollama_manifest",
                    availability=AVAILABILITY_LOCAL_INCOMPLETE,
                    reason="manifest_unreadable",
                    blob_path="",
                    declared_digest="",
                    declared_size_bytes=None,
                    held_size_bytes=None,
                )
            )
            continue

        layers = [
            layer
            for layer in data.get("layers", [])
            if isinstance(layer, dict)
            and layer.get("mediaType") == "application/vnd.ollama.image.model"
        ]
        if not layers:
            # The manifest declares no local model layer at all, so the
            # weights are served from elsewhere. Absent bytes are not an
            # unsupported model.
            results.append(
                HeldModel(
                    tag=tag,
                    source="ollama_manifest",
                    availability=AVAILABILITY_REMOTE_ONLY,
                    reason="manifest_declares_no_model_layer",
                    blob_path="",
                    declared_digest="",
                    declared_size_bytes=None,
                    held_size_bytes=None,
                )
            )
            continue

        for layer in layers:
            digest = layer.get("digest", "")
            size = layer.get("size")
            declared_size = size if isinstance(size, int) and not isinstance(size, bool) else None
            if not is_manifest_digest(digest):
                results.append(
                    HeldModel(
                        tag=tag,
                        source="ollama_manifest",
                        availability=AVAILABILITY_LOCAL_INCOMPLETE,
                        reason="manifest_digest_malformed",
                        blob_path="",
                        declared_digest=str(digest),
                        declared_size_bytes=declared_size,
                        held_size_bytes=None,
                    )
                )
                continue
            blob_path = (blobs_dir / blob_filename(digest)).resolve()
            if not _within(blob_path, real_root):
                results.append(
                    HeldModel(
                        tag=tag,
                        source="ollama_manifest",
                        availability=AVAILABILITY_LOCAL_INCOMPLETE,
                        reason="blob_path_escapes_store",
                        blob_path="",
                        declared_digest=digest,
                        declared_size_bytes=declared_size,
                        held_size_bytes=None,
                    )
                )
                continue
            if not blob_path.is_file():
                results.append(
                    HeldModel(
                        tag=tag,
                        source="ollama_manifest",
                        availability=AVAILABILITY_LOCAL_INCOMPLETE,
                        reason="declared_blob_absent",
                        blob_path=str(blob_path),
                        declared_digest=digest,
                        declared_size_bytes=declared_size,
                        held_size_bytes=None,
                    )
                )
                continue
            held_size = blob_path.stat().st_size
            if declared_size is not None and held_size != declared_size:
                results.append(
                    HeldModel(
                        tag=tag,
                        source="ollama_manifest",
                        availability=AVAILABILITY_LOCAL_INCOMPLETE,
                        reason="held_size_differs_from_manifest",
                        blob_path=str(blob_path),
                        declared_digest=digest,
                        declared_size_bytes=declared_size,
                        held_size_bytes=held_size,
                    )
                )
                continue
            results.append(
                HeldModel(
                    tag=tag,
                    source="ollama_manifest",
                    availability=AVAILABILITY_LOCAL_COMPLETE,
                    reason="",
                    blob_path=str(blob_path),
                    declared_digest=digest,
                    declared_size_bytes=declared_size,
                    held_size_bytes=held_size,
                )
            )
    return results


def discover_ollama_models(manifest_root: Path) -> list[OllamaModelEntry]:
    """Read Ollama manifests and resolve model-layer blobs without touching the daemon."""
    real_root = manifest_root.resolve()
    manifests_dir = real_root / "manifests"
    blobs_dir = real_root / "blobs"
    results: list[OllamaModelEntry] = []
    if not manifests_dir.is_dir():
        return results

    for manifest_path in sorted(manifests_dir.rglob("*")):
        if not manifest_path.is_file():
            continue
        resolved_manifest = manifest_path.resolve()
        if not _within(resolved_manifest, real_root):
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for layer in data.get("layers", []):
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            digest = layer.get("digest", "")
            if not is_manifest_digest(digest):
                continue
            blob_path = (blobs_dir / blob_filename(digest)).resolve()
            if not _within(blob_path, real_root):
                continue
            if not blob_path.is_file():
                continue
            results.append(
                OllamaModelEntry(
                    manifest_relative_path=str(resolved_manifest.relative_to(real_root)),
                    declared_digest=digest,
                    identity=ModelIdentity.from_file(blob_path),
                )
            )
    return results
