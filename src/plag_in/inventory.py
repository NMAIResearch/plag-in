"""Read-only discovery of engines, GGUF roots and Ollama manifests.

Discovery never modifies a source and never depends on the Ollama daemon.
Files outside a declared root, including after symlink resolution, are
ignored rather than reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plag_in.identity import ModelIdentity


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
            if not digest.startswith("sha256:"):
                continue
            blob_hex = digest.split(":", 1)[1]
            blob_path = (blobs_dir / f"sha256-{blob_hex}").resolve()
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
