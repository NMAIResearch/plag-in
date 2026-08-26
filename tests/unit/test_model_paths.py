"""Targeted Ollama blob containment and identity tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ModelIdentityMismatchError, PathContainmentError
from plag_in.model_paths import resolve_contained_model


class OllamaBlobResolutionTests(unittest.TestCase):
    def _store(self, root: Path, content: bytes = b"GGUF-model") -> tuple[Path, Path]:
        digest = hashlib.sha256(content).hexdigest()
        blob = root / "blobs" / f"sha256-{digest}"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(content)
        manifest = root / "manifests" / "registry.ollama.ai" / "library" / "fixture" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": f"sha256:{digest}",
                            "size": len(content),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return blob, manifest

    def test_only_requested_manifest_bound_blob_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob, _manifest = self._store(root)
            identity = resolve_contained_model(blob, [], root)
            self.assertEqual(identity.path, str(blob.resolve()))
            self.assertEqual(identity.size_bytes, len(b"GGUF-model"))

    def test_unreferenced_blob_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob, _manifest = self._store(root)
            other = root / "blobs" / ("sha256-" + "0" * 64)
            other.write_bytes(b"unreferenced")
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(other, [], root)

    def test_manifest_bound_blob_with_mutated_bytes_fails_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob, _manifest = self._store(root)
            blob.write_bytes(b"mutated")
            with self.assertRaises(ModelIdentityMismatchError):
                resolve_contained_model(blob, [], root)


if __name__ == "__main__":
    unittest.main()
