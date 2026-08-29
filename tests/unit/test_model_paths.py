"""Targeted Ollama blob containment and identity tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ModelIdentityMismatchError, PathContainmentError
from plag_in.model_paths import (
    resolve_contained_model,
    verify_model_unchanged,
    witness_model_file,
)


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


class ContainmentRefusalTests(unittest.TestCase):
    """Probe 5: relative, absolute, symlink and out-of-root selection refuse."""

    def _root(self, tmp: Path) -> Path:
        root = tmp / "models"
        root.mkdir()
        (root / "inside.gguf").write_bytes(b"GGUF-inside")
        return root

    def test_absolute_path_outside_every_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = self._root(tmp_path)
            outside = tmp_path / "outside.gguf"
            outside.write_bytes(b"GGUF-outside")
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(outside, [root], None)

    def test_relative_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = self._root(tmp_path)
            (tmp_path / "outside.gguf").write_bytes(b"GGUF-outside")
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(root / ".." / "outside.gguf", [root], None)

    def test_symlink_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = self._root(tmp_path)
            target = tmp_path / "outside.gguf"
            target.write_bytes(b"GGUF-outside")
            link = root / "escape.gguf"
            link.symlink_to(target)
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(link, [root], None)

    def test_contained_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            identity = resolve_contained_model(root / "inside.gguf", [root], None)
            self.assertEqual(identity.sha256, hashlib.sha256(b"GGUF-inside").hexdigest())


class ModelFileWitnessTests(unittest.TestCase):
    """Probe 4: a file replaced between inspection and load refuses the load."""

    def test_unchanged_file_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF-original")
            verify_model_unchanged(witness_model_file(model))

    def test_replacement_by_a_different_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF-original")
            witness = witness_model_file(model)
            replacement = root / "other.gguf"
            replacement.write_bytes(b"GGUF-replaced-same-length")
            replacement.replace(model)
            with self.assertRaises(ModelIdentityMismatchError):
                verify_model_unchanged(witness)

    def test_rewritten_contents_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF-original")
            witness = witness_model_file(model)
            model.write_bytes(b"GGUF-different")
            with self.assertRaises(ModelIdentityMismatchError):
                verify_model_unchanged(witness)

    def test_removed_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF-original")
            witness = witness_model_file(model)
            model.unlink()
            with self.assertRaises(ModelIdentityMismatchError):
                verify_model_unchanged(witness)

    def test_a_symbolic_link_is_not_a_witnessable_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.gguf"
            target.write_bytes(b"GGUF-target")
            link = root / "link.gguf"
            link.symlink_to(target)
            with self.assertRaises(ModelIdentityMismatchError):
                witness_model_file(link)


if __name__ == "__main__":
    unittest.main()
