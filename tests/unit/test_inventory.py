import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plag_in.inventory import (
    AVAILABILITY_LOCAL_COMPLETE,
    AVAILABILITY_LOCAL_INCOMPLETE,
    AVAILABILITY_REMOTE_ONLY,
    discover_gguf,
    discover_held_models,
    discover_ollama_model_summaries,
    discover_ollama_models,
)


class DiscoverGgufTests(unittest.TestCase):
    def test_discovers_files_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            (root / "a.gguf").write_bytes(b"weights-a")
            (root / "not-a-model.txt").write_bytes(b"ignore me")
            results = discover_gguf([root])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].path.endswith("a.gguf"))
            self.assertEqual(results[0].sha256, hashlib.sha256(b"weights-a").hexdigest())

    def test_ignores_files_outside_declared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "secret.gguf").write_bytes(b"should-not-appear")

            root = Path(tmp) / "declared_root"
            root.mkdir()

            results = discover_gguf([root])
            self.assertEqual(results, [])

    def test_symlink_escape_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "secret.gguf").write_bytes(b"escaped-weights")

            root = Path(tmp) / "declared_root"
            root.mkdir()
            symlink = root / "escape.gguf"
            symlink.symlink_to(outside / "secret.gguf")

            results = discover_gguf([root])
            self.assertEqual(results, [], "a symlink resolving outside the declared root must be ignored")

    def test_missing_root_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_root = Path(tmp) / "does_not_exist"
            self.assertEqual(discover_gguf([missing_root]), [])


class DiscoverOllamaModelsTests(unittest.TestCase):
    def _build_manifest_root(self, tmp: Path) -> tuple[Path, bytes]:
        root = Path(tmp) / "ollama_root"
        manifests_dir = root / "manifests" / "registry.ollama.ai" / "library" / "fixture-model"
        blobs_dir = root / "blobs"
        manifests_dir.mkdir(parents=True)
        blobs_dir.mkdir(parents=True)

        blob_content = b"ollama-fixture-weights"
        blob_digest = hashlib.sha256(blob_content).hexdigest()
        (blobs_dir / f"sha256-{blob_digest}").write_bytes(blob_content)

        manifest = {
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{blob_digest}", "size": len(blob_content)},
                {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:deadbeef", "size": 3},
            ]
        }
        (manifests_dir / "latest").write_text(json.dumps(manifest))
        return root, blob_content

    def test_reads_manifest_and_resolves_blob_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, blob_content = self._build_manifest_root(Path(tmp))
            before = (root / "manifests" / "registry.ollama.ai" / "library" / "fixture-model" / "latest").read_text()

            results = discover_ollama_models(root)

            after = (root / "manifests" / "registry.ollama.ai" / "library" / "fixture-model" / "latest").read_text()
            self.assertEqual(before, after, "manifest must not be modified by discovery")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].identity.sha256, hashlib.sha256(blob_content).hexdigest())

    def test_missing_manifest_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(discover_ollama_models(Path(tmp) / "absent"), [])

    def test_summary_lists_held_model_without_hashing_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, blob_content = self._build_manifest_root(Path(tmp))
            with patch(
                "plag_in.inventory.ModelIdentity.from_file",
                side_effect=AssertionError("summary discovery must not hash weights"),
            ):
                summaries = discover_ollama_model_summaries(root)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(summaries[0].held)
            self.assertEqual(summaries[0].size_bytes, len(blob_content))

    def test_summary_keeps_missing_blob_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _blob_content = self._build_manifest_root(Path(tmp))
            for blob in (root / "blobs").iterdir():
                blob.unlink()
            summaries = discover_ollama_model_summaries(root)
            self.assertEqual(len(summaries), 1)
            self.assertFalse(summaries[0].held)


class HeldModelAvailabilityTests(unittest.TestCase):
    """Probe 3: incomplete and remote-only entries stay visible with a reason."""

    def _manifest(self, root: Path, name: str, layers: list[dict], tag: str = "latest") -> Path:
        # The real store nests <registry>/<namespace>/<model>/<tag>, and the
        # reported tag is the last two components of that path.
        path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
        return path

    def _blob(self, root: Path, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        blob = root / "blobs" / f"sha256-{digest}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        return digest

    def _model_layer(self, digest: str, size: int | None) -> dict:
        layer = {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}
        if size is not None:
            layer["size"] = size
        return layer

    def test_held_blob_is_local_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"complete-model-bytes"
            digest = self._blob(root, content)
            self._manifest(root, "held", [self._model_layer(digest, len(content))])
            entries = discover_held_models(root, [])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].tag, "held:latest")
            self.assertEqual(entries[0].availability, AVAILABILITY_LOCAL_COMPLETE)
            self.assertEqual(entries[0].reason, "")
            self.assertEqual(entries[0].held_size_bytes, len(content))

    def test_manifest_without_a_model_layer_is_remote_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blobs").mkdir(parents=True)
            self._manifest(root, "cloud", [])
            entries = discover_held_models(root, [])
            self.assertEqual(entries[0].availability, AVAILABILITY_REMOTE_ONLY)
            self.assertEqual(entries[0].reason, "manifest_declares_no_model_layer")
            self.assertEqual(entries[0].blob_path, "")

    def test_absent_blob_is_local_incomplete_and_not_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blobs").mkdir(parents=True)
            self._manifest(root, "missing", [self._model_layer("ab" * 32, 10)])
            entries = discover_held_models(root, [])
            self.assertEqual(entries[0].availability, AVAILABILITY_LOCAL_INCOMPLETE)
            self.assertEqual(entries[0].reason, "declared_blob_absent")

    def test_held_size_disagreeing_with_the_manifest_is_local_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"short"
            digest = self._blob(root, content)
            self._manifest(root, "partial", [self._model_layer(digest, len(content) + 500)])
            entries = discover_held_models(root, [])
            self.assertEqual(entries[0].availability, AVAILABILITY_LOCAL_INCOMPLETE)
            self.assertEqual(entries[0].reason, "held_size_differs_from_manifest")

    def test_unreadable_manifest_is_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blobs").mkdir(parents=True)
            path = root / "manifests" / "registry.ollama.ai" / "library" / "broken" / "latest"
            path.parent.mkdir(parents=True)
            path.write_text("{not json", encoding="utf-8")
            entries = discover_held_models(root, [])
            self.assertEqual(entries[0].reason, "manifest_unreadable")

    def test_gguf_roots_join_the_same_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            (models / "local.gguf").write_bytes(b"gguf-bytes")
            entries = discover_held_models(None, [models])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].source, "gguf_root")
            self.assertEqual(entries[0].availability, AVAILABILITY_LOCAL_COMPLETE)

    def test_every_manifest_appears_exactly_once_per_model_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"complete-model-bytes"
            digest = self._blob(root, content)
            self._manifest(root, "held", [self._model_layer(digest, len(content))])
            self._manifest(root, "cloud", [])
            self._manifest(root, "missing", [self._model_layer("cd" * 32, 4)])
            entries = discover_held_models(root, [])
            self.assertEqual(
                sorted(entry.tag for entry in entries),
                ["cloud:latest", "held:latest", "missing:latest"],
            )


if __name__ == "__main__":
    unittest.main()
