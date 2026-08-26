import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from plag_in.inventory import discover_gguf, discover_ollama_models


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


if __name__ == "__main__":
    unittest.main()
