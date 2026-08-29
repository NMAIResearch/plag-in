import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from plag_in.config import load_config, load_config_file
from plag_in.identity import hash_file
from plag_in.setup_flow import (
    build_config,
    configure_tested_model,
    detect_tested_candidate,
    write_config,
)
from tests.support import (
    FIXTURE_COMPATIBILITY_RECORD,
    abi_profile_from_native_files,
    registered_fixture_compatibility,
)


class SetupFlowTests(unittest.TestCase):
    def _candidate(self, root: Path) -> dict:
        runtime = root / "runtime"
        models = root / "models"
        native_files = {
            "library": "libllama.so",
            "ggml_base_library": "libggml-base.so",
            "ggml_library": "libggml.so",
            "cuda_backend": "cuda/libggml-cuda.so",
            "cpu_backend": "libggml-cpu.so",
        }
        file_specs = {}
        for role, relative in native_files.items():
            path = runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture-{role}".encode())
            file_specs[role] = (relative, hash_file(path))

        model_bytes = b"fixture-model"
        model_digest = __import__("hashlib").sha256(model_bytes).hexdigest()
        blob = models / "blobs" / f"sha256-{model_digest}"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(model_bytes)
        manifest_relative = "manifests/registry.example/library/fixture/1b"
        manifest = models / manifest_relative
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": f"sha256:{model_digest}",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        bundle = {
            "upstream_identity": "fixture native bundle",
            "files": file_specs,
            "model_manifest": manifest_relative,
            "model_digest": model_digest,
            "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
            "alias": "fixture-1b",
            "display_name": "Fixture 1B",
        }
        return detect_tested_candidate(runtime, models, bundle=bundle)

    def _registered(self, candidate: dict):
        """Register the reviewed record this scratch candidate stands for."""
        return registered_fixture_compatibility(
            model_sha256=candidate["model"]["declared_sha256"],
            abi_profile=abi_profile_from_native_files(
                candidate["native_files"], candidate["upstream_identity"]
            ),
        )

    def test_exact_candidate_builds_a_valid_scoped_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._candidate(Path(tmp))
            self.assertTrue(candidate["available"])
            data = build_config(candidate, "s" * 64)
            with self._registered(candidate):
                config = load_config(data)
        self.assertEqual(set(config.profiles), {"fixture-1b"})
        self.assertEqual(config.profiles["fixture-1b"].compatibility_status, "tested")
        self.assertEqual(config.security.api_keys[0].aliases, ("fixture-1b",))
        self.assertEqual(config.bind.host, "127.0.0.1")

    def test_write_config_is_private_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root)
            target = root / "config" / "config.json"
            write_config(target, build_config(candidate, "s" * 64))
            with self._registered(candidate):
                loaded = load_config_file(target)
            mode = os.stat(target).st_mode & 0o777
        self.assertIn("fixture-1b", loaded.profiles)
        self.assertEqual(mode, 0o600)

    def test_confirmation_decline_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config.json"
            output = io.StringIO()
            result = configure_tested_model(
                input_fn=lambda _: "n",
                output=output,
                config_path=target,
                candidate=self._candidate(root),
            )
            self.assertIsNone(result)
            self.assertFalse(target.exists())
        self.assertIn("No configuration was written", output.getvalue())

    def test_confirmation_writes_the_previewed_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config.json"
            output = io.StringIO()
            candidate = self._candidate(root)
            result = configure_tested_model(
                input_fn=lambda _: "yes",
                output=output,
                config_path=target,
                candidate=candidate,
            )
            self.assertEqual(result, target.resolve())
            self.assertTrue(target.is_file())
            with self._registered(candidate):
                self.assertIn("fixture-1b", load_config_file(target).profiles)

    def test_escape_poisoned_confirmation_reprompts_then_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config.json"
            output = io.StringIO()
            answers = iter(["\x1b[Ay", "y"])
            result = configure_tested_model(
                input_fn=lambda _: next(answers),
                output=output,
                config_path=target,
                candidate=self._candidate(root),
            )
            self.assertEqual(result, target.resolve())
            self.assertTrue(target.is_file())
        self.assertIn("Input not recognised", output.getvalue())
        self.assertEqual(output.getvalue().count("Configuration written"), 1)


if __name__ == "__main__":
    unittest.main()
