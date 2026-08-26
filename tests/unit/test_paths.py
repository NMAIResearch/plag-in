import stat
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ConfigurationError
from plag_in.paths import StateLayout


class BackendApiKeyStateTests(unittest.TestCase):
    def test_backend_key_is_private_stable_and_alias_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateLayout(Path(tmp) / "state").ensure()
            path_a, secret_a = state.load_or_create_backend_api_key("model-a")
            path_b, secret_b = state.load_or_create_backend_api_key("model-a")

            self.assertEqual(path_a, path_b)
            self.assertEqual(secret_a, secret_b)
            self.assertEqual(len(secret_a), 64)
            self.assertEqual(stat.S_IMODE(path_a.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(state.engine_keys_dir.stat().st_mode), 0o700)

    def test_malformed_existing_backend_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateLayout(Path(tmp) / "state").ensure()
            path = state.backend_api_key_file("model-a")
            path.write_text("short\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                state.load_or_create_backend_api_key("model-a")


if __name__ == "__main__":
    unittest.main()
