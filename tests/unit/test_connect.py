import unittest

from plag_in.connect import emit


class ConnectTests(unittest.TestCase):
    def test_default_export_redacts_secret(self):
        for fmt in ("env", "json", "curl"):
            output = emit(fmt, "http://127.0.0.1:8080/v1", "super-secret-key", "fixture-alias")
            self.assertNotIn("super-secret-key", output)

    def test_reveal_secret_includes_key(self):
        for fmt in ("env", "json", "curl"):
            output = emit(fmt, "http://127.0.0.1:8080/v1", "super-secret-key", "fixture-alias", reveal_secret=True)
            self.assertIn("super-secret-key", output)

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            emit("xml", "http://127.0.0.1:8080/v1", "key", "alias")


if __name__ == "__main__":
    unittest.main()
