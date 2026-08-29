import hashlib
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ConfigurationError, PathContainmentError
from plag_in.identity import (
    RUNTIME_SOURCE_SCOPE,
    ModelIdentity,
    _tree_digest_from_manifest,
    canonical_digest,
    compute_source_tree_digest,
    hash_file,
    runtime_source_identity,
)


class HashFileTests(unittest.TestCase):
    def test_matches_hashlib_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight.bin"
            content = b"x" * (1024 * 1024 + 37)  # spans multiple read chunks
            path.write_bytes(content)
            self.assertEqual(hash_file(path), hashlib.sha256(content).hexdigest())

    def test_full_file_hashed_not_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight.bin"
            path.write_bytes(b"a" * 10 + b"b" * 10)
            full_hash = hash_file(path)
            prefix_hash = hashlib.sha256(b"a" * 10).hexdigest()
            self.assertNotEqual(full_hash, prefix_hash)


class ModelIdentityTests(unittest.TestCase):
    def test_from_file_reports_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight.bin"
            path.write_bytes(b"hello world")
            identity = ModelIdentity.from_file(path)
            self.assertEqual(identity.size_bytes, 11)
            self.assertEqual(identity.sha256, hashlib.sha256(b"hello world").hexdigest())


class CanonicalDigestTests(unittest.TestCase):
    def test_key_order_independent(self):
        a = canonical_digest({"b": 1, "a": 2})
        b = canonical_digest({"a": 2, "b": 1})
        self.assertEqual(a, b)

    def test_differs_on_value_change(self):
        a = canonical_digest({"a": 1})
        b = canonical_digest({"a": 2})
        self.assertNotEqual(a, b)


class RuntimeSourceDigestTests(unittest.TestCase):
    def _write(self, root: Path, rel: str, content: bytes) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_manifest_order_does_not_change_the_digest(self):
        a = {"path": "a.py", "sha256": "1" * 64}
        b = {"path": "b.py", "sha256": "2" * 64}
        self.assertEqual(
            _tree_digest_from_manifest([a, b]),
            _tree_digest_from_manifest([b, a]),
        )

    def test_changing_one_source_byte_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            self._write(root, "mod.py", b"print('one')\n")
            before = compute_source_tree_digest(root)["digest"]
            self._write(root, "mod.py", b"print('two')\n")
            after = compute_source_tree_digest(root)["digest"]
        self.assertNotEqual(before, after)

    def test_equal_bytes_at_different_paths_do_not_collide(self):
        same = b"identical source bytes\n"
        with tempfile.TemporaryDirectory() as tmp:
            root_ab = Path(tmp) / "ab"
            self._write(root_ab, "a.py", same)
            self._write(root_ab, "b.py", same)
            root_ac = Path(tmp) / "ac"
            self._write(root_ac, "a.py", same)
            self._write(root_ac, "c.py", same)
            digest_ab = compute_source_tree_digest(root_ab)["digest"]
            digest_ac = compute_source_tree_digest(root_ac)["digest"]
        self.assertNotEqual(
            digest_ab, digest_ac,
            "relative path must participate so equal bytes at different paths differ",
        )

    def test_non_python_and_out_of_tree_files_do_not_change_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            self._write(root, "mod.py", b"x = 1\n")
            baseline = compute_source_tree_digest(root)["digest"]
            # A config file, model bytes and a bytecode cache inside the tree
            # must not affect the runtime source digest.
            self._write(root, "config.json", b"{}")
            self._write(root, "weights.gguf", b"\x00\x01\x02")
            self._write(root, "__pycache__/mod.cpython-312.pyc", b"cache")
            self._write(root, "__pycache__/shadow.py", b"y = 2\n")
            # A test file living outside the runtime tree must not affect it.
            self._write(Path(tmp) / "tests", "test_mod.py", b"assert True\n")
            after = compute_source_tree_digest(root)["digest"]
        self.assertEqual(baseline, after)

    def test_runtime_source_identity_scope_and_shape(self):
        identity = runtime_source_identity()
        self.assertEqual(identity["scope"], RUNTIME_SOURCE_SCOPE)
        self.assertRegex(identity["digest"], r"^[0-9a-f]{64}$")
        self.assertGreater(identity["file_count"], 0)
        # Recomputed fresh each call (no lifetime cache); stable while the
        # source tree is unchanged.
        self.assertEqual(identity, runtime_source_identity())

    def test_symlinked_source_file_to_external_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            target = outside / "real.py"
            target.write_bytes(b"x = 1\n")
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "keep.py").write_bytes(b"y = 2\n")
            (root / "link.py").symlink_to(target)
            with self.assertRaises(PathContainmentError):
                compute_source_tree_digest(root)

    def test_in_tree_symlink_is_also_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            real = root / "real.py"
            real.write_bytes(b"z = 3\n")
            (root / "alias.py").symlink_to(real)
            with self.assertRaises(PathContainmentError):
                compute_source_tree_digest(root)

    def test_symlinked_source_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "external.py").write_bytes(b"outside = True\n")
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "keep.py").write_bytes(b"inside = True\n")
            (root / "linked_package").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PathContainmentError):
                compute_source_tree_digest(root)

    def test_empty_source_tree_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            with self.assertRaises(ConfigurationError):
                compute_source_tree_digest(root)


if __name__ == "__main__":
    unittest.main()
