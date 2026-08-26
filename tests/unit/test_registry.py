import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ModelIdentityMismatchError
from plag_in.identity import ModelIdentity
from plag_in.registry import Registry, RegistryEntry


class RegistryTests(unittest.TestCase):
    def test_resolve_unknown_alias_raises(self):
        registry = Registry()
        with self.assertRaises(ModelIdentityMismatchError):
            registry.resolve("missing")

    def test_mutation_invalidates_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"original-weights")
            identity = ModelIdentity.from_file(path)

            registry = Registry()
            registry.bind(RegistryEntry(alias="a1", identity=identity, engine="llama_server", engine_config_digest="d1"))

            # Unmodified: verification passes.
            registry.verify_identity("a1")

            # Mutate the file in place.
            path.write_bytes(b"tampered-weights-different-length")
            with self.assertRaises(ModelIdentityMismatchError) as ctx:
                registry.verify_identity("a1")
            self.assertNotEqual(ctx.exception.fields["expected"], ctx.exception.fields["actual"])

    def test_aliases_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry()
            for name in ("zeta", "alpha", "mid"):
                path = Path(tmp) / f"{name}.gguf"
                path.write_bytes(name.encode())
                registry.bind(
                    RegistryEntry(
                        alias=name,
                        identity=ModelIdentity.from_file(path),
                        engine="llama_server",
                        engine_config_digest="d",
                    )
                )
            self.assertEqual(registry.aliases(), ["alpha", "mid", "zeta"])


if __name__ == "__main__":
    unittest.main()
