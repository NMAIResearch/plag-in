import unittest

from plag_in.config import load_config
from plag_in.errors import ConfigurationError, UnknownConfigurationFieldError


def _base_config(**overrides):
    data = {
        "model_roots": ["/tmp/models"],
        "bind": {"host": "127.0.0.1", "port": 8080},
        "security": {"auth_mode": "none"},
    }
    data.update(overrides)
    return data


class LoadConfigTests(unittest.TestCase):
    def test_valid_loopback_config(self):
        config = load_config(_base_config())
        self.assertEqual(config.bind.host, "127.0.0.1")
        self.assertEqual(config.security.auth_mode, "none")

    def test_defaults_bind_to_loopback(self):
        config = load_config({})
        self.assertEqual(config.bind.host, "127.0.0.1")
        self.assertEqual(config.runtime.context_size, 8192)
        self.assertEqual(config.runtime.gpu_layers, "auto")
        self.assertEqual(config.runtime.parallel, 1)

    def test_unknown_top_level_field_rejected(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(_base_config(nonsense_field=True))

    def test_unknown_security_field_rejected(self):
        data = _base_config()
        data["security"]["mystery_toggle"] = True
        with self.assertRaises(UnknownConfigurationFieldError) as ctx:
            load_config(data)
        self.assertEqual(ctx.exception.fields["section"], "security")
        self.assertIn("mystery_toggle", ctx.exception.fields["fields"])

    def test_unknown_bind_field_rejected(self):
        data = _base_config()
        data["bind"]["backdoor"] = "yes"
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(data)

    def test_unknown_api_key_field_rejected(self):
        data = _base_config()
        data["security"]["auth_mode"] = "api_key"
        data["security"]["api_keys"] = [{"id": "k1", "secret": "s1", "admin_override": True}]
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(data)

    def test_non_loopback_requires_auth(self):
        data = _base_config(bind={"host": "0.0.0.0", "port": 8080})
        with self.assertRaises(ConfigurationError):
            load_config(data)

    def test_non_loopback_with_api_key_accepted(self):
        data = _base_config(
            bind={"host": "0.0.0.0", "port": 8080},
            security={"auth_mode": "api_key", "api_keys": [{"id": "k1", "secret": "s1"}]},
        )
        config = load_config(data)
        self.assertEqual(config.bind.host, "0.0.0.0")
        self.assertEqual(len(config.security.api_keys), 1)

    def test_explicit_runtime_profile_is_loaded(self):
        config = load_config(
            _base_config(
                runtime={
                    "context_size": 4096,
                    "gpu_layers": "all",
                    "threads": 6,
                    "batch_size": 1024,
                    "ubatch_size": 256,
                    "parallel": 2,
                    "temperature": 0.2,
                    "top_p": 0.9,
                }
            )
        )
        self.assertEqual(config.runtime.context_size, 4096)
        self.assertEqual(config.runtime.gpu_layers, "all")
        self.assertEqual(config.runtime.threads, 6)
        self.assertEqual(config.runtime.batch_size, 1024)
        self.assertEqual(config.runtime.ubatch_size, 256)
        self.assertEqual(config.runtime.parallel, 2)

    def test_unknown_runtime_field_fails_closed(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(_base_config(runtime={"hidden_default": True}))

    def test_invalid_runtime_relationship_fails_closed(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"batch_size": 256, "ubatch_size": 512}))

    def test_embedded_engine_requires_complete_hash_bound_identity(self):
        config = load_config(
            _base_config(
                engines={
                    "libllama": {
                        "library": "/opt/lib/libllama.so",
                        "library_sha256": "a" * 64,
                        "ggml_base_library": "/opt/lib/libggml-base.so",
                        "ggml_base_library_sha256": "d" * 64,
                        "ggml_library": "/opt/lib/libggml.so",
                        "ggml_library_sha256": "b" * 64,
                        "backend_libraries": [{"path": "/opt/lib/libggml-cpu.so", "sha256": "c" * 64}],
                        "upstream_identity": "llama.cpp-b10380",
                    }
                }
            )
        )
        self.assertEqual(config.engines["libllama"].kind, "embedded")
        self.assertEqual(config.engines["libllama"].backend_libraries[0].sha256, "c" * 64)
        self.assertEqual(config.engines["libllama"].ggml_base_library_sha256, "d" * 64)

    def test_embedded_engine_missing_identity_fails_closed(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"libllama": {"library": "/opt/lib/libllama.so"}}))

    def test_embedded_engine_missing_ggml_base_library_fails_closed(self):
        with self.assertRaises(ConfigurationError) as ctx:
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": "/opt/lib/libllama.so",
                            "library_sha256": "a" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "b" * 64,
                            "backend_libraries": [
                                {"path": "/opt/lib/libggml-cpu.so", "sha256": "c" * 64}
                            ],
                            "upstream_identity": "llama.cpp-b10380",
                        }
                    }
                )
            )
        self.assertIn("ggml_base_library", str(ctx.exception))

    def test_unknown_engine_field_rejected(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(
                _base_config(
                    engines={"mystery": {"executable": "/bin/false", "extra_field": True}}
                )
            )

    def test_unknown_backend_library_field_rejected(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": "/opt/lib/libllama.so",
                            "library_sha256": "a" * 64,
                            "ggml_base_library": "/opt/lib/libggml-base.so",
                            "ggml_base_library_sha256": "d" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "b" * 64,
                            "backend_libraries": [
                                {"path": "/opt/lib/libggml-cpu.so", "sha256": "c" * 64, "extra": True}
                            ],
                            "upstream_identity": "llama.cpp-b10380",
                        }
                    }
                )
            )

    def test_engine_cannot_mix_executable_and_library(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "mixed": {
                            "executable": "/bin/false",
                            "library": "/opt/lib/libllama.so",
                        }
                    }
                )
            )

    def test_profile_binds_alias_to_configured_engine_and_model(self):
        data = _base_config(
            engines={"worker": {"executable": "/bin/false"}},
            profiles={
                "fixture-model": {
                    "model_path": "/tmp/models/fixture.gguf",
                    "engine": "worker",
                    "display_name": "Fixture model",
                    "compatibility_status": "tested",
                }
            },
        )
        config = load_config(data)
        profile = config.profiles["fixture-model"]
        self.assertEqual(profile.engine, "worker")
        self.assertEqual(profile.display_name, "Fixture model")
        self.assertEqual(profile.compatibility_status, "tested")

    def test_profile_rejects_unconfigured_engine(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    profiles={
                        "fixture-model": {
                            "model_path": "/tmp/models/fixture.gguf",
                            "engine": "missing",
                            "display_name": "Fixture model",
                            "compatibility_status": "tested",
                        }
                    }
                )
            )

    def test_profile_rejects_unknown_field(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config(
                _base_config(
                    engines={"worker": {"executable": "/bin/false"}},
                    profiles={
                        "fixture-model": {
                            "model_path": "/tmp/models/fixture.gguf",
                            "engine": "worker",
                            "display_name": "Fixture model",
                            "compatibility_status": "tested",
                            "remote_fallback": True,
                        }
                    },
                )
            )


class RuntimeTypeValidationTests(unittest.TestCase):
    """Handoff 07 item A: booleans, non-finite floats and numeric strings
    must never be silently accepted or coerced into a runtime field."""

    def test_context_size_true_is_rejected_not_coerced_to_one(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"context_size": True}))

    def test_threads_false_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"threads": False}))

    def test_batch_size_numeric_string_is_rejected_not_coerced(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"batch_size": "2048"}))

    def test_ubatch_size_float_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"ubatch_size": 512.0}))

    def test_parallel_bool_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"parallel": True}))

    def test_temperature_nan_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"temperature": float("nan")}))

    def test_temperature_positive_infinity_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"temperature": float("inf")}))

    def test_temperature_negative_infinity_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"temperature": float("-inf")}))

    def test_top_p_nan_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"top_p": float("nan")}))

    def test_top_p_positive_infinity_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"top_p": float("inf")}))

    def test_top_p_negative_infinity_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"top_p": float("-inf")}))

    def test_temperature_bool_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"temperature": True}))

    def test_gpu_layers_non_string_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(runtime={"gpu_layers": 12}))

    def test_valid_runtime_types_are_still_accepted(self):
        config = load_config(
            _base_config(runtime={"context_size": 4096, "temperature": 0.2, "top_p": 0.9})
        )
        self.assertEqual(config.runtime.context_size, 4096)


class BindPortValidationTests(unittest.TestCase):
    def test_bind_port_true_is_rejected_not_coerced_to_one(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(bind={"host": "127.0.0.1", "port": True}))

    def test_bind_port_negative_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(bind={"host": "127.0.0.1", "port": -1}))

    def test_bind_port_above_65535_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(bind={"host": "127.0.0.1", "port": 65536}))

    def test_bind_port_numeric_string_is_rejected_not_coerced(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(bind={"host": "127.0.0.1", "port": "8080"}))


class DeclaredDigestValidationTests(unittest.TestCase):
    """Handoff 07 item A: every declared SHA-256 must be exactly 64
    lowercase hexadecimal characters."""

    def _engine(self, **overrides) -> dict:
        engine = {
            "library": "/opt/lib/libllama.so",
            "library_sha256": "a" * 64,
            "ggml_base_library": "/opt/lib/libggml-base.so",
            "ggml_base_library_sha256": "b" * 64,
            "ggml_library": "/opt/lib/libggml.so",
            "ggml_library_sha256": "c" * 64,
            "backend_libraries": [{"path": "/opt/lib/libggml-cpu.so", "sha256": "d" * 64}],
            "upstream_identity": "llama.cpp-b10380",
        }
        engine.update(overrides)
        return engine

    def test_uppercase_digest_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"libllama": self._engine(library_sha256="A" * 64)}))

    def test_short_digest_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"libllama": self._engine(library_sha256="a" * 63)}))

    def test_long_digest_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"libllama": self._engine(library_sha256="a" * 65)}))

    def test_non_hex_digest_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"libllama": self._engine(library_sha256="g" * 64)}))

    def test_backend_library_digest_format_is_validated(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "libllama": self._engine(
                            backend_libraries=[{"path": "/opt/lib/libggml-cpu.so", "sha256": "Z" * 64}]
                        )
                    }
                )
            )

    def test_valid_lowercase_digest_is_accepted(self):
        config = load_config(_base_config(engines={"libllama": self._engine()}))
        self.assertEqual(config.engines["libllama"].library_sha256, "a" * 64)


class NativePathTypeValidationTests(unittest.TestCase):
    """Handoff 07 item A: a non-string native path or upstream identity
    must be rejected before an EngineConfig is constructed, never silently
    coerced with str(...)."""

    def test_non_string_library_path_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": 12345,
                            "library_sha256": "a" * 64,
                            "ggml_base_library": "/opt/lib/libggml-base.so",
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [
                                {"path": "/opt/lib/libggml-cpu.so", "sha256": "d" * 64}
                            ],
                            "upstream_identity": "llama.cpp-b10380",
                        }
                    }
                )
            )

    def test_non_string_ggml_base_library_path_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": "/opt/lib/libllama.so",
                            "library_sha256": "a" * 64,
                            "ggml_base_library": ["not", "a", "string"],
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [
                                {"path": "/opt/lib/libggml-cpu.so", "sha256": "d" * 64}
                            ],
                            "upstream_identity": "llama.cpp-b10380",
                        }
                    }
                )
            )

    def test_non_string_upstream_identity_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": "/opt/lib/libllama.so",
                            "library_sha256": "a" * 64,
                            "ggml_base_library": "/opt/lib/libggml-base.so",
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [
                                {"path": "/opt/lib/libggml-cpu.so", "sha256": "d" * 64}
                            ],
                            "upstream_identity": 12345,
                        }
                    }
                )
            )

    def test_non_string_backend_library_path_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    engines={
                        "libllama": {
                            "library": "/opt/lib/libllama.so",
                            "library_sha256": "a" * 64,
                            "ggml_base_library": "/opt/lib/libggml-base.so",
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": "/opt/lib/libggml.so",
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [{"path": 42, "sha256": "d" * 64}],
                            "upstream_identity": "llama.cpp-b10380",
                        }
                    }
                )
            )

    def test_empty_string_executable_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config(_base_config(engines={"worker": {"executable": ""}}))


if __name__ == "__main__":
    unittest.main()
