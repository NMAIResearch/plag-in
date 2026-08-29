"""Embedded libllama boundary tests that do not load a live model.

Native calls are always mocked here (fixtures below), never routed to the
real installed ``libllama``/``libggml`` binaries: this proves the Python
lifecycle, ordering and validation logic, not compatibility with any
particular installed library or model build.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from plag_in.config import BackendLibraryConfig, BindConfig, GatewayConfig, RuntimeConfig
from plag_in.engines.libllama import EmbeddedLibLlama, LibLlamaAdapter, ffi_structure_sizes
from plag_in.errors import (
    InvalidRequestError,
    ModelIdentityMismatchError,
    NativeInitializationError,
    NativeRuntimeError,
    OverloadedError,
    RequestCancelledError,
    UnsupportedCapabilityError,
)
from plag_in.gateway import BackendInfo, GatewayContext
from plag_in.identity import ModelIdentity, canonical_digest, hash_file
from plag_in.native_profiles import NativeAbiProfile
from plag_in.receipts import ReceiptStore
from plag_in.registry import Registry, RegistryEntry


class _FixtureEmbeddedBackend:
    def ready(self) -> bool:
        return True

    def chat_completion(self, body: dict) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": "fixture"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "model": body["model"],
        }


# -- shared fixture: a recording, permissive stand-in for ctypes.CDLL -------


class _FakeNativeFn:
    """Stands in for one native function pointer.

    Mimics enough of the real ctypes attribute protocol (`argtypes` /
    `restype` may be assigned) that `EmbeddedLibLlama._configure_api` runs
    unmodified, while recording every call in shared order and returning a
    caller-supplied result.
    """

    def __init__(self, tag: str, name: str, call_log: list[str], result_factory):
        self.tag = tag
        self.name = name
        self.call_log = call_log
        self.result_factory = result_factory
        self.argtypes = None
        self.restype = None

    def __call__(self, *args, **kwargs):
        self.call_log.append(f"{self.tag}.{self.name}")
        return self.result_factory(self.tag, self.name, args, kwargs)


class _FakeNativeLibrary:
    def __init__(self, tag: str, call_log: list[str], result_factory, *, missing_symbol=None, path_str=""):
        self.tag = tag
        self.call_log = call_log
        self.result_factory = result_factory
        self._fns: dict[str, _FakeNativeFn] = {}
        self._missing_symbol = missing_symbol
        self._path_str = path_str

    def __getattr__(self, name: str):
        if name == self._missing_symbol:
            # Mirrors real ctypes.CDLL.__getattr__ on Linux: a missing
            # symbol raises AttributeError with the library's own path
            # embedded in the message (handoff 07 item B: that text must
            # never reach the public error body).
            raise AttributeError(f"{self._path_str}: undefined symbol: {name}")
        if name not in self._fns:
            self._fns[name] = _FakeNativeFn(self.tag, name, self.call_log, self.result_factory)
        return self._fns[name]


def _default_result_factory(tag: str, name: str, args, kwargs):
    if name in (
        "llama_model_default_params",
        "llama_context_default_params",
        "llama_sampler_chain_default_params",
    ):
        return types.SimpleNamespace()
    if name == "llama_model_chat_template":
        return b"fake-template"
    if name in (
        "llama_n_ctx",
        "llama_n_batch",
        "llama_n_ubatch",
        "llama_n_seq_max",
        "llama_n_threads",
        "llama_n_threads_batch",
    ):
        return 1
    # Every other native call (loads, inits, handle getters) only needs a
    # truthy pointer-shaped sentinel for these lifecycle-only fixtures.
    return 1


def _make_fake_cdll(
    call_log: list[str],
    *,
    fail_backend_load: bool = False,
    fail_at: str | None = None,
    missing_symbol: str | None = None,
):
    def _fake_cdll(path, mode=0):
        path_str = str(path)
        if "base" in path_str:
            tag = "ggml_base"
        elif "ggml" in path_str:
            tag = "ggml_front"
        else:
            tag = "llama"
        call_log.append(f"CDLL:{tag}")

        def _result_factory(tag_, name, args, kwargs):
            if fail_backend_load and name == "ggml_backend_load":
                return 0
            if fail_at is not None and name == fail_at:
                return None
            return _default_result_factory(tag_, name, args, kwargs)

        return _FakeNativeLibrary(
            tag, call_log, _result_factory, missing_symbol=missing_symbol, path_str=path_str
        )

    return _fake_cdll


def _write(path: Path, content: bytes = b"fixture-native-bytes") -> tuple[str, str]:
    path.write_bytes(content)
    return str(path), hash_file(path)


def _build_backend(tmp_path: Path, *, runtime: RuntimeConfig | None = None) -> EmbeddedLibLlama:
    llama_path, llama_sha = _write(tmp_path / "libllama.so", b"llama-bytes")
    base_path, base_sha = _write(tmp_path / "libggml-base.so", b"ggml-base-bytes")
    front_path, front_sha = _write(tmp_path / "libggml.so", b"ggml-front-bytes")
    backend_path, backend_sha = _write(tmp_path / "libggml-cpu.so", b"ggml-cpu-bytes")
    model_path, model_sha = _write(tmp_path / "model.gguf", b"GGUF-fixture")
    profile = NativeAbiProfile(
        profile_id="fixture-abi",
        upstream_identity="fixture",
        library_sha256=llama_sha,
        ggml_base_library_sha256=base_sha,
        ggml_library_sha256=front_sha,
        backend_library_sha256=(backend_sha,),
    )
    return EmbeddedLibLlama(
        library_path=llama_path,
        library_sha256=llama_sha,
        ggml_base_library_path=base_path,
        ggml_base_library_sha256=base_sha,
        ggml_library_path=front_path,
        ggml_library_sha256=front_sha,
        backend_libraries=(BackendLibraryConfig(path=backend_path, sha256=backend_sha),),
        upstream_identity="fixture",
        model_path=model_path,
        model_sha256=model_sha,
        runtime=runtime or RuntimeConfig(context_size=4096),
        native_abi_profiles=(profile,),
    )


class LibLlamaStructureTests(unittest.TestCase):
    def test_b10380_64_bit_structure_sizes(self):
        self.assertEqual(
            ffi_structure_sizes(),
            {
                "model_params": 72,
                "context_params": 160,
                "sampler_chain_params": 1,
                "batch": 56,
                "chat_message": 16,
            },
        )

    def test_library_digest_mismatch_fails_before_native_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "libllama.so"
            path.write_bytes(b"not-a-library")
            backend = EmbeddedLibLlama(
                library_path=str(path),
                library_sha256="0" * 64,
                ggml_base_library_path=str(path),
                ggml_base_library_sha256=hash_file(path),
                ggml_library_path=str(path),
                ggml_library_sha256=hash_file(path),
                backend_libraries=(BackendLibraryConfig(path=str(path), sha256=hash_file(path)),),
                upstream_identity="fixture",
                model_path=str(path),
                model_sha256=hash_file(path),
                runtime=RuntimeConfig(),
            )
            with self.assertRaises(ModelIdentityMismatchError):
                backend.load()


class NativeLoadOrderTests(unittest.TestCase):
    """Regressions 1 to 5: verification and load-order boundary."""

    def test_ggml_base_digest_mismatch_fails_before_any_cdll_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            backend.expected_ggml_base_sha256 = "0" * 64
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                with self.assertRaises(ModelIdentityMismatchError):
                    backend.load()
            self.assertEqual(call_log, [])

    def test_backend_digest_mismatch_fails_before_any_native_library_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            mutated = BackendLibraryConfig(path=backend.backend_libraries[0].path, sha256="0" * 64)
            backend.backend_libraries = (mutated,)
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                with self.assertRaises(ModelIdentityMismatchError):
                    backend.load()
            self.assertEqual(call_log, [])

    def test_unregistered_abi_profile_fails_before_any_native_library_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            backend.native_abi_profiles = ()
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                with self.assertRaises(NativeInitializationError):
                    backend.load()
            self.assertEqual(call_log, [])

    def test_registered_abi_profile_is_recorded_after_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                backend.load()
                self.assertEqual(
                    backend.runtime_profile["native_abi_profile"], "fixture-abi"
                )
                backend.close()

    def test_foreign_unprivileged_owner_is_refused_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.so"
            path.write_bytes(b"native")
            real_stat = Path.stat

            def foreign_stat(target, *args, **kwargs):
                observed = real_stat(target, *args, **kwargs)
                fields = list(observed)
                fields[4] = os.getuid() + 1
                return os.stat_result(fields)

            with patch("pathlib.Path.stat", new=foreign_stat):
                with self.assertRaises(NativeRuntimeError):
                    EmbeddedLibLlama._verify_file(path, hash_file(path))

    def test_native_load_order_is_base_then_suppression_then_front_then_backend_then_llama(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                backend.load()
            try:
                base_cdll = call_log.index("CDLL:ggml_base")
                log_suppress = call_log.index("ggml_base.ggml_log_set")
                front_cdll = call_log.index("CDLL:ggml_front")
                backend_load = call_log.index("ggml_front.ggml_backend_load")
                llama_cdll = call_log.index("CDLL:llama")
                llama_init = call_log.index("llama.llama_backend_init")
            except ValueError:
                self.fail(f"expected native call missing from recorded order: {call_log}")
            self.assertLess(base_cdll, log_suppress)
            self.assertLess(log_suppress, front_cdll)
            self.assertLess(front_cdll, backend_load)
            self.assertLess(backend_load, llama_cdll)
            self.assertLess(llama_cdll, llama_init)
            backend.close()

    def test_failed_backend_load_releases_only_what_was_acquired_and_close_stays_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log, fail_backend_load=True)):
                with self.assertRaises(NativeRuntimeError):
                    backend.load()
            # load() already called close() internally on this failure path;
            # a second, external close() must remain a no-op (regression 12).
            backend.close()
            self.assertIsNone(backend._lib)
            self.assertIsNone(backend._ggml)


class MissingSymbolLoadTests(unittest.TestCase):
    """Regression 6: a missing native symbol fails typed, without leaking

    the library's local file path into the public error body, and leaves
    every acquired handle and backend state released.
    """

    def test_missing_symbol_fails_typed_and_leaks_no_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch(
                "ctypes.CDLL",
                side_effect=_make_fake_cdll(call_log, missing_symbol="llama_backend_init"),
            ):
                with self.assertRaises(NativeInitializationError) as ctx:
                    backend.load()
            message = str(ctx.exception)
            self.assertNotIn(str(backend.library_path), message)
            self.assertNotIn(".so", message)
            self.assertNotIn(tmp, message)
            # close() was already invoked internally on this failure path;
            # a second, external close() must remain a safe no-op.
            backend.close()
            self.assertIsNone(backend._lib)
            self.assertIsNone(backend._ggml)
            self.assertIsNone(backend._ggml_base)
            self.assertEqual(backend._backend_handles, [])

    def test_missing_symbol_during_backend_load_leaves_only_acquired_backends_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch(
                "ctypes.CDLL",
                side_effect=_make_fake_cdll(call_log, missing_symbol="ggml_backend_load"),
            ):
                with self.assertRaises(NativeInitializationError):
                    backend.load()
            backend.close()
            self.assertIsNone(backend._ggml)
            self.assertIsNone(backend._ggml_base)
            self.assertEqual(backend._backend_handles, [])


class LibraryOpenFailureLoadTests(unittest.TestCase):
    """Failure injection at the library-open point itself (handoff 07 item

    B): `ctypes.CDLL(...)` can raise `OSError` directly (a malformed or
    incompatible shared object), distinct from a missing symbol on an
    otherwise successfully opened handle.
    """

    def test_cdll_open_failure_fails_typed_without_leaking_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []

            def _raising_cdll(path, mode=0):
                path_str = str(path)
                call_log.append(f"CDLL-attempt:{path_str}")
                if "llama.so" in path_str and "ggml" not in path_str:
                    raise OSError(f"{path_str}: cannot open shared object file")
                return _make_fake_cdll(call_log)(path, mode)

            with patch("ctypes.CDLL", side_effect=_raising_cdll):
                with self.assertRaises(NativeInitializationError) as ctx:
                    backend.load()
            self.assertNotIn(str(backend.library_path), str(ctx.exception))
            backend.close()
            self.assertIsNone(backend._lib)
            self.assertIsNone(backend._ggml)
            self.assertIsNone(backend._ggml_base)
            self.assertEqual(backend._backend_handles, [])


class NullPointerLoadTests(unittest.TestCase):
    """Regression 11 (load-time half): null model, context and vocabulary
    pointers each fail typed and release only what load() had already
    acquired at that point."""

    def _assert_load_fails_and_cleans_up(self, fail_at: str):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log, fail_at=fail_at)):
                with self.assertRaises(NativeRuntimeError):
                    backend.load()
            backend.close()  # must remain a safe no-op after the internal close()
            self.assertIsNone(backend._lib)
            self.assertIsNone(backend._model)
            self.assertIsNone(backend._ctx)
            self.assertIsNone(backend._vocab)

    def test_null_model_pointer_fails_typed_and_cleans_up(self):
        self._assert_load_fails_and_cleans_up("llama_model_load_from_file")

    def test_null_context_pointer_fails_typed_and_cleans_up(self):
        self._assert_load_fails_and_cleans_up("llama_init_from_model")

    def test_null_vocab_pointer_fails_typed_and_cleans_up(self):
        self._assert_load_fails_and_cleans_up("llama_model_get_vocab")

    def test_null_template_fails_typed_and_cleans_up(self):
        self._assert_load_fails_and_cleans_up("llama_model_chat_template")


class LoadFailureNeverAnnouncesReadinessTests(unittest.TestCase):
    """Probe 9: a refused model never reaches a ready listener state.

    `ready()` is what the gateway's `/ready` route and the session start
    path consult, so a load that failed must leave it false at every
    failure point, with every acquired handle already released.
    """

    FAILURE_POINTS = (
        # An architecture libllama cannot build is a null model pointer.
        "llama_model_load_from_file",
        "llama_init_from_model",
        "llama_model_get_vocab",
        # No usable embedded chat template.
        "llama_model_chat_template",
    )

    def test_no_failure_point_leaves_the_backend_ready(self):
        for fail_at in self.FAILURE_POINTS:
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                backend = _build_backend(Path(tmp))
                self.assertFalse(backend.ready())
                with patch("ctypes.CDLL", side_effect=_make_fake_cdll([], fail_at=fail_at)):
                    with self.assertRaises(NativeRuntimeError):
                        backend.load()
                self.assertFalse(backend.ready())
                backend.close()
                self.assertFalse(backend.ready())

    def test_no_failure_point_leaves_a_runtime_profile_or_template_digest(self):
        for fail_at in self.FAILURE_POINTS:
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                backend = _build_backend(Path(tmp))
                with patch("ctypes.CDLL", side_effect=_make_fake_cdll([], fail_at=fail_at)):
                    with self.assertRaises(NativeRuntimeError):
                        backend.load()
                with self.assertRaises(NativeRuntimeError):
                    _ = backend.runtime_profile
                with self.assertRaises(NativeRuntimeError):
                    _ = backend.template_digest

    def test_no_failure_point_leaves_a_loaded_backend_handle(self):
        for fail_at in self.FAILURE_POINTS:
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                backend = _build_backend(Path(tmp))
                with patch("ctypes.CDLL", side_effect=_make_fake_cdll([], fail_at=fail_at)):
                    with self.assertRaises(NativeRuntimeError):
                        backend.load()
                self.assertEqual(backend._backend_handles, [])
                self.assertIsNone(backend._ggml)
                self.assertIsNone(backend._ggml_base)
                self.assertFalse(backend._backend_initialized)

    def test_a_successful_load_is_the_only_route_to_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll([])):
                backend.load()
                self.assertTrue(backend.ready())
                backend.close()
            self.assertFalse(backend.ready())


class LogSuppressionTests(unittest.TestCase):
    """Regression 5: a fixture native log line never reaches stdout/stderr."""

    def test_noop_log_callbacks_swallow_fixture_log_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                backend._log_callback(0, b"fixture llama log line\n", None)
                backend._ggml_log_callback(0, b"fixture ggml log line\n", None)
            self.assertEqual(out.getvalue(), "")
            self.assertEqual(err.getvalue(), "")


class CloseIdempotencyTests(unittest.TestCase):
    """Regression 12: close() is safe before load, after partial load and after full load."""

    def test_close_before_any_load_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            backend.close()
            backend.close()

    def test_close_after_full_load_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _build_backend(Path(tmp))
            call_log: list[str] = []
            with patch("ctypes.CDLL", side_effect=_make_fake_cdll(call_log)):
                backend.load()
                backend.close()
                backend.close()
            self.assertIsNone(backend._lib)
            self.assertIsNone(backend._ggml)
            self.assertIsNone(backend._ggml_base)
            self.assertIsNone(backend._ctx)
            self.assertIsNone(backend._model)


class ChatCompletionValidationTests(unittest.TestCase):
    """Regressions 7, 8, 9, 10, 11: typed, total validation and null-pointer checks."""

    def _loaded_backend(self, tmp_path: Path) -> EmbeddedLibLlama:
        backend = _build_backend(tmp_path, runtime=RuntimeConfig(context_size=4096))
        backend._lib = Mock()
        backend._lib.llama_chat_apply_template.side_effect = [10, 10]
        backend._lib.llama_tokenize.side_effect = [-3, 3]
        backend._lib.llama_get_memory.return_value = 1
        backend._lib.llama_decode.return_value = 0
        backend._lib.llama_sampler_chain_init.return_value = 1
        backend._lib.llama_sampler_init_greedy.return_value = 1
        backend._lib.llama_sampler_init_top_p.return_value = 1
        backend._lib.llama_sampler_init_temp.return_value = 1
        backend._lib.llama_sampler_init_dist.return_value = 1
        backend._lib.llama_vocab_is_eog.return_value = True
        backend._ctx = object()
        backend._vocab = object()
        backend._template = b"fake-template"
        backend._effective_profile = {"effective": {"context_size": 4096}}
        return backend

    def _body(self, **overrides) -> dict:
        body = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}
        body.update(overrides)
        return body

    def test_stream_true_returns_unsupported_capability_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(UnsupportedCapabilityError):
                backend.chat_completion(self._body(stream=True))

    def test_max_tokens_true_is_rejected_not_silently_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(InvalidRequestError):
                backend.chat_completion(self._body(max_tokens=True))

    def test_max_tokens_zero_negative_and_float_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (0, -1, 4.5):
                backend = self._loaded_backend(Path(tmp))
                with self.assertRaises(InvalidRequestError):
                    backend.chat_completion(self._body(max_tokens=bad))

    def test_max_tokens_beyond_effective_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=10_000))

    def test_non_finite_temperature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(InvalidRequestError):
                backend.chat_completion(self._body(temperature=float("nan")))

    def test_out_of_range_top_p_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(InvalidRequestError):
                backend.chat_completion(self._body(top_p=1.5))

    def test_negative_temperature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(InvalidRequestError):
                backend.chat_completion(self._body(temperature=-0.1))

    def test_non_integer_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(InvalidRequestError):
                backend.chat_completion(self._body(seed=1.5))

    def test_null_memory_pointer_fails_typed_and_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_get_memory.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4))
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_null_sampler_pointer_fails_typed_and_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_sampler_chain_init.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4))
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_null_greedy_sampler_component_fails_typed_and_frees_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_sampler_init_greedy.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4, temperature=0.0))
            backend._lib.llama_sampler_free.assert_called_once_with(1)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_null_top_p_sampler_component_fails_typed_and_frees_the_chain_before_further_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_sampler_init_top_p.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4, temperature=0.5))
            backend._lib.llama_sampler_init_temp.assert_not_called()
            backend._lib.llama_sampler_init_dist.assert_not_called()
            backend._lib.llama_sampler_free.assert_called_once_with(1)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_null_temperature_sampler_component_fails_typed_and_frees_the_chain_before_dist(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_sampler_init_temp.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4, temperature=0.5))
            backend._lib.llama_sampler_init_dist.assert_not_called()
            backend._lib.llama_sampler_free.assert_called_once_with(1)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_null_distribution_sampler_component_fails_typed_and_frees_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lib.llama_sampler_init_dist.return_value = None
            with self.assertRaises(NativeRuntimeError):
                backend.chat_completion(self._body(max_tokens=4, temperature=0.5))
            backend._lib.llama_sampler_free.assert_called_once_with(1)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_valid_request_completes_and_frees_the_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            result = backend.chat_completion(self._body(max_tokens=4))
            self.assertEqual(result["model"], "fixture")
            backend._lib.llama_sampler_free.assert_called_once_with(1)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_cancelled_request_stops_before_native_prompt_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            with self.assertRaises(RequestCancelledError) as caught:
                backend.chat_completion(
                    self._body(max_tokens=4), cancel_check=lambda: True
                )
            self.assertEqual(caught.exception.fields["reason"], "client_disconnected")
            backend._lib.llama_chat_apply_template.assert_not_called()
            self.assertIsNone(backend._active_cancel_check)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()

    def test_overloaded_when_already_processing_one_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._loaded_backend(Path(tmp))
            backend._lock.acquire()
            try:
                with self.assertRaises(OverloadedError):
                    backend.chat_completion(self._body(max_tokens=4))
            finally:
                backend._lock.release()


class EmbeddedGatewayRouteTests(unittest.TestCase):
    def test_chat_reaches_in_process_backend_and_receipt_names_no_backend_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"GGUF-fixture")
            identity = ModelIdentity.from_file(model_path)
            config = GatewayConfig(model_roots=(root,), bind=BindConfig(host="127.0.0.1", port=8080))
            registry = Registry()
            digest = canonical_digest({"engine": "libllama", "fixture": True})
            registry.bind(
                RegistryEntry(
                    alias="embedded-fixture",
                    identity=identity,
                    engine="libllama",
                    engine_config_digest=digest,
                )
            )
            store = ReceiptStore(root / "receipts.jsonl", hmac_key_path=root / "key")
            context = GatewayContext(config, registry, store, {"libllama": LibLlamaAdapter()})
            context.register_backend(
                "embedded-fixture",
                BackendInfo(
                    engine="libllama",
                    engine_version="fixture",
                    inference_mode="embedded",
                    host="127.0.0.1",
                    port=0,
                    weight_digest=identity.sha256,
                    config_digest=digest,
                    # An embedded backend must never carry worker-era
                    # identity (handoff 07 item C): both stay unset.
                    executable_digest=None,
                    argv_digest=None,
                    template_digest="c" * 64,
                    native_identity_digest="f" * 64,
                    native_component_digests={"libllama": "1" * 64, "backends": ["2" * 64]},
                    upstream_identity="fixture-upstream",
                    requested_runtime_profile={"context_size": 4096},
                    effective_runtime_profile={"context_size": 4096},
                    measurement_status={"context_size": "measured", "gpu_layers": "unassessed"},
                    gpu_offload={"requested": "0", "observed": None, "status": "unassessed"},
                    runtime_profile={"engine": "libllama_embedded"},
                    inference_backend=_FixtureEmbeddedBackend(),
                ),
            )
            status, body, headers = context.handle(
                "POST",
                "/v1/chat/completions",
                {},
                json.dumps(
                    {"model": "embedded-fixture", "messages": [{"role": "user", "content": "hi"}]}
                ).encode(),
                "127.0.0.1",
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "fixture")
            receipt = store.get(headers["X-PLAG-IN-Receipt-ID"])
            self.assertEqual(receipt["backend_address"], "in_process")
            self.assertEqual(receipt["plag_in_adapter_record"]["engine"], "libllama_embedded")
            self.assertEqual(
                sorted(receipt["runtime_profile"]),
                ["effective", "measurement_status", "requested"],
            )
            self.assertEqual(receipt["inference_mode"], "embedded")
            self.assertIsNone(receipt["engine_executable_digest"])
            self.assertIsNone(receipt["argv_digest"])
            self.assertEqual(receipt["native_identity_digest"], "f" * 64)
            self.assertEqual(receipt["upstream_identity"], "fixture-upstream")
            self.assertEqual(receipt["measurement_status"]["gpu_layers"], "unassessed")
            self.assertEqual(receipt["gpu_offload"]["status"], "unassessed")
            # R3-F2: the embedded path must satisfy the reference verifier on
            # the same terms as the worker path, checked against the bytes this
            # route actually wrote rather than a fixture receipt.
            tool = Path(__file__).resolve().parents[2] / "tools" / "verify_inference_receipts.py"
            verified = subprocess.run(
                [
                    sys.executable, str(tool), str(store.path),
                    "--hmac-key", str(store.hmac_key_path),
                    "--checkpoint", str(store.path) + ".checkpoint",
                ],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(verified.returncode, 0, verified.stdout)
            self.assertIn("result=valid", verified.stdout)

            capabilities = context.capabilities_document(None)
            self.assertEqual(capabilities["endpoints"]["chat_completions"], "declared")
            self.assertEqual(capabilities["tool_call_transport_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
