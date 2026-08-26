"""Pinned in-process libllama adapter for exact local GGUF files.

The binding exposes only local model loading, chat templating, tokenisation,
decoding and sampling. It does not provide a downloader, URL resolver, tool
executor or second network listener. Native library and compute-backend bytes
are verified before loading.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from plag_in.config import BackendLibraryConfig, RuntimeConfig
from plag_in.engines.base import EngineAdapter, EngineCapabilities
from plag_in.errors import (
    InvalidRequestError,
    ModelIdentityMismatchError,
    NativeInitializationError,
    NativeRuntimeError,
    OverloadedError,
    RequestCancelledError,
    UnsupportedCapabilityError,
)
from plag_in.identity import canonical_digest, hash_file
from plag_in.native_profiles import (
    REGISTERED_NATIVE_ABI_PROFILES,
    NativeAbiProfile,
    identify_native_abi_profile,
)
from plag_in.receipts import RECEIPT_SCHEMA_VERSION


class _ModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class _ContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class _SamplerChainParams(ctypes.Structure):
    _fields_ = [("no_perf", ctypes.c_bool)]


class _Batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class _ChatMessage(ctypes.Structure):
    _fields_ = [("role", ctypes.c_char_p), ("content", ctypes.c_char_p)]


_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_ABORT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
EMBEDDED_GENERATION_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class NativeIdentity:
    library_sha256: str
    ggml_base_library_sha256: str
    ggml_library_sha256: str
    backend_library_sha256: tuple[str, ...]
    upstream_identity: str
    abi_profile_id: str

    def digest(self) -> str:
        return canonical_digest(
            {
                "library_sha256": self.library_sha256,
                "ggml_base_library_sha256": self.ggml_base_library_sha256,
                "ggml_library_sha256": self.ggml_library_sha256,
                "backend_library_sha256": list(self.backend_library_sha256),
                "upstream_identity": self.upstream_identity,
                "abi_profile_id": self.abi_profile_id,
            }
        )


class LibLlamaAdapter(EngineAdapter):
    name = "libllama"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(chat_completions="declared")


class EmbeddedLibLlama:
    supports_cancel_check = True

    def __init__(
        self,
        *,
        library_path: str,
        library_sha256: str,
        ggml_base_library_path: str,
        ggml_base_library_sha256: str,
        ggml_library_path: str,
        ggml_library_sha256: str,
        backend_libraries: tuple[BackendLibraryConfig, ...],
        upstream_identity: str,
        model_path: str,
        model_sha256: str,
        runtime: RuntimeConfig,
        native_abi_profiles: tuple[NativeAbiProfile, ...] = REGISTERED_NATIVE_ABI_PROFILES,
    ):
        self.library_path = Path(library_path).resolve()
        self.expected_library_sha256 = library_sha256
        self.ggml_base_library_path = Path(ggml_base_library_path).resolve()
        self.expected_ggml_base_sha256 = ggml_base_library_sha256
        self.ggml_library_path = Path(ggml_library_path).resolve()
        self.expected_ggml_sha256 = ggml_library_sha256
        self.backend_libraries = tuple(backend_libraries)
        self.upstream_identity = upstream_identity
        self.model_path = Path(model_path).resolve()
        self.expected_model_sha256 = model_sha256
        self.runtime = runtime
        self.native_abi_profiles = native_abi_profiles
        self.identity: NativeIdentity | None = None
        self._abi_profile: NativeAbiProfile | None = None
        self._lock = threading.Lock()
        # Native handles, tracked separately from the higher-level readiness
        # state below so `close()` can release exactly what was actually
        # acquired, in reverse order, from any partial-load point.
        self._ggml_base = None
        self._ggml = None
        self._backend_handles: list[int] = []
        self._lib = None
        self._backend_initialized = False
        self._model = None
        self._ctx = None
        self._vocab = None
        self._template: bytes | None = None
        self._effective_profile: dict | None = None
        self._active_cancel_check: Callable[[], bool] | None = None
        # Retained for the process lifetime of this instance: ctypes does
        # not keep a callback alive on the native side, only through this
        # Python reference, so dropping it would let the native library
        # call into freed memory the next time it logs.
        self._log_callback = _LOG_CALLBACK(lambda _level, _text, _user: None)
        self._ggml_log_callback = _LOG_CALLBACK(lambda _level, _text, _user: None)
        self._abort_callback = _ABORT_CALLBACK(self._native_abort_requested)

    def _native_abort_requested(self, _user_data) -> bool:
        check = self._active_cancel_check
        if check is None:
            return False
        try:
            return bool(check())
        except BaseException:
            return True

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise NativeRuntimeError("required native or model file is absent")
        file_stat = path.stat()
        if file_stat.st_uid not in {0, os.getuid()}:
            raise NativeRuntimeError(
                "refusing a native or model file owned by another unprivileged user"
            )
        if stat.S_IMODE(file_stat.st_mode) & 0o022:
            raise NativeRuntimeError("refusing a group-writable or world-writable native or model file")
        actual = hash_file(path)
        if actual != expected_sha256:
            raise ModelIdentityMismatchError(
                "native or model file digest mismatch",
                expected_sha256=expected_sha256,
                actual_sha256=actual,
            )

    def _configure_api(self) -> None:
        lib = self._lib
        lib.llama_log_set.argtypes = [_LOG_CALLBACK, ctypes.c_void_p]
        lib.llama_log_set.restype = None
        lib.llama_backend_init.argtypes = []
        lib.llama_backend_init.restype = None
        lib.llama_backend_free.argtypes = []
        lib.llama_backend_free.restype = None
        lib.llama_model_default_params.argtypes = []
        lib.llama_model_default_params.restype = _ModelParams
        lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _ModelParams]
        lib.llama_model_load_from_file.restype = ctypes.c_void_p
        lib.llama_model_free.argtypes = [ctypes.c_void_p]
        lib.llama_model_free.restype = None
        lib.llama_context_default_params.argtypes = []
        lib.llama_context_default_params.restype = _ContextParams
        lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _ContextParams]
        lib.llama_init_from_model.restype = ctypes.c_void_p
        lib.llama_free.argtypes = [ctypes.c_void_p]
        lib.llama_free.restype = None
        lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        lib.llama_model_get_vocab.restype = ctypes.c_void_p
        lib.llama_model_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.llama_model_chat_template.restype = ctypes.c_char_p
        lib.llama_chat_apply_template.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(_ChatMessage),
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        lib.llama_chat_apply_template.restype = ctypes.c_int32
        lib.llama_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        lib.llama_tokenize.restype = ctypes.c_int32
        lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.llama_batch_get_one.restype = _Batch
        lib.llama_decode.argtypes = [ctypes.c_void_p, _Batch]
        lib.llama_decode.restype = ctypes.c_int32
        lib.llama_sampler_chain_default_params.argtypes = []
        lib.llama_sampler_chain_default_params.restype = _SamplerChainParams
        lib.llama_sampler_chain_init.argtypes = [_SamplerChainParams]
        lib.llama_sampler_chain_init.restype = ctypes.c_void_p
        lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llama_sampler_chain_add.restype = None
        lib.llama_sampler_init_greedy.argtypes = []
        lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
        lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
        lib.llama_sampler_init_top_p.restype = ctypes.c_void_p
        lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
        lib.llama_sampler_init_temp.restype = ctypes.c_void_p
        lib.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
        lib.llama_sampler_init_dist.restype = ctypes.c_void_p
        lib.llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.llama_sampler_sample.restype = ctypes.c_int32
        lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
        lib.llama_sampler_free.restype = None
        lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_vocab_is_eog.restype = ctypes.c_bool
        lib.llama_token_to_piece.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_bool,
        ]
        lib.llama_token_to_piece.restype = ctypes.c_int32
        lib.llama_get_memory.argtypes = [ctypes.c_void_p]
        lib.llama_get_memory.restype = ctypes.c_void_p
        lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.llama_memory_clear.restype = None
        for name in ("llama_n_ctx", "llama_n_batch", "llama_n_ubatch", "llama_n_seq_max"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_uint32
        lib.llama_n_threads.argtypes = [ctypes.c_void_p]
        lib.llama_n_threads.restype = ctypes.c_int32
        lib.llama_n_threads_batch.argtypes = [ctypes.c_void_p]
        lib.llama_n_threads_batch.restype = ctypes.c_int32

    def _identify_abi_profile(self) -> NativeAbiProfile:
        profile = identify_native_abi_profile(
            upstream_identity=self.upstream_identity,
            library_sha256=self.expected_library_sha256,
            ggml_base_library_sha256=self.expected_ggml_base_sha256,
            ggml_library_sha256=self.expected_ggml_sha256,
            backend_library_sha256=tuple(item.sha256 for item in self.backend_libraries),
            profiles=self.native_abi_profiles,
        )
        if profile is None:
            raise NativeInitializationError(
                "native bundle has no registered ABI profile"
            )
        return profile

    def load(self) -> None:
        # Every native and model file is verified, complete file, before any
        # ctypes.CDLL call is made for any of them (regressions 1 to 3):
        # a mismatch on one file must never be masked by another file
        # having already been loaded. No handle is acquired yet at this
        # point, so a mismatch here needs no cleanup and is left to
        # propagate as its own typed error, unwrapped.
        self._verify_file(self.ggml_base_library_path, self.expected_ggml_base_sha256)
        self._verify_file(self.ggml_library_path, self.expected_ggml_sha256)
        self._verify_file(self.library_path, self.expected_library_sha256)
        self._verify_file(self.model_path, self.expected_model_sha256)
        for backend in self.backend_libraries:
            self._verify_file(Path(backend.path).resolve(), backend.sha256)

        self._abi_profile = self._identify_abi_profile()
        self.identity = NativeIdentity(
            library_sha256=self.expected_library_sha256,
            ggml_base_library_sha256=self.expected_ggml_base_sha256,
            ggml_library_sha256=self.expected_ggml_sha256,
            backend_library_sha256=tuple(item.sha256 for item in self.backend_libraries),
            upstream_identity=self.upstream_identity,
            abi_profile_id=self._abi_profile.profile_id,
        )

        try:
            self._load_native()
        except NativeRuntimeError:
            # Every explicit failure point below already calls close() and
            # raises a typed, already-cleaned-up error; let it propagate
            # unchanged rather than re-wrapping an already-typed failure.
            raise
        except BaseException as exc:
            # Anything else (a missing native symbol, a ctypes ABI mismatch,
            # any other unanticipated native-loader failure) may occur after
            # a handle has already been acquired, and on some platforms
            # carries the library's local file path in its own message. It
            # must still leave every acquired handle released, and the
            # public error body must never repeat that path (handoff 07
            # item B; a real `ctypes.CDLL(...).missing_symbol` access raises
            # `AttributeError` with the library path embedded in its text).
            self.close()
            raise NativeInitializationError(
                "native library initialisation failed"
            ) from exc

    def _load_native(self) -> None:
        # Load order (regression 4): GGML base with global symbol
        # visibility, then log suppression registered on it before any
        # backend is loaded, then the GGML front library, then the
        # explicitly allowlisted compute backends, then libllama.
        self._ggml_base = ctypes.CDLL(str(self.ggml_base_library_path), mode=os.RTLD_GLOBAL)
        self._ggml_base.ggml_log_set.argtypes = [_LOG_CALLBACK, ctypes.c_void_p]
        self._ggml_base.ggml_log_set.restype = None
        self._ggml_base.ggml_log_set(self._ggml_log_callback, None)

        self._ggml = ctypes.CDLL(str(self.ggml_library_path), mode=os.RTLD_GLOBAL)
        self._ggml.ggml_backend_load.argtypes = [ctypes.c_char_p]
        self._ggml.ggml_backend_load.restype = ctypes.c_void_p
        self._ggml.ggml_backend_unload.argtypes = [ctypes.c_void_p]
        self._ggml.ggml_backend_unload.restype = None
        for backend in self.backend_libraries:
            handle = self._ggml.ggml_backend_load(str(Path(backend.path).resolve()).encode())
            if not handle:
                self.close()
                raise NativeRuntimeError("failed to load an allowlisted compute backend")
            self._backend_handles.append(handle)

        self._lib = ctypes.CDLL(str(self.library_path), mode=os.RTLD_LOCAL)
        self._configure_api()
        self._lib.llama_log_set(self._log_callback, None)
        self._lib.llama_backend_init()
        self._backend_initialized = True

        model_params = self._lib.llama_model_default_params()
        if self.runtime.gpu_layers in {"auto", "all"}:
            model_params.n_gpu_layers = -1
        else:
            model_params.n_gpu_layers = int(self.runtime.gpu_layers)
        model_params.check_tensors = True
        self._model = self._lib.llama_model_load_from_file(str(self.model_path).encode(), model_params)
        if not self._model:
            self.close()
            raise NativeRuntimeError("libllama refused the selected GGUF")

        context_params = self._lib.llama_context_default_params()
        context_params.n_ctx = self.runtime.context_size
        context_params.n_batch = self.runtime.batch_size
        context_params.n_ubatch = self.runtime.ubatch_size
        context_params.n_seq_max = self.runtime.parallel
        if self.runtime.threads != -1:
            context_params.n_threads = self.runtime.threads
            context_params.n_threads_batch = self.runtime.threads
        context_params.embeddings = False
        context_params.abort_callback = ctypes.cast(self._abort_callback, ctypes.c_void_p).value
        context_params.abort_callback_data = None
        self._ctx = self._lib.llama_init_from_model(self._model, context_params)
        if not self._ctx:
            self.close()
            raise NativeRuntimeError("libllama failed to create a model context")
        self._vocab = self._lib.llama_model_get_vocab(self._model)
        if not self._vocab:
            self.close()
            raise NativeRuntimeError("libllama returned no vocabulary")
        self._template = self._lib.llama_model_chat_template(self._model, None)
        if not self._template:
            self.close()
            raise NativeRuntimeError("selected GGUF has no supported embedded chat template")
        self._effective_profile = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "inference_mode": "embedded",
            "engine": "libllama_embedded",
            "ollama_runtime_dependency": False,
            "requested": self.runtime.as_dict(),
            "effective": {
                "context_size": self._lib.llama_n_ctx(self._ctx),
                "batch_size": self._lib.llama_n_batch(self._ctx),
                "ubatch_size": self._lib.llama_n_ubatch(self._ctx),
                "parallel": self._lib.llama_n_seq_max(self._ctx),
                "threads": self._lib.llama_n_threads(self._ctx),
                "threads_batch": self._lib.llama_n_threads_batch(self._ctx),
            },
            # Each of the "effective" fields above came from a real native
            # query against the loaded context; GPU offload has no such
            # verified query in this pinned ABI, so it stays a separate,
            # explicitly unmeasured record rather than being inferred from
            # the requested layer count, a CUDA library name, a model
            # response or the absence of an error (handoff 07 item D).
            "measurement_status": {
                "context_size": "measured",
                "batch_size": "measured",
                "ubatch_size": "measured",
                "parallel": "measured",
                "threads": "measured",
                "threads_batch": "measured",
                "gpu_layers": "unassessed",
            },
            "gpu_offload": {
                "requested": self.runtime.gpu_layers,
                "observed": None,
                "measurement_method": (
                    "no native ABI query for effective GPU layer offload is verified "
                    "against the held header or source identity for this pinned build; "
                    "requested value only"
                ),
                "status": "unassessed",
            },
            "native_identity_digest": self.identity.digest(),
            "native_abi_profile": self._abi_profile.profile_id,
            # The digest of every native file actually loaded, not merely
            # configured: this block is only reached after every load step
            # above has already succeeded, so each of these files was
            # verified and then opened, in this order, on this call.
            "native_component_digests": {
                "libllama": self.expected_library_sha256,
                "ggml_base": self.expected_ggml_base_sha256,
                "ggml_front": self.expected_ggml_sha256,
                "backends": [item.sha256 for item in self.backend_libraries],
            },
            "upstream_identity": self.upstream_identity,
            "sampling_scope": "request value or confirmed fallback",
        }

    def ready(self) -> bool:
        return bool(self._ctx and self._model and self._vocab and self._template)

    @property
    def runtime_profile(self) -> dict:
        if self._effective_profile is None:
            raise NativeRuntimeError("embedded runtime is not loaded")
        return dict(self._effective_profile)

    @property
    def template_digest(self) -> str:
        if self._template is None:
            raise NativeRuntimeError("embedded runtime is not loaded")
        return hashlib.sha256(self._template).hexdigest()

    def _apply_template(self, messages: list[dict]) -> bytes:
        encoded = [(item["role"].encode(), item["content"].encode()) for item in messages]
        array_type = _ChatMessage * len(encoded)
        chat = array_type(*(_ChatMessage(role, content) for role, content in encoded))
        needed = self._lib.llama_chat_apply_template(
            self._template, chat, len(encoded), True, None, 0
        )
        if needed <= 0:
            raise NativeRuntimeError("chat template application failed")
        output = ctypes.create_string_buffer(needed + 1)
        written = self._lib.llama_chat_apply_template(
            self._template, chat, len(encoded), True, output, len(output)
        )
        if written < 0 or written > needed:
            raise NativeRuntimeError("chat template output was inconsistent")
        return output.raw[:written]

    def _tokenize(self, prompt: bytes) -> ctypes.Array:
        needed = self._lib.llama_tokenize(
            self._vocab, prompt, len(prompt), None, 0, True, True
        )
        if needed == -(2**31):
            raise NativeRuntimeError("prompt token count overflowed the native API")
        count = -needed if needed < 0 else needed
        if count <= 0:
            raise NativeRuntimeError("prompt tokenisation returned no tokens")
        tokens = (ctypes.c_int32 * count)()
        written = self._lib.llama_tokenize(
            self._vocab, prompt, len(prompt), tokens, count, True, True
        )
        if written != count:
            raise NativeRuntimeError("prompt tokenisation produced an inconsistent count")
        return tokens

    def _token_piece(self, token: int) -> bytes:
        output = ctypes.create_string_buffer(128)
        written = self._lib.llama_token_to_piece(
            self._vocab, token, output, len(output), 0, False
        )
        if written < 0:
            output = ctypes.create_string_buffer(-written)
            written = self._lib.llama_token_to_piece(
                self._vocab, token, output, len(output), 0, False
            )
        if written < 0:
            raise NativeRuntimeError("native token decoding failed")
        return output.raw[:written]

    @staticmethod
    def _validate_max_tokens(body: dict) -> int:
        max_tokens = body.get("max_tokens", 64)
        # bool is a subclass of int in Python: an explicit isinstance check
        # is required or `max_tokens: true` would silently become 1.
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise InvalidRequestError("'max_tokens' must be a positive integer")
        if max_tokens <= 0:
            raise InvalidRequestError("'max_tokens' must be a positive integer")
        return max_tokens

    def _validate_sampling(self, body: dict) -> tuple[float, float, int]:
        temperature = body.get("temperature", self.runtime.temperature)
        top_p = body.get("top_p", self.runtime.top_p)
        seed = body.get("seed", 0)
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidRequestError(f"'{name}' must be a finite number")
            if not math.isfinite(float(value)):
                raise InvalidRequestError(f"'{name}' must be a finite number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidRequestError("'seed' must be an integer")
        temperature = float(temperature)
        top_p = float(top_p)
        if temperature < 0:
            raise InvalidRequestError("'temperature' must not be negative")
        if not 0 <= top_p <= 1:
            raise InvalidRequestError("'top_p' must be between 0 and 1")
        return temperature, top_p, seed & 0xFFFFFFFF

    def _build_sampler_chain(self, temperature: float, top_p: float, seed: int) -> int:
        """Build one sampler chain, checking each component before chain insertion.

        Every `llama_sampler_init_*` call can return a null pointer; passing
        one to `llama_sampler_chain_add` must never happen (regression 8).
        A component already added is owned by the chain, so freeing the
        chain on a later component's failure is sufficient to release
        everything acquired so far; nothing further needs a separate free
        (regression 9).
        """
        chain_params = self._lib.llama_sampler_chain_default_params()
        chain_params.no_perf = True
        sampler = self._lib.llama_sampler_chain_init(chain_params)
        if not sampler:
            raise NativeRuntimeError("libllama failed to create a sampler chain")
        try:
            if temperature <= 0:
                component = self._lib.llama_sampler_init_greedy()
                if not component:
                    raise NativeRuntimeError("libllama failed to create the greedy sampler component")
                self._lib.llama_sampler_chain_add(sampler, component)
            else:
                top_p_component = self._lib.llama_sampler_init_top_p(ctypes.c_float(top_p), 1)
                if not top_p_component:
                    raise NativeRuntimeError("libllama failed to create the top-p sampler component")
                self._lib.llama_sampler_chain_add(sampler, top_p_component)

                temp_component = self._lib.llama_sampler_init_temp(ctypes.c_float(temperature))
                if not temp_component:
                    raise NativeRuntimeError("libllama failed to create the temperature sampler component")
                self._lib.llama_sampler_chain_add(sampler, temp_component)

                dist_component = self._lib.llama_sampler_init_dist(seed)
                if not dist_component:
                    raise NativeRuntimeError("libllama failed to create the distribution sampler component")
                self._lib.llama_sampler_chain_add(sampler, dist_component)
        except NativeRuntimeError:
            self._lib.llama_sampler_free(sampler)
            raise
        return sampler

    def chat_completion(
        self,
        body: dict,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        if body.get("stream"):
            raise UnsupportedCapabilityError(
                "streaming is not implemented by the embedded trial adapter"
            )
        # All typed validation happens before any native call below,
        # including tokenisation, so a rejected request never reaches
        # decode and never produces a completed-success receipt.
        max_tokens = self._validate_max_tokens(body)
        temperature, top_p, seed = self._validate_sampling(body)

        if not self._lock.acquire(blocking=False):
            raise OverloadedError("embedded model is already processing one request")
        sampler = None
        deadline = time.monotonic() + EMBEDDED_GENERATION_TIMEOUT_S
        cancel_reason = None

        def generation_cancelled() -> bool:
            nonlocal cancel_reason
            if time.monotonic() >= deadline:
                cancel_reason = "generation_time_limit"
                return True
            if cancel_check is None:
                return False
            try:
                if cancel_check():
                    cancel_reason = "client_disconnected"
                    return True
            except BaseException:
                cancel_reason = "client_state_unavailable"
                return True
            return False

        self._active_cancel_check = generation_cancelled
        try:
            if generation_cancelled():
                raise RequestCancelledError(
                    "embedded generation was cancelled before native decoding",
                    reason=cancel_reason,
                )
            messages = body["messages"]
            prompt = self._apply_template(messages)
            tokens = self._tokenize(prompt)
            effective_ctx = self._effective_profile["effective"]["context_size"]
            if len(tokens) + max_tokens > effective_ctx:
                raise NativeRuntimeError("prompt and requested output exceed the effective context")

            memory = self._lib.llama_get_memory(self._ctx)
            if not memory:
                raise NativeRuntimeError("libllama returned no memory handle for context clearing")
            self._lib.llama_memory_clear(memory, True)
            batch = self._lib.llama_batch_get_one(tokens, len(tokens))
            decode_status = self._lib.llama_decode(self._ctx, batch)
            if decode_status != 0:
                if generation_cancelled():
                    raise RequestCancelledError(
                        "embedded generation was cancelled during prompt decoding",
                        reason=cancel_reason,
                    )
                raise NativeRuntimeError("native prompt decoding failed", decode_status=decode_status)

            sampler = self._build_sampler_chain(temperature, top_p, seed)

            pieces = []
            finish_reason = "length"
            for _ in range(max_tokens):
                if generation_cancelled():
                    raise RequestCancelledError(
                        "embedded generation was cancelled before completion",
                        reason=cancel_reason,
                    )
                token = self._lib.llama_sampler_sample(sampler, self._ctx, -1)
                if self._lib.llama_vocab_is_eog(self._vocab, token):
                    finish_reason = "stop"
                    break
                pieces.append(self._token_piece(token))
                token_value = ctypes.c_int32(token)
                batch = self._lib.llama_batch_get_one(ctypes.byref(token_value), 1)
                decode_status = self._lib.llama_decode(self._ctx, batch)
                if decode_status != 0:
                    if generation_cancelled():
                        raise RequestCancelledError(
                            "embedded generation was cancelled during decoding",
                            reason=cancel_reason,
                        )
                    raise NativeRuntimeError(
                        "native generated-token decoding failed", decode_status=decode_status
                    )

            content = b"".join(pieces).decode("utf-8", errors="replace")
            return {
                "id": f"chatcmpl-plag-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": len(pieces),
                    "total_tokens": len(tokens) + len(pieces),
                },
            }
        finally:
            self._active_cancel_check = None
            if sampler:
                self._lib.llama_sampler_free(sampler)
            self._lock.release()

    def close(self) -> None:
        """Release exactly what was acquired, in reverse order.

        Safe to call before any load, after a partial load and after a
        full load or generation failure (regression 12): every step below
        is guarded by the state it would release, and every handle is
        cleared afterwards so a repeated call finds nothing left to do.
        """
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None
        if self._backend_initialized:
            try:
                self._lib.llama_backend_free()
            except AttributeError:
                pass
            self._backend_initialized = False
        if self._ggml is not None:
            for handle in reversed(self._backend_handles):
                self._ggml.ggml_backend_unload(handle)
        self._backend_handles.clear()
        self._vocab = None
        self._template = None
        self._effective_profile = None
        self._active_cancel_check = None
        self._abi_profile = None
        self.identity = None
        self._lib = None
        self._ggml = None
        self._ggml_base = None


def ffi_structure_sizes() -> dict[str, int]:
    return {
        "model_params": ctypes.sizeof(_ModelParams),
        "context_params": ctypes.sizeof(_ContextParams),
        "sampler_chain_params": ctypes.sizeof(_SamplerChainParams),
        "batch": ctypes.sizeof(_Batch),
        "chat_message": ctypes.sizeof(_ChatMessage),
    }
