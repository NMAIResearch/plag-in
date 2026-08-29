"""OpenAI-compatible local gateway: routing, auth, capabilities and receipts.

Two request protocols are implemented over one generation path: Chat
Completions and a bounded text-only Responses subset. Both are named for
the wire protocol they speak, neither is preferred over the other in
status or capabilities, and neither inspects the client's name,
user-agent or executable to decide how a request is handled. The exact
implemented subset of each is published in the capability document, so a
client learns what is supported without probing for it.

The remaining capability-gated endpoints (`/v1/completions`,
`/v1/embeddings`, `/v1/messages`) always return a typed error in this
MVP; no adapter marks them `tested`, so the gateway never silently
rewrites a request to fit an untested path. Tool-call fields returned by
a backend are forwarded verbatim: the gateway has no code path that
executes a tool, and the Responses route neither executes a tool nor
keeps conversation state.

Locality (CODEX_REVIEW_MVP_2026-08-25.md F1): a non-loopback inference
backend is refused at registration, and L1 requires both a loopback
gateway bind and at least one registered loopback backend; configuration
inspected with no running backend stays at L0. The gateway's *reported*
bind address is the effective socket address obtained after the listener
actually binds, not the pre-bind configured value, so a configured port
`0` resolves to the assigned port in status and receipts rather than
staying `0` (CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md R5).

The HTTP work boundary is bounded before any request body is read
(R8): `_BoundedThreadingHTTPServer` admits at most
`MAX_CONCURRENT_CONNECTIONS` accepted connections into handler threads at
once. An excess connection receives a typed `overloaded` response and is
closed immediately, before its body, headers or even a route are ever
inspected; the separate `MAX_CONCURRENT_CHAT_REQUESTS` semaphore remains
for inference-specific admission after routing.
"""
from __future__ import annotations

import hmac
import itertools
import json
import math
import re
import secrets
import select
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Callable

from plag_in.aliasing import validate_alias
from plag_in.config import ApiKey, GatewayConfig, is_loopback, parse_utc_timestamp
from plag_in.engines.base import EngineAdapter
from plag_in.errors import (
    AuthenticationError,
    BackendResponseError,
    BackendUnavailableError,
    InvalidRequestError,
    LocalityPolicyError,
    OverloadedError,
    PayloadTooLargeError,
    PlagInError,
    PortConflictError,
    ProcessIdentityError,
    ReceiptPersistenceError,
    RequestTimeoutError,
    UnsupportedCapabilityError,
)
from plag_in.identity import runtime_source_identity
from plag_in.receipts import (
    INTEGRITY_MODE,
    LOCALITY_EVIDENCE_CLASS,
    RECEIPT_SCHEMA_VERSION,
    RECEIPT_SPEC_VERSION,
    Receipt,
    ReceiptStore,
)
from plag_in.registry import Registry

if TYPE_CHECKING:  # annotation only; active_store imports paths, which imports receipts
    from plag_in.active_store import ActiveReceiptStore

_RECEIPT_RE = re.compile(r"^/plag-in/v1/receipts/([^/]+)$")
_UNSUPPORTED_ENDPOINTS = ("/v1/completions", "/v1/embeddings", "/v1/messages")
_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
_MAX_TOKENS_UPPER_BOUND = 1_000_000  # generic sanity bound; an adapter also enforces its effective context

MAX_REQUEST_BODY_BYTES = 1_000_000  # 1 MB; documented default for text-only MVP use
MAX_CONCURRENT_CHAT_REQUESTS = 8
MAX_CONCURRENT_CONNECTIONS = 32  # server-boundary admission bound, ahead of any body read (R8)
BODY_READ_TIMEOUT_S = 15.0

# A refusal that closes the connection while the client is still sending its
# body reaches the client as a broken pipe rather than as the refusal: the
# client's write fails before it reads the response it was sent. So a refused
# body is discarded for a bounded interval, which lets the client finish
# writing and read the typed answer. These bounds are what keep the discard
# from being a cost the refusal was supposed to avoid: the bytes are never
# accumulated, at most `MAX_DISCARD_BYTES` are read, and the discard stops at
# `DISCARD_TIMEOUT_S` whatever remains.
MAX_DISCARD_BYTES = 8 * 1024 * 1024
DISCARD_TIMEOUT_S = 2.0
DISCARD_CHUNK_BYTES = 65536
_CAPABILITY_RANK = {"unknown": 0, "inferred": 1, "declared": 2, "tested": 3}

# The exact implemented subset of each request protocol, published so a
# client is told what is supported rather than discovering it by failure.
# Both protocols reach the same generation path; neither is ranked above
# the other, and neither is named after a harness.
CHAT_COMPLETIONS_SUBSET = {
    "path": "/v1/chat/completions",
    "message_content": ["string"],
    "streaming": "buffered_single_delta",
    "supported_fields": [
        "model",
        "messages",
        "stream",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
    ],
    "unsupported_message_fields": [
        "image_url",
        "input_audio",
        "input_video",
        "multimodal_data",
    ],
    "unsupported_message_content": ["structured_content_parts"],
    "tool_call_execution": "absent",
    "tool_call_transport": "backend_fields_forwarded_verbatim",
    "conversation_state": "absent",
}

RESPONSES_SUBSET = {
    "path": "/v1/responses",
    "input_forms": ["string", "message_items_of_input_text"],
    "output_forms": ["message_item_of_output_text"],
    "streaming": "buffered_single_delta",
    "supported_fields": [
        "model",
        "input",
        "instructions",
        "stream",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
    ],
    "unsupported_fields": [
        "tools",
        "tool_choice",
        "background",
        "previous_response_id",
        "conversation",
        "store",
        "include",
        "reasoning",
        "truncation",
        "text_format",
        "parallel_tool_calls",
    ],
    "unsupported_input_content_types": ["input_image", "input_file", "input_audio"],
    "tool_execution": "absent",
    "agent_loop": "absent",
    "conversation_state": "absent",
    "response_storage": "absent",
}

# Present with a value the route will not honour, these are refused rather
# than ignored: a silently dropped field would let a client believe a
# semantic was applied that never was.
_RESPONSES_REFUSED_FIELDS = (
    "tools",
    "tool_choice",
    "previous_response_id",
    "conversation",
    "include",
    "reasoning",
    "truncation",
    "text",
    "parallel_tool_calls",
    "prompt",
    "attachments",
)
_RESPONSES_SUPPORTED_INPUT_CONTENT_TYPES = {"input_text", "text", "output_text"}
_RESPONSES_INPUT_ROLES = {"system", "developer", "user", "assistant"}
_MAX_OUTPUT_TOKENS_UPPER_BOUND = _MAX_TOKENS_UPPER_BOUND


def _common_capability_state(values: list[str]) -> str:
    """Return the weakest evidence state shared by all selected backends."""
    if not values:
        return "unknown"
    if any(value not in _CAPABILITY_RANK for value in values):
        return "unknown"
    return min(values, key=_CAPABILITY_RANK.__getitem__)


@dataclass(frozen=True)
class BackendInfo:
    """What the gateway knows about one served alias's backend.

    `inference_mode` distinguishes the worker (`llama_server_direct`) and
    embedded (`libllama_embedded`) adapters; the identity fields below are
    populated according to which mode this is (handoff 07 item C). Only
    `executable_digest`/`argv_digest` carry worker-era meaning; only
    `native_identity_digest`/`native_component_digests`/`upstream_identity`
    carry embedded-library meaning. Neither pair is repurposed for the
    other adapter.
    """

    engine: str
    engine_version: str
    host: str
    port: int
    weight_digest: str
    config_digest: str
    template_digest: str = "n/a"
    inference_mode: str = "worker"
    executable_digest: str | None = None
    argv_digest: str | None = None
    native_identity_digest: str | None = None
    native_component_digests: dict | None = None
    upstream_identity: str | None = None
    requested_runtime_profile: dict | None = None
    effective_runtime_profile: dict | None = None
    measurement_status: dict | None = None
    gpu_offload: dict | None = None
    runtime_profile: dict | None = None
    backend_api_key: str | None = None
    inference_backend: object | None = None
    # Bound once when the backend is registered, derived by the registering
    # sink from the exact model bytes it resolved. A request that succeeds
    # never raises it: a model is `tested` only through an exact reviewed
    # profile (D-021).
    compatibility_status: str = "unverified"
    # What that state was decided on: the reason, the reviewed record and
    # the recorded and observed model digests. Carried onto every surface
    # that reports the state, so a reader is never left to infer what was
    # compared (independent review of repair 2, F1-R3).
    compatibility_evidence: dict | None = None


@dataclass(frozen=True)
class BufferedEventStream:
    """A completed chat response encoded as an SSE-compatible byte stream."""

    data: bytes


def _portable_runtime_profile(backend: BackendInfo) -> dict:
    """Build the Inference Receipt Specification section 5 `runtime_profile`.

    One shape for both adapters. The worker adapter forwards to an external
    process and queries no effective value, so its `effective` and
    `measurement_status` are empty objects: nothing was measured, and section 5
    is satisfied vacuously because every key in `effective` has a status. The
    reason it measures nothing is not discarded, it stays in the adapter's own
    record under `plag_in_adapter_record`.

    Until schema version 4 the worker adapter put its record directly in this
    field, which produced a receipt the reference verifier rejected
    (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F2).
    """
    measurement = backend.measurement_status if isinstance(backend.measurement_status, dict) else {}
    return {
        "requested": dict(backend.requested_runtime_profile or {}),
        "effective": dict(backend.effective_runtime_profile or {}),
        "measurement_status": dict(measurement),
    }


class GatewayContext:
    def __init__(
        self,
        config: GatewayConfig,
        registry: Registry,
        # Either a store bound to one path, or the pointer-following writer
        # that rebinds when a current-schema store is activated (D2).
        receipt_store: "ReceiptStore | ActiveReceiptStore",
        engine_adapters: dict[str, EngineAdapter],
        gateway_version: str = "0.1.0-mvp",
    ):
        self.config = config
        self.registry = registry
        self.receipts = receipt_store
        self.engine_adapters = engine_adapters
        self.gateway_version = gateway_version
        self.backends: dict[str, BackendInfo] = {}
        self._request_counter = itertools.count(1)
        self._inflight = threading.BoundedSemaphore(MAX_CONCURRENT_CHAT_REQUESTS)
        # Effective bind defaults to the declared configuration until a real
        # listener reports where it actually bound (R5); a configured port
        # of 0 must never be reported as the observed listening address.
        self._effective_bind_host = config.bind.host
        self._effective_bind_port = config.bind.port
        # Source identity snapshot taken once at initialisation (CPO-03).
        # It is reverified before status and before every receipt; drift
        # fails closed rather than letting a receipt claim the earlier
        # source state. Stored values are scalar, so a shallow copy handed
        # to a caller cannot mutate this snapshot.
        self._source_snapshot = runtime_source_identity()

    def _verified_source_snapshot(self) -> dict:
        """Return the init-time source snapshot, failing closed on drift.

        Recomputes the runtime source identity and compares it to the
        snapshot bound at initialisation. Any change to the digest, scope or
        file count raises a typed error so no status document or receipt can
        report a source state that no longer matches the running tree.
        """
        current = runtime_source_identity()
        snap = self._source_snapshot
        if (
            current["digest"] != snap["digest"]
            or current["scope"] != snap["scope"]
            or current["file_count"] != snap["file_count"]
        ):
            raise ProcessIdentityError(
                "runtime source tree changed since gateway initialisation; refusing to serve",
                expected_digest=snap["digest"],
                observed_digest=current["digest"],
            )
        return dict(snap)

    def set_effective_bind(self, host: str, port: int) -> None:
        self._effective_bind_host = host
        self._effective_bind_port = port

    def register_backend(self, alias: str, info: BackendInfo) -> None:
        validate_alias(alias)
        if not is_loopback(info.host):
            raise LocalityPolicyError(
                "refusing to register a non-loopback inference backend", alias=alias, host=info.host
            )
        self.backends[alias] = info

    def new_request_id(self) -> str:
        return f"req-{next(self._request_counter)}-{secrets.token_hex(4)}"

    def locality_level(self) -> str:
        if not is_loopback(self._effective_bind_host):
            return "L0"
        if not self.backends:
            return "L0"  # configuration inspected only; nothing measured
        if all(is_loopback(b.host) for b in self.backends.values()):
            return "L1"
        return "L0"

    # -- auth -----------------------------------------------------------

    def authenticate(self, headers, client_host: str) -> ApiKey | None:
        if self.config.security.auth_mode == "none":
            return None
        auth_header = headers.get("Authorization", "") if headers else ""
        if not auth_header.startswith("Bearer "):
            raise AuthenticationError("missing or malformed Authorization header")
        token = auth_header[len("Bearer "):]
        for key in self.config.security.api_keys:
            if hmac.compare_digest(key.secret, token):
                self._check_expiry(key)
                self._check_origin(key, client_host)
                return key
        raise AuthenticationError("invalid API key")

    def _check_expiry(self, key: ApiKey) -> None:
        if key.expiry is None:
            return
        expiry_utc = parse_utc_timestamp(key.expiry)
        if datetime.now(timezone.utc) >= expiry_utc:
            raise AuthenticationError("API key has expired", key_id=key.id)

    def _check_origin(self, key: ApiKey, client_host: str) -> None:
        if key.origin == "any":
            return
        if key.origin == "loopback" and not is_loopback(client_host):
            raise AuthenticationError(
                "API key is scoped to loopback clients", key_id=key.id, client_host=client_host
            )

    def check_scope(self, api_key: ApiKey | None, endpoint: str, alias: str | None = None) -> None:
        if api_key is None:
            return
        if api_key.endpoints and endpoint not in api_key.endpoints:
            raise AuthenticationError("endpoint not permitted for this key", endpoint=endpoint)
        if alias and api_key.aliases and alias not in api_key.aliases:
            raise AuthenticationError("model alias not permitted for this key", alias=alias)

    def authorized_aliases(self, api_key: ApiKey | None) -> list[str]:
        all_aliases = self.registry.aliases()
        if api_key is None or not api_key.aliases:
            return all_aliases
        return [a for a in all_aliases if a in api_key.aliases]

    # -- documents --------------------------------------------------------

    def capabilities_document(self, api_key: ApiKey | None) -> dict:
        authorized = set(self.authorized_aliases(api_key))
        models = {}
        for alias in self.registry.aliases():
            if alias not in authorized:
                continue
            entry = self.registry.resolve(alias)
            adapter = self.engine_adapters.get(entry.engine)
            models[alias] = adapter.capabilities().as_dict() if adapter else {}
        served_models = [
            models[alias]
            for alias in sorted(self.backends)
            if alias in authorized and alias in models
        ]
        generation_state = _common_capability_state(
            [model.get("chat_completions", "unknown") for model in served_models]
        )
        return {
            "protocol_version": "0.1",
            "endpoints": {
                "chat_completions": generation_state,
                "models_list": "tested",
                "completions": "unknown",
                "embeddings": "unknown",
                # The Responses route reaches the same backend generation
                # call as Chat Completions, so it carries the same evidence
                # state and never a better one. What differs between the two
                # is the request and response shape, published below.
                "responses": generation_state,
                "messages": "unknown",
            },
            "protocol_subsets": {
                "chat_completions": dict(CHAT_COMPLETIONS_SUBSET),
                "responses": dict(RESPONSES_SUBSET),
            },
            "models": models,
            "model_compatibility": {
                alias: self.backends[alias].compatibility_status
                for alias in sorted(self.backends)
                if alias in authorized
            },
            "model_compatibility_evidence": {
                alias: self.backends[alias].compatibility_evidence
                for alias in sorted(self.backends)
                if alias in authorized
            },
            "streaming_status": "tested_buffered",
            "tool_call_transport_status": _common_capability_state(
                [model.get("tool_call_transport", "unknown") for model in served_models]
            ),
            "locality_enforcement_level": self.locality_level(),
            "receipt_support": True,
            "receipt_integrity_mode": INTEGRITY_MODE,
            "receipt_spec_version": RECEIPT_SPEC_VERSION,
        }

    def status_document(self, api_key: ApiKey | None) -> dict:
        authorized = set(self.authorized_aliases(api_key))
        served = sorted(a for a in self.backends if a in authorized)
        level = self.locality_level()
        return {
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "receipt_spec_version": RECEIPT_SPEC_VERSION,
            # Identity of the running PLAG IN source, scope-labelled so it
            # reads as a runtime-source digest and not a repository digest;
            # reverified against the init-time snapshot (fails closed on
            # drift) and the same value is bound into every receipt.
            "runtime_source": self._verified_source_snapshot(),
            "bind": {"host": self._effective_bind_host, "port": self._effective_bind_port},
            "locality_enforcement_level": level,
            "locality_evidence_class": LOCALITY_EVIDENCE_CLASS[level],
            "auth_mode": self.config.security.auth_mode,
            "served_aliases": served,
            # Read from the registered backend, which is built once from the
            # confirmed profile. A completed request never edits it, so a
            # trial that answers cannot report itself as tested service.
            "model_compatibility": {
                alias: self.backends[alias].compatibility_status for alias in served
            },
            "model_compatibility_evidence": {
                alias: self.backends[alias].compatibility_evidence for alias in served
            },
            "service_class": {
                alias: (
                    "tested_profile_service"
                    if self.backends[alias].compatibility_status == "tested"
                    else "unverified_local_trial"
                )
                for alias in served
            },
            # A successful generation never mutates these entries: each is
            # built once at load time (embedded) or registration time
            # (worker) and copied here, so an `unassessed` measurement
            # cannot silently become `measured` by virtue of a request
            # having succeeded (handoff 07 item E).
            "runtime_profiles": {
                alias: {
                    **(self.backends[alias].runtime_profile or {}),
                    "inference_mode": self.backends[alias].inference_mode,
                    "native_identity_digest": self.backends[alias].native_identity_digest,
                    "requested_runtime_profile": self.backends[alias].requested_runtime_profile,
                    "effective_runtime_profile": self.backends[alias].effective_runtime_profile,
                    "measurement_status": self.backends[alias].measurement_status,
                    "gpu_offload": self.backends[alias].gpu_offload,
                    "compatibility_status": self.backends[alias].compatibility_status,
                }
                for alias in served
            },
            "receipt_integrity_mode": INTEGRITY_MODE,
        }

    # -- routing ----------------------------------------------------------

    def handle(
        self,
        method: str,
        path: str,
        headers,
        body_bytes: bytes,
        client_host: str,
        cancel_check: Callable[[], bool] | None = None,
    ):
        try:
            return self._route(
                method, path, headers, body_bytes, client_host, cancel_check
            )
        except PlagInError as exc:
            return exc.http_status, exc.to_dict(), {}

    def _route(
        self,
        method: str,
        path: str,
        headers,
        body_bytes: bytes,
        client_host: str,
        cancel_check: Callable[[], bool] | None,
    ):
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}, {}
        if method == "GET" and path == "/ready":
            return self._ready()
        if method == "GET" and path == "/v1/models":
            api_key = self.authenticate(headers, client_host)
            self.check_scope(api_key, "models_list")
            aliases = self.authorized_aliases(api_key)
            return 200, {"object": "list", "data": [{"id": a, "object": "model"} for a in aliases]}, {}
        if method == "POST" and path == "/v1/chat/completions":
            api_key = self.authenticate(headers, client_host)
            body = _parse_json_body(body_bytes)
            return self._chat_completions(api_key, body, cancel_check)
        if method == "POST" and path == "/v1/responses":
            api_key = self.authenticate(headers, client_host)
            body = _parse_json_body(body_bytes)
            return self._responses(api_key, body, cancel_check)
        if method == "POST" and path in _UNSUPPORTED_ENDPOINTS:
            self.authenticate(headers, client_host)
            raise UnsupportedCapabilityError(
                f"endpoint is not in the tested capability set: {path}", endpoint=path
            )
        if method == "GET" and path == "/plag-in/v1/capabilities":
            api_key = self.authenticate(headers, client_host)
            self.check_scope(api_key, "capabilities")
            return 200, self.capabilities_document(api_key), {}
        if method == "GET" and path == "/plag-in/v1/status":
            api_key = self.authenticate(headers, client_host)
            self.check_scope(api_key, "status")
            return 200, self.status_document(api_key), {}
        match = _RECEIPT_RE.match(path)
        if method == "GET" and match:
            api_key = self.authenticate(headers, client_host)
            self.check_scope(api_key, "receipts")
            record = self.receipts.get(match.group(1))
            self.check_scope(api_key, "receipts", alias=record.get("model_alias"))
            return 200, record, {}
        raise InvalidRequestError(f"no such route: {method} {path}")

    def _ready(self):
        if not self.backends:
            return 503, {"status": "not_ready", "reason": "no engine served"}, {}
        for backend in self.backends.values():
            if backend.inference_backend is not None:
                if not backend.inference_backend.ready():
                    return 503, {"status": "not_ready"}, {}
                continue
            url = f"http://{backend.host}:{backend.port}/health"
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:  # noqa: S310
                    if resp.status != 200:
                        return 503, {"status": "not_ready"}, {}
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                return 503, {"status": "not_ready"}, {}
        return 200, {"status": "ready"}, {}

    def _require_backend(self, alias: str) -> BackendInfo:
        backend = self.backends.get(alias)
        if backend is None:
            raise InvalidRequestError(f"model alias not being served: {alias}", alias=alias)
        return backend

    def _generate(
        self,
        alias: str,
        backend: BackendInfo,
        backend_body: dict,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[dict, str, dict]:
        """Run one bounded backend generation and record its receipt.

        Both request protocols pass through here, so admission, receipt
        writing and failure handling cannot drift apart between them.
        """
        acquired = self._inflight.acquire(blocking=False)
        if not acquired:
            raise OverloadedError(
                f"too many concurrent chat requests (limit {MAX_CONCURRENT_CHAT_REQUESTS})"
            )
        try:
            request_id = self.new_request_id()
            start = time.monotonic()
            try:
                response_body = self._forward_chat(backend, backend_body, cancel_check)
            except (BackendUnavailableError, BackendResponseError) as backend_exc:
                backend_exc.fields["request_id"] = request_id
                try:
                    failed_record = self._append_receipt(
                        request_id=request_id, alias=alias, backend=backend,
                        start=start, usage={}, status="failed",
                    )
                except ReceiptPersistenceError as receipt_exc:
                    # No receipt exists: supplying a receipt_id here would
                    # assert an audit record that was never persisted (R4).
                    raise ReceiptPersistenceError(
                        "backend request failed and the failure receipt could not be persisted; "
                        "no receipt was recorded for this request",
                        request_id=request_id,
                        backend_error_type=backend_exc.error_type,
                        terminal_status="incomplete",
                    ) from receipt_exc
                backend_exc.fields["receipt_id"] = failed_record["request_id"]
                raise
            latency_us = round((time.monotonic() - start) * 1_000_000)
            usage = response_body.get("usage", {}) if isinstance(response_body, dict) else {}
            record = self._append_receipt(
                request_id=request_id, alias=alias, backend=backend,
                start=start, usage=usage, status="completed", latency_us=latency_us,
            )
        finally:
            self._inflight.release()
        return response_body, request_id, record

    def _response_headers(self, request_id: str, record: dict) -> dict:
        return {
            "X-PLAG-IN-Request-ID": request_id,
            "X-PLAG-IN-Receipt-ID": record["request_id"],
            "X-PLAG-IN-Locality-Level": self.locality_level(),
        }

    def _chat_completions(
        self,
        api_key: ApiKey | None,
        body: dict,
        cancel_check: Callable[[], bool] | None,
    ):
        _validate_text_only_chat_body(body)
        stream_requested = body.get("stream", False)
        alias = body.get("model")
        if not alias:
            raise InvalidRequestError("missing 'model' field")
        validate_alias(alias)
        self.check_scope(api_key, "chat_completions", alias)
        backend = self._require_backend(alias)

        backend_body = dict(body)
        if stream_requested:
            backend_body["stream"] = False
        response_body, request_id, record = self._generate(
            alias, backend, backend_body, cancel_check
        )
        stream_body = _buffered_event_stream(response_body) if stream_requested else None
        return 200, stream_body or response_body, self._response_headers(request_id, record)

    def _responses(
        self,
        api_key: ApiKey | None,
        body: dict,
        cancel_check: Callable[[], bool] | None,
    ):
        """Serve the bounded text-only Responses subset over the same backend."""
        request = _validate_responses_body(body)
        alias = request["model"]
        self.check_scope(api_key, "responses", alias)
        backend = self._require_backend(alias)

        backend_body = {"model": alias, "messages": request["messages"]}
        for source, target in (
            ("max_output_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("seed", "seed"),
        ):
            if source in request:
                backend_body[target] = request[source]
        _validate_text_only_chat_body(backend_body)

        response_body, request_id, record = self._generate(
            alias, backend, backend_body, cancel_check
        )
        payload = _responses_document(alias, response_body)
        if request["stream"]:
            return 200, _buffered_responses_stream(payload), self._response_headers(
                request_id, record
            )
        return 200, payload, self._response_headers(request_id, record)

    def _append_receipt(self, *, request_id, alias, backend, start, usage, status, latency_us=None):
        level = self.locality_level()
        source_identity = self._verified_source_snapshot()
        # PLAG IN's own namespaced record, so the compatibility state and the
        # effective model identity ride an existing field and the receipt
        # schema is unchanged (D-021). Read from the registered backend, which
        # a completed request never edits.
        adapter_record = dict(backend.runtime_profile or {})
        adapter_record["compatibility_status"] = backend.compatibility_status
        adapter_record["compatibility_evidence"] = backend.compatibility_evidence
        adapter_record["effective_model_identity"] = {
            "model_alias": alias,
            "weight_digest": backend.weight_digest,
            "template_digest": backend.template_digest,
            "engine": backend.engine,
            "inference_mode": backend.inference_mode,
        }
        receipt = Receipt(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gateway_version=self.gateway_version,
            schema_version=RECEIPT_SCHEMA_VERSION,
            engine=backend.engine,
            engine_version=backend.engine_version,
            inference_mode=backend.inference_mode,
            model_alias=alias,
            weight_digest=backend.weight_digest,
            template_digest=backend.template_digest,
            config_digest=backend.config_digest,
            engine_executable_digest=backend.executable_digest,
            argv_digest=backend.argv_digest,
            native_identity_digest=backend.native_identity_digest,
            native_component_digests=backend.native_component_digests,
            upstream_identity=backend.upstream_identity,
            requested_runtime_profile=backend.requested_runtime_profile,
            effective_runtime_profile=backend.effective_runtime_profile,
            measurement_status=backend.measurement_status,
            gpu_offload=backend.gpu_offload,
            locality_level=level,
            locality_evidence_class=LOCALITY_EVIDENCE_CLASS[level],
            route="local",
            listen_address=f"{self._effective_bind_host}:{self._effective_bind_port}",
            backend_address=(
                "in_process" if backend.inference_backend is not None
                else f"{backend.host}:{backend.port}"
            ),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_us=latency_us if latency_us is not None else round((time.monotonic() - start) * 1_000_000),
            status=status,
            content_retained=False,
            runtime_profile=_portable_runtime_profile(backend),
            plag_in_adapter_record=adapter_record,
            runtime_source_digest=source_identity["digest"],
            runtime_source_scope=source_identity["scope"],
        )
        try:
            return self.receipts.append(receipt)
        except ReceiptPersistenceError:
            # A receipt-write failure must never be reported as a fully
            # audited response, success or failure alike.
            raise

    def _forward_chat(
        self,
        backend: BackendInfo,
        body: dict,
        cancel_check: Callable[[], bool] | None,
    ) -> dict:
        if backend.inference_backend is not None:
            if getattr(backend.inference_backend, "supports_cancel_check", False):
                response = backend.inference_backend.chat_completion(
                    body, cancel_check=cancel_check
                )
            else:
                response = backend.inference_backend.chat_completion(body)
            if not isinstance(response, dict):
                raise BackendResponseError("embedded backend response must be a JSON object")
            return response
        url = f"http://{backend.host}:{backend.port}/v1/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
        req.add_header("Content-Type", "application/json")
        if backend.backend_api_key:
            req.add_header("Authorization", f"Bearer {backend.backend_api_key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise BackendUnavailableError(f"backend request failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendResponseError(f"backend returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BackendResponseError("backend response must be a JSON object")
        return parsed


def _parse_json_body(body_bytes: bytes) -> dict:
    try:
        parsed = json.loads(body_bytes or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError(f"request body is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise InvalidRequestError("request body must be a JSON object")
    return parsed


def _buffered_event_stream(response: dict) -> BufferedEventStream:
    """Encode one completed response as ordered Chat Completions SSE events."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendResponseError("backend response has no choices for streaming")
    common = {
        key: response[key]
        for key in ("id", "created", "model", "system_fingerprint")
        if key in response
    }
    common["object"] = "chat.completion.chunk"
    content_choices = []
    finish_choices = []
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise BackendResponseError("backend response choice is invalid for streaming")
        index = choice.get("index", position)
        content_choices.append(
            {
                "index": index,
                "delta": dict(choice["message"]),
                "finish_reason": None,
            }
        )
        finish_choices.append(
            {
                "index": index,
                "delta": {},
                "finish_reason": choice.get("finish_reason"),
            }
        )
    events = [
        {**common, "choices": content_choices},
        {**common, "choices": finish_choices},
    ]
    data = b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n\n"
        for event in events
    ) + b"data: [DONE]\n\n"
    return BufferedEventStream(data)


def _validate_responses_body(body: dict) -> dict:
    """Total, typed validation of a Responses request before any backend call.

    Every field the route will not honour is refused rather than dropped,
    including tools, conversation state, stored responses and non-text
    input parts. A client is told exactly what is unsupported instead of
    receiving an answer that quietly ignored half its request.
    """
    alias = body.get("model")
    if not alias:
        raise InvalidRequestError("missing 'model' field")
    validate_alias(alias)

    for field in _RESPONSES_REFUSED_FIELDS:
        value = body.get(field)
        if value in (None, [], {}, ""):
            continue
        if field == "tool_choice" and value == "none":
            continue
        raise UnsupportedCapabilityError(
            f"'{field}' is not supported by the bounded text-only Responses subset",
            field=field,
            supported_subset=dict(RESPONSES_SUBSET),
        )
    for field in ("background", "store"):
        if body.get(field):
            raise UnsupportedCapabilityError(
                f"'{field}' is not supported by the bounded text-only Responses subset",
                field=field,
                supported_subset=dict(RESPONSES_SUBSET),
            )

    if "stream" in body and not isinstance(body["stream"], bool):
        raise InvalidRequestError("'stream' must be a boolean")

    messages: list[dict] = []
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise InvalidRequestError("'instructions' must be a string")
        messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        if not raw_input:
            raise InvalidRequestError("'input' must not be an empty array")
        for index, item in enumerate(raw_input):
            messages.append(_responses_input_message(item, index))
    else:
        raise InvalidRequestError("'input' must be a string or an array of input items")

    request: dict = {"model": alias, "messages": messages, "stream": bool(body.get("stream", False))}

    if "max_output_tokens" in body and body["max_output_tokens"] is not None:
        max_output_tokens = body["max_output_tokens"]
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
            raise InvalidRequestError("'max_output_tokens' must be a positive integer")
        if max_output_tokens <= 0 or max_output_tokens > _MAX_OUTPUT_TOKENS_UPPER_BOUND:
            raise InvalidRequestError("'max_output_tokens' must be a positive bounded integer")
        request["max_output_tokens"] = max_output_tokens
    for field in ("temperature", "top_p", "seed"):
        if field in body and body[field] is not None:
            request[field] = body[field]
    return request


def _responses_input_message(item, index: int) -> dict:
    """Translate one Responses input item into one chat message."""
    if not isinstance(item, dict):
        raise InvalidRequestError("each input item must be a JSON object", input_index=index)
    item_type = item.get("type", "message")
    if item_type != "message":
        raise UnsupportedCapabilityError(
            "only message input items are supported by the bounded Responses subset",
            input_index=index,
            item_type=item_type,
            supported_subset=dict(RESPONSES_SUBSET),
        )
    role = item.get("role")
    if role not in _RESPONSES_INPUT_ROLES:
        raise InvalidRequestError(
            "input item has an unsupported or missing role", input_index=index
        )
    # `developer` has no distinct meaning for a local chat template, so it
    # is carried as `system` rather than dropped or invented.
    chat_role = "system" if role == "developer" else role

    content = item.get("content")
    if isinstance(content, str):
        return {"role": chat_role, "content": content}
    if not isinstance(content, list) or not content:
        raise InvalidRequestError(
            "input item content must be a string or a non-empty array", input_index=index
        )
    parts: list[str] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise InvalidRequestError(
                "each input content part must be a JSON object",
                input_index=index,
                content_index=part_index,
            )
        part_type = part.get("type")
        if part_type not in _RESPONSES_SUPPORTED_INPUT_CONTENT_TYPES:
            raise UnsupportedCapabilityError(
                "non-text input content is not supported by the bounded Responses subset",
                input_index=index,
                content_index=part_index,
                content_type=part_type,
                supported_subset=dict(RESPONSES_SUBSET),
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise InvalidRequestError(
                "input content part text must be a string",
                input_index=index,
                content_index=part_index,
            )
        parts.append(text)
    return {"role": chat_role, "content": "".join(parts)}


def _responses_document(alias: str, response: dict) -> dict:
    """Build one Responses object from a completed chat response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendResponseError("backend response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise BackendResponseError("backend response choice is invalid")
    content = first["message"].get("content")
    if not isinstance(content, str):
        raise BackendResponseError("backend response content is not text")

    response_id = str(response.get("id") or f"resp-plag-{time.time_ns()}")
    created_at = response.get("created")
    if not isinstance(created_at, int) or isinstance(created_at, bool):
        created_at = int(time.time())
    finish_reason = first.get("finish_reason")
    incomplete = (
        {"reason": "max_output_tokens"} if finish_reason == "length" else None
    )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    document = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": alias,
        "output": [
            {
                "id": f"msg-{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        "output_text": content,
        "incomplete_details": incomplete,
        "error": None,
        # Stated on every response so a client is never left to infer these
        # from a silent success.
        "tools": [],
        "tool_choice": "none",
        "store": False,
        "parallel_tool_calls": False,
    }
    if usage:
        document["usage"] = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return document


def _buffered_responses_stream(document: dict) -> BufferedEventStream:
    """Encode one completed Responses object as ordered Responses SSE events.

    The backend produces a complete answer before any event is written, so
    the text arrives as one delta. The event order, the identifiers and the
    terminal `response.completed` are the ones a streaming client expects,
    and no event is emitted for a stage that did not happen.
    """
    item = document["output"][0]
    text = item["content"][0]["text"]
    in_progress = {**document, "status": "in_progress", "output": []}
    pending_item = {**item, "status": "in_progress", "content": []}
    events = [
        ("response.created", {"response": {**document, "status": "in_progress", "output": []}}),
        ("response.in_progress", {"response": in_progress}),
        ("response.output_item.added", {"output_index": 0, "item": pending_item}),
        (
            "response.content_part.added",
            {
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
        (
            "response.output_text.delta",
            {"item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text},
        ),
        (
            "response.output_text.done",
            {"item_id": item["id"], "output_index": 0, "content_index": 0, "text": text},
        ),
        (
            "response.content_part.done",
            {
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        ),
        ("response.output_item.done", {"output_index": 0, "item": item}),
        ("response.completed", {"response": document}),
    ]
    chunks = []
    for sequence_number, (event_type, payload) in enumerate(events):
        body = {"type": event_type, "sequence_number": sequence_number, **payload}
        chunks.append(
            b"event: " + event_type.encode("ascii")
            + b"\ndata: " + json.dumps(body, separators=(",", ":")).encode("utf-8")
            + b"\n\n"
        )
    return BufferedEventStream(b"".join(chunks))


def _validate_text_only_chat_body(body: dict) -> None:
    """Total, typed validation of the chat request before any backend call.

    Every check here runs ahead of tokenisation or decoding for every
    engine (worker or embedded), so a rejected request never reaches
    model decoding and never produces a completed-success receipt.
    Refuses media structures that can make llama-server fetch a URL or
    file, and refuses malformed sampling fields that would otherwise be
    silently coerced (`bool` is an `int` subclass in Python, so `True`
    would pass an unguarded `int()` conversion as `1`).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("'messages' must be a non-empty JSON array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidRequestError("each message must be a JSON object", message_index=index)
        role = message.get("role")
        if role not in _ALLOWED_MESSAGE_ROLES:
            raise InvalidRequestError(
                "message has an unsupported or missing role", message_index=index
            )
        content = message.get("content")
        if isinstance(content, list):
            raise UnsupportedCapabilityError(
                "structured media content is not supported by the text-only MVP",
                message_index=index,
            )
        if not isinstance(content, str):
            raise InvalidRequestError(
                "message content must be a string", message_index=index
            )
        for field in ("image_url", "input_audio", "input_video", "multimodal_data"):
            if field in message:
                raise UnsupportedCapabilityError(
                    "media input is not supported by the text-only MVP",
                    field=field,
                    message_index=index,
                )

    if "stream" in body and not isinstance(body["stream"], bool):
        raise InvalidRequestError("'stream' must be a boolean")

    if "max_tokens" in body:
        max_tokens = body["max_tokens"]
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise InvalidRequestError("'max_tokens' must be a positive integer")
        if max_tokens <= 0 or max_tokens > _MAX_TOKENS_UPPER_BOUND:
            raise InvalidRequestError("'max_tokens' must be a positive bounded integer")

    for field, low, high in (("top_p", 0.0, 1.0), ("temperature", 0.0, None)):
        if field not in body:
            continue
        value = body[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidRequestError(f"'{field}' must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise InvalidRequestError(f"'{field}' must be a finite number")
        if value < low or (high is not None and value > high):
            raise InvalidRequestError(f"'{field}' is out of the supported range")

    if "seed" in body:
        seed = body["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidRequestError("'seed' must be an integer")


def _make_handler(context: GatewayContext):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PLAGIN/0.1"
        timeout = BODY_READ_TIMEOUT_S

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass  # suppress default access logging; avoids incidental content in logs

        def _send(self, status: int, payload: dict | BufferedEventStream, extra_headers: dict) -> None:
            streaming = isinstance(payload, BufferedEventStream)
            data = payload.data if streaming else json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "text/event-stream" if streaming else "application/json",
                )
                if streaming:
                    self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                for key, value in extra_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _discard_refused_body(self, length: int) -> None:
            """Read and throw away the body of a request already refused.

            The refusal is what protects the backend and the receipt store; this
            protects the answer. A client that is still writing when the server
            closes sees its own write fail and never reads the response, so the
            typed refusal it was sent is lost to it.

            Three bounds keep this from becoming the cost the refusal avoids.
            Bytes are read into a fixed-size chunk and dropped, never
            accumulated, so memory does not follow the declared length. At most
            `MAX_DISCARD_BYTES` are read, so a declared length far beyond the
            limit is not honoured. The discard stops at `DISCARD_TIMEOUT_S`
            whatever remains, so a client that stalls cannot hold the connection
            open through it.
            """
            remaining = min(length, MAX_DISCARD_BYTES)
            deadline = time.monotonic() + DISCARD_TIMEOUT_S
            previous_timeout = None
            try:
                previous_timeout = self.connection.gettimeout()
                while remaining > 0:
                    budget = deadline - time.monotonic()
                    if budget <= 0:
                        return
                    self.connection.settimeout(budget)
                    chunk = self.rfile.read(min(remaining, DISCARD_CHUNK_BYTES))
                    if not chunk:
                        return
                    remaining -= len(chunk)
            except (OSError, ValueError):
                return
            finally:
                if previous_timeout is not None:
                    try:
                        self.connection.settimeout(previous_timeout)
                    except OSError:
                        pass

        def _client_disconnected(self) -> bool:
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (BlockingIOError, InterruptedError):
                return False
            except OSError:
                return True

        def _dispatch(self, method: str) -> None:
            try:
                length_header = self.headers.get("Content-Length")
                length = int(length_header) if length_header else 0
            except ValueError:
                exc = InvalidRequestError("malformed Content-Length header")
                self._send(exc.http_status, exc.to_dict(), {})
                return

            if length < 0:
                exc = InvalidRequestError("Content-Length must not be negative")
                self._send(exc.http_status, exc.to_dict(), {})
                return

            if length > MAX_REQUEST_BODY_BYTES:
                exc = PayloadTooLargeError(
                    f"request body exceeds the {MAX_REQUEST_BODY_BYTES}-byte MVP limit",
                    limit_bytes=MAX_REQUEST_BODY_BYTES,
                )
                self._send(exc.http_status, exc.to_dict(), {})
                # The refusal is sent first, then the client is allowed to
                # finish sending. Closing here instead left the client's write
                # failing with a broken pipe before it read the refusal, so an
                # oversized request intermittently produced no typed answer at
                # all. Nothing read here is kept or forwarded.
                self._discard_refused_body(length)
                return

            try:
                body = self.rfile.read(length) if length else b""
            except TimeoutError:
                exc = RequestTimeoutError("timed out reading the request body")
                self._send(exc.http_status, exc.to_dict(), {})
                return

            status, payload, extra_headers = context.handle(
                method,
                self.path,
                self.headers,
                body,
                self.client_address[0],
                self._client_disconnected,
            )
            self._send(status, payload, extra_headers)

        def do_GET(self):  # noqa: N802 - stdlib method name
            self._dispatch("GET")

        def do_POST(self):  # noqa: N802 - stdlib method name
            self._dispatch("POST")

    return Handler


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Bounds accepted-connection handler work before any body is read.

    `ThreadingHTTPServer` otherwise spawns one handler thread per accepted
    connection with no limit, so a slow or incomplete client can consume
    unbounded handler threads without ever reaching the in-app inference
    semaphore, which is only acquired after the request body has been
    fully read, decoded, authenticated and routed
    (CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md R8). This bounds admission
    at the point a connection is accepted: a connection beyond
    `max_connections` receives a typed `overloaded` response and is closed
    immediately, before any handler thread is created for it.
    """

    daemon_threads = True

    def __init__(self, server_address, handler_cls, *, max_connections: int):
        self._connection_semaphore = threading.BoundedSemaphore(max_connections)
        super().__init__(server_address, handler_cls)

    def process_request(self, request, client_address):
        if not self._connection_semaphore.acquire(blocking=False):
            self._reject_overloaded(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_semaphore.release()

    @staticmethod
    def _reject_overloaded(request) -> None:
        error = OverloadedError(
            f"too many concurrent connections at the server boundary (limit {MAX_CONCURRENT_CONNECTIONS})"
        )
        body = json.dumps(error.to_dict()).encode("utf-8")
        response = (
            b"HTTP/1.1 429 Too Many Requests\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n\r\n" + body
        )
        try:
            request.sendall(response)
        except OSError:
            pass
        finally:
            try:
                request.close()
            except OSError:
                pass


class GatewayServer:
    """Owns the listening socket. Refuses to construct an insecure exposure."""

    def __init__(self, context: GatewayContext):
        self.context = context
        config = context.config
        if not is_loopback(config.bind.host) and config.security.auth_mode == "none":
            raise LocalityPolicyError(
                "refusing to bind a non-loopback address without authentication",
                host=config.bind.host,
            )
        handler_cls = _make_handler(context)
        try:
            self._httpd = _BoundedThreadingHTTPServer(
                (config.bind.host, config.bind.port), handler_cls, max_connections=MAX_CONCURRENT_CONNECTIONS
            )
        except OSError as exc:
            raise PortConflictError(
                f"gateway failed to bind {config.bind.host}:{config.bind.port}: {exc}",
                host=config.bind.host,
                port=config.bind.port,
            ) from exc
        # The listener is now actually bound: record the effective address
        # (a configured port of 0 has now been assigned a real one) rather
        # than the pre-bind configured value (R5).
        context.set_effective_bind(*self._httpd.server_address[:2])
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # `shutdown()` waits for the `serve_forever` loop to notice the
        # shutdown flag; calling it before that loop has ever run would
        # block forever, so it is only safe once `start()` has actually
        # been called (D: close must be safe before and after serving).
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
        self._httpd.server_close()
