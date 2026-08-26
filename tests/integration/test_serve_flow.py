import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from plag_in.errors import BackendUnavailableError, RequestCancelledError
from plag_in.identity import RUNTIME_SOURCE_SCOPE
from tests.support import build_gateway_stack


class _CancellableEmbeddedFixture:
    supports_cancel_check = True

    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def ready(self):
        return True

    def chat_completion(self, body, *, cancel_check=None):
        self.started.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                self.cancelled.set()
                raise RequestCancelledError(
                    "fixture observed client disconnect",
                    reason="client_disconnected",
                )
            time.sleep(0.01)
        raise AssertionError("client disconnect was not observed")


def _get(base_url, path, headers=None):
    # Note: resp.headers is an email.message.Message, which resolves
    # header names case-insensitively on __getitem__/.get() - deliberately
    # not converted to a plain dict so header lookups below stay correct
    # regardless of the exact casing the server sent.
    req = urllib.request.Request(f"{base_url}{path}", headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read()), exc.headers
        finally:
            exc.close()


def _post(base_url, path, body, headers=None):
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=hdrs, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read()), exc.headers
        finally:
            exc.close()


class ServeFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stack = build_gateway_stack(Path(self.tmp.name), alias="fixture-alias")

    def tearDown(self):
        self.stack.stop()
        self.tmp.cleanup()

    def test_health(self):
        status, body, _headers = _get(self.stack.server.base_url, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_ready_reflects_backend_health(self):
        status, body, _headers = _get(self.stack.server.base_url, "/ready")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")

    def test_models_lists_the_served_alias(self):
        status, body, _headers = _get(self.stack.server.base_url, "/v1/models")
        self.assertEqual(status, 200)
        ids = [m["id"] for m in body["data"]]
        self.assertEqual(ids, ["fixture-alias"])

    def test_chat_completions_round_trip_and_receipt_headers(self):
        status, body, headers = _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(status, 200)
        self.assertIn("choices", body)
        self.assertIsNotNone(headers.get("X-PLAG-IN-Request-ID"))
        self.assertEqual(headers.get("X-PLAG-IN-Locality-Level"), "L1")

    def test_client_disconnect_cancels_embedded_backend_work(self):
        fixture = _CancellableEmbeddedFixture()
        current = self.stack.context.backends["fixture-alias"]
        self.stack.context.backends["fixture-alias"] = replace(
            current,
            inference_mode="embedded",
            inference_backend=fixture,
        )
        body = json.dumps(
            {
                "model": "fixture-alias",
                "messages": [{"role": "user", "content": "disconnect fixture"}],
            }
        ).encode("utf-8")
        host, port_text = self.stack.server.base_url.removeprefix("http://").split(":")
        client = socket.create_connection((host, int(port_text)), timeout=2)
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            + f"Host: {host}:{port_text}\r\n".encode("ascii")
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + body
        )
        client.sendall(request)
        self.assertTrue(fixture.started.wait(1))
        client.close()
        self.assertTrue(fixture.cancelled.wait(2))

    def test_direct_backend_refuses_bypass_without_its_private_key(self):
        status, body, _headers = _post(
            f"http://127.0.0.1:{self.stack.engine_port}",
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorised_backend_fixture")

    def test_backend_key_is_absent_from_status_and_receipts(self):
        backend_key = self.stack.context.backends["fixture-alias"].backend_api_key
        status, status_body, _headers = _get(self.stack.server.base_url, "/plag-in/v1/status")
        self.assertEqual(status, 200)
        _status, _body, _headers = _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertNotIn(backend_key, json.dumps(status_body))
        self.assertNotIn(backend_key, self.stack.receipts_path.read_text(encoding="utf-8"))

    def test_structured_media_is_refused_before_direct_backend_forwarding(self):
        status, body, _headers = _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {
                "model": "fixture-alias",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}],
                    }
                ],
            },
        )
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["type"], "unsupported_capability")

    def test_receipt_lookup_by_request_id(self):
        _status, _body, headers = _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        request_id = headers.get("X-PLAG-IN-Request-ID")
        status, receipt, _headers = _get(self.stack.server.base_url, f"/plag-in/v1/receipts/{request_id}")
        self.assertEqual(status, 200)
        self.assertEqual(receipt["model_alias"], "fixture-alias")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["runtime_profile"]["engine"], "llama_server_direct")
        self.assertFalse(receipt["runtime_profile"]["ollama_runtime_dependency"])
        self.assertEqual(receipt["runtime_profile"]["arguments"]["context_size"], 8192)
        # Regression 11: a worker receipt preserves its declared
        # compatibility (worker-era identity) fields under the new schema,
        # and carries no embedded-only native identity (handoff 07 item C).
        self.assertEqual(receipt["schema_version"], "3")
        self.assertEqual(receipt["inference_mode"], "worker")
        self.assertEqual(len(receipt["engine_executable_digest"]), 64)
        self.assertEqual(len(receipt["argv_digest"]), 64)
        self.assertIsNone(receipt["native_identity_digest"])
        self.assertIsNone(receipt["upstream_identity"])
        # Schema v3: the running source identity is bound into the receipt
        # with an explicit scope label (repair B).
        self.assertRegex(receipt["runtime_source_digest"], r"^[0-9a-f]{64}$")
        self.assertTrue(
            receipt["runtime_source_scope"].startswith("runtime_python_source:")
        )

    def test_status_and_receipt_report_the_same_runtime_source_digest(self):
        # Schema v3 agreement (repair B): status and every receipt written
        # by the same process report one identical runtime source digest.
        status, doc, _headers = _get(self.stack.server.base_url, "/plag-in/v1/status")
        self.assertEqual(status, 200)
        source = doc["runtime_source"]
        self.assertRegex(source["digest"], r"^[0-9a-f]{64}$")
        _post_status, _body, headers = _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        request_id = headers.get("X-PLAG-IN-Request-ID")
        _get_status, receipt, _h = _get(
            self.stack.server.base_url, f"/plag-in/v1/receipts/{request_id}"
        )
        self.assertEqual(receipt["runtime_source_digest"], source["digest"])
        self.assertEqual(receipt["runtime_source_scope"], source["scope"])

    def test_capabilities_document_shape(self):
        status, doc, _headers = _get(self.stack.server.base_url, "/plag-in/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertIn("fixture-alias", doc["models"])
        self.assertEqual(doc["models"]["fixture-alias"]["chat_completions"], "tested")
        self.assertEqual(doc["models"]["fixture-alias"]["completions"], "unknown")
        self.assertEqual(doc["endpoints"]["chat_completions"], "tested")
        self.assertEqual(doc["tool_call_transport_status"], "tested")
        self.assertEqual(doc["locality_enforcement_level"], "L1")

    def test_status_document_reports_bind_and_served_aliases(self):
        status, doc, _headers = _get(self.stack.server.base_url, "/plag-in/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(doc["served_aliases"], ["fixture-alias"])
        self.assertEqual(doc["bind"]["host"], "127.0.0.1")
        profile = doc["runtime_profiles"]["fixture-alias"]
        self.assertEqual(profile["engine"], "llama_server_direct")
        self.assertFalse(profile["ollama_runtime_dependency"])
        self.assertEqual(profile["arguments"]["parallel"], 1)
        # Handoff 07 items D and E: status exposes the schema version,
        # inference mode and per-field measurement status truthfully; a
        # requested GPU offload with no verified measurement route stays
        # `unassessed`, never inferred from a requested layer count.
        self.assertEqual(doc["receipt_schema_version"], "3")
        self.assertIn("locality_evidence_class", doc)
        self.assertEqual(profile["inference_mode"], "worker")
        self.assertEqual(profile["gpu_offload"]["status"], "unassessed")
        self.assertIsNone(profile["gpu_offload"]["observed"])

    def test_gpu_offload_stays_unassessed_after_a_successful_chat_response(self):
        # Regression 14: a successful generation must not turn an
        # `unassessed` measurement into `measured`.
        _post(
            self.stack.server.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        status, doc, _headers = _get(self.stack.server.base_url, "/plag-in/v1/status")
        self.assertEqual(status, 200)
        profile = doc["runtime_profiles"]["fixture-alias"]
        self.assertEqual(profile["gpu_offload"]["status"], "unassessed")

    def test_source_drift_fails_status_closed(self):
        # CPO-03: a source tree that changed since gateway init must fail
        # closed rather than report a stale snapshot.
        drifted = {"scope": RUNTIME_SOURCE_SCOPE, "digest": "0" * 64, "file_count": 1}
        with patch("plag_in.gateway.runtime_source_identity", return_value=drifted):
            status, _body, _headers = _get(self.stack.server.base_url, "/plag-in/v1/status")
        self.assertEqual(status, 409)

    def test_source_drift_prevents_completed_receipt(self):
        before = len(self.stack.receipts_path.read_text(encoding="utf-8").splitlines())
        drifted = {"scope": RUNTIME_SOURCE_SCOPE, "digest": "0" * 64, "file_count": 1}
        with patch("plag_in.gateway.runtime_source_identity", return_value=drifted):
            status, _body, _headers = _post(
                self.stack.server.base_url,
                "/v1/chat/completions",
                {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(status, 409)
        after = len(self.stack.receipts_path.read_text(encoding="utf-8").splitlines())
        self.assertEqual(after, before, "no receipt may be written on source drift")

    def test_source_drift_prevents_failed_backend_receipt(self):
        before = len(self.stack.receipts_path.read_text(encoding="utf-8").splitlines())
        drifted = {"scope": RUNTIME_SOURCE_SCOPE, "digest": "0" * 64, "file_count": 1}
        with (
            patch.object(
                self.stack.context,
                "_forward_chat",
                side_effect=BackendUnavailableError("fixture backend unavailable"),
            ),
            patch("plag_in.gateway.runtime_source_identity", return_value=drifted),
        ):
            status, body, _headers = _post(
                self.stack.server.base_url,
                "/v1/chat/completions",
                {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(status, 409)
        self.assertNotIn("receipt_id", body["error"].get("fields", {}))
        after = len(self.stack.receipts_path.read_text(encoding="utf-8").splitlines())
        self.assertEqual(after, before, "no failed receipt may be written on source drift")

    def test_status_runtime_source_is_a_copy(self):
        ctx = self.stack.context
        doc = ctx.status_document(None)
        doc["runtime_source"]["digest"] = "tampered"
        again = ctx.status_document(None)
        self.assertNotEqual(again["runtime_source"]["digest"], "tampered")

    def test_unsupported_endpoint_returns_typed_error(self):
        status, body, _headers = _post(self.stack.server.base_url, "/v1/embeddings", {"model": "fixture-alias", "input": "x"})
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["type"], "unsupported_capability")


if __name__ == "__main__":
    unittest.main()
