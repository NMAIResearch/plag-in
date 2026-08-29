"""Required regressions from CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md /
handoffs/04_SONNET_SECURITY_BOUNDARY_REPAIR.md (R1-R8, numbered regressions
1-15; regression 16 is the whole-suite ResourceWarning run performed and
reported separately, not a unit test). All fixtures are scratch: a tempdir,
the fake_engine.py fixture process and real loopback sockets. No host
service or live model store is used.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from plag_in.errors import AliasAlreadyRunningError, PortConflictError, ReceiptChainError, ReceiptPersistenceError
from plag_in.identity import RUNTIME_SOURCE_SCOPE
from plag_in.receipts import Receipt, ReceiptStore
from plag_in.supervisor import Supervisor

import plag_in.cli as cli_module
import plag_in.gateway as gateway_module

from tests.support import FAKE_ENGINE_PATH, build_gateway_stack, free_port, write_fixture_weight

_SRC_DIR = str(Path(__file__).parent.parent.parent / "src")


def _post(base_url, path, body=None, headers=None, raw_body=None):
    data = raw_body if raw_body is not None else json.dumps(body).encode("utf-8")
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


def _get(base_url, path, headers=None):
    req = urllib.request.Request(f"{base_url}{path}", headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read()), exc.headers
        finally:
            exc.close()


def _make_receipt(request_id: str, **overrides) -> Receipt:
    base = dict(
        request_id=request_id, timestamp="2026-08-24T00:00:00+00:00", gateway_version="0.1.0-mvp",
        engine="llama_server", engine_version="unknown", model_alias="fixture-alias",
        weight_digest="a" * 64, template_digest="b" * 64, config_digest="c" * 64,
        engine_executable_digest="d" * 64, argv_digest="e" * 64, locality_level="L1",
        route="local", listen_address="127.0.0.1:8080", backend_address="127.0.0.1:9000",
        input_tokens=1, output_tokens=1, latency_us=1, status="completed", content_retained=False,
        schema_version="5", runtime_source_digest="a" * 64,
        runtime_source_scope=RUNTIME_SOURCE_SCOPE,
    )
    base.update(overrides)
    return Receipt(**base)


def _serve_one_raw_response(port: int, raw_response: bytes) -> threading.Thread:
    def _serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        srv.settimeout(5)
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        try:
            conn.recv(65536)
            conn.sendall(raw_response)
        finally:
            conn.close()
            srv.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# R1 / regression 1: tail deletion is detected via the checkpoint
# ---------------------------------------------------------------------------


class TailDeletionDetectedByCheckpointTest(unittest.TestCase):
    def test_removing_the_final_record_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            store.append(_make_receipt("r2"))
            valid, count = store.verify_chain()
            self.assertTrue(valid)
            self.assertEqual(count, 2)

            lines = store.path.read_text().splitlines()
            store.path.write_text(lines[0] + "\n")  # drop the final record only

            valid, count = store.verify_chain()
            self.assertFalse(valid, "tail deletion must be detected once a checkpoint exists")


# ---------------------------------------------------------------------------
# R2 / regression 2: altered state cannot be returned or extended
# ---------------------------------------------------------------------------


class AlteredReceiptStateRefusedByGetAndAppendTest(unittest.TestCase):
    def test_get_and_append_both_raise_the_typed_integrity_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))

            lines = store.path.read_text().splitlines()
            record = json.loads(lines[0])
            record["model_alias"] = "tampered-alias"
            store.path.write_text(json.dumps(record) + "\n")

            with self.assertRaises(ReceiptChainError):
                store.get("r1")
            with self.assertRaises(ReceiptChainError):
                store.append(_make_receipt("r2"))


# ---------------------------------------------------------------------------
# R1/R2 / regression 3: malformed JSON and checkpoint mismatch fail closed
# without an uncaught decoder exception
# ---------------------------------------------------------------------------


class MalformedStateFailsClosedWithoutUncaughtExceptionTest(unittest.TestCase):
    def test_malformed_log_line_returns_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write("{not valid json at all\n")

            valid, _count = store.verify_chain()  # must not raise
            self.assertFalse(valid)

    def test_malformed_checkpoint_returns_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            checkpoint_path = store.path.with_name(store.path.name + ".checkpoint")
            checkpoint_path.write_text("not json{{{")

            valid, _count = store.verify_chain()  # must not raise
            self.assertFalse(valid)

    def test_valid_json_array_in_log_returns_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            store.path.write_text("[]\n")

            valid, _count = store.verify_chain()
            self.assertFalse(valid)

    def test_valid_json_array_checkpoint_returns_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            checkpoint_path = store.path.with_name(store.path.name + ".checkpoint")
            checkpoint_path.write_text("[]")

            valid, _count = store.verify_chain()
            self.assertFalse(valid)

    def test_removed_complete_log_and_checkpoint_fail_while_key_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            store.path.write_text("")
            store._checkpoint_path.unlink()  # noqa: SLF001 - deliberate integrity fixture

            valid, count = store.verify_chain()
            self.assertFalse(valid)
            self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# regression 4: concurrent append and retrieval never observe a partial
# or unauthenticated record
# ---------------------------------------------------------------------------


class ConcurrentAppendAndRetrievalNeverPartialTest(unittest.TestCase):
    def test_interleaved_verification_during_concurrent_append_never_sees_invalid_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            n = 30
            observed_invalid = []
            stop_flag = threading.Event()

            def appender():
                for i in range(n):
                    store.append(_make_receipt(f"req-{i}"))

            def watcher():
                while not stop_flag.is_set():
                    valid, count = store.verify_chain()
                    if not valid:
                        observed_invalid.append(count)
                    time.sleep(0.001)

            t_watch = threading.Thread(target=watcher)
            t_watch.start()
            appender()
            stop_flag.set()
            t_watch.join(timeout=10)

            self.assertEqual(observed_invalid, [], "a concurrent reader must never observe a torn append")
            valid, count = store.verify_chain()
            self.assertTrue(valid)
            self.assertEqual(count, n)


# ---------------------------------------------------------------------------
# R4 / regression 5: backend failure plus receipt-write failure
# ---------------------------------------------------------------------------


class BackendFailurePlusReceiptWriteFailureReturnsNoReceiptIdTest(unittest.TestCase):
    def test_no_receipt_id_and_an_incomplete_terminal_state_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                stack.supervisor.stop(stack.alias)  # backend is now unreachable
                with patch.object(
                    stack.context.receipts, "append", side_effect=ReceiptPersistenceError("disk full (fixture)")
                ):
                    status, body, _headers = _post(
                        stack.server.base_url, "/v1/chat/completions",
                        {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                    )
                self.assertEqual(status, 500)
                self.assertEqual(body["error"]["type"], "receipt_persistence_failed")
                self.assertNotIn("receipt_id", body["error"])
                self.assertEqual(body["error"]["terminal_status"], "incomplete")
                self.assertEqual(body["error"]["backend_error_type"], "backend_unavailable")
            finally:
                stack.stop()


# ---------------------------------------------------------------------------
# R3 / regressions 6, 7, 8, 9: complete typed request parsing
# ---------------------------------------------------------------------------


class NonObjectChatBodyReturnsTyped400Test(unittest.TestCase):
    def test_json_array_body_returns_typed_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                status, body, _headers = _post(stack.server.base_url, "/v1/chat/completions", raw_body=b"[]")
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["type"], "invalid_request")
            finally:
                stack.stop()


class InvalidUtf8ChatBodyReturnsTyped400Test(unittest.TestCase):
    def test_invalid_utf8_body_returns_typed_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                raw = b'{"model": "fixture-alias", "messages": [{"role": "user", "content": "\xff"}]}'
                status, body, _headers = _post(stack.server.base_url, "/v1/chat/completions", raw_body=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["type"], "invalid_request")
            finally:
                stack.stop()


class NegativeContentLengthReturnsTyped400WithoutTimeoutTest(unittest.TestCase):
    def test_negative_content_length_returns_typed_400_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                host, port = stack.server._httpd.server_address[:2]  # noqa: SLF001
                sock = socket.create_connection((host, port), timeout=5)
                try:
                    request = (
                        f"POST /v1/chat/completions HTTP/1.1\r\n"
                        f"Host: {host}\r\n"
                        f"Content-Length: -1\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode("ascii")
                    started = time.monotonic()
                    sock.sendall(request)
                    response = b""
                    sock.settimeout(5)
                    while True:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response += chunk
                    elapsed = time.monotonic() - started
                finally:
                    sock.close()
                self.assertLess(elapsed, gateway_module.BODY_READ_TIMEOUT_S, "must not wait for the body timeout")
                self.assertIn(b"400", response)
                self.assertIn(b"invalid_request", response)
            finally:
                stack.stop()


class InvalidBackendResponseReturnsTypedBackendErrorTest(unittest.TestCase):
    def test_backend_json_array_returns_typed_backend_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                bogus_port = free_port()
                body_bytes = b"[]"
                response = (
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(body_bytes)).encode() + b"\r\n\r\n" + body_bytes
                )
                thread = _serve_one_raw_response(bogus_port, response)
                backend = stack.context.backends["fixture-alias"]
                stack.context.backends["fixture-alias"] = gateway_module.BackendInfo(
                    engine=backend.engine, engine_version=backend.engine_version,
                    host="127.0.0.1", port=bogus_port, weight_digest=backend.weight_digest,
                    config_digest=backend.config_digest, executable_digest=backend.executable_digest,
                    argv_digest=backend.argv_digest, template_digest=backend.template_digest,
                )
                status, body, _headers = _post(
                    stack.server.base_url, "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                )
                thread.join(timeout=5)
                self.assertEqual(status, 502)
                self.assertEqual(body["error"]["type"], "backend_response_invalid")
            finally:
                stack.stop()

    def test_backend_invalid_utf8_returns_typed_backend_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                bogus_port = free_port()
                body_bytes = b"\xff\xfe not utf8"
                response = (
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(body_bytes)).encode() + b"\r\n\r\n" + body_bytes
                )
                thread = _serve_one_raw_response(bogus_port, response)
                backend = stack.context.backends["fixture-alias"]
                stack.context.backends["fixture-alias"] = gateway_module.BackendInfo(
                    engine=backend.engine, engine_version=backend.engine_version,
                    host="127.0.0.1", port=bogus_port, weight_digest=backend.weight_digest,
                    config_digest=backend.config_digest, executable_digest=backend.executable_digest,
                    argv_digest=backend.argv_digest, template_digest=backend.template_digest,
                )
                status, body, _headers = _post(
                    stack.server.base_url, "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                )
                thread.join(timeout=5)
                self.assertEqual(status, 502)
                self.assertEqual(body["error"]["type"], "backend_response_invalid")
            finally:
                stack.stop()


# ---------------------------------------------------------------------------
# R6 / regression 10: duplicate alias, same supervisor
# ---------------------------------------------------------------------------


class DuplicateAliasSameSupervisorRefusesSecondStartTest(unittest.TestCase):
    def test_second_start_refused_while_first_still_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = write_fixture_weight(tmp_path / "m.gguf")
            supervisor = Supervisor(tmp_path / "sessions")
            port1 = free_port()
            record1 = supervisor.start(
                alias="same-alias",
                argv=[str(FAKE_ENGINE_PATH), "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port1)],
                allowlisted_executable=str(FAKE_ENGINE_PATH), host="127.0.0.1", port=port1, health_path="/health",
            )
            try:
                port2 = free_port()
                with self.assertRaises(AliasAlreadyRunningError):
                    supervisor.start(
                        alias="same-alias",
                        argv=[str(FAKE_ENGINE_PATH), "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port2)],
                        allowlisted_executable=str(FAKE_ENGINE_PATH), host="127.0.0.1", port=port2, health_path="/health",
                    )
                status = supervisor.status("same-alias")
                self.assertEqual(status["state"], "running")
                self.assertEqual(status["pid"], record1.pid)
            finally:
                supervisor.stop("same-alias")


# ---------------------------------------------------------------------------
# R6 / regression 11: duplicate alias, two separate processes racing
# ---------------------------------------------------------------------------


_ALIAS_RACE_WORKER = """
import sys, json
sys.path.insert(0, {src_dir!r})
from pathlib import Path
from plag_in.supervisor import Supervisor

supervisor = Supervisor(Path({sessions_dir!r}))
try:
    record = supervisor.start(
        alias={alias!r},
        argv=[{engine_path!r}, "--model", {model_path!r}, "--host", "127.0.0.1", "--port", str({port!r})],
        allowlisted_executable={engine_path!r},
        host="127.0.0.1",
        port={port!r},
        health_path="/health",
    )
    print(json.dumps({{"outcome": "started", "pid": record.pid}}))
except Exception as exc:
    print(json.dumps({{"outcome": "refused", "error_type": getattr(exc, "error_type", type(exc).__name__)}}))
"""


class DuplicateAliasTwoProcessRaceTest(unittest.TestCase):
    def test_two_processes_racing_to_start_one_alias_leave_one_live_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = write_fixture_weight(tmp_path / "m.gguf")
            sessions_dir = tmp_path / "sessions"
            port = free_port()
            script = _ALIAS_RACE_WORKER.format(
                src_dir=_SRC_DIR, sessions_dir=str(sessions_dir), alias="raced-alias",
                engine_path=str(FAKE_ENGINE_PATH), model_path=str(model_path), port=port,
            )
            procs = [
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
                )
                for _ in range(2)
            ]
            outcomes = []
            for p in procs:
                stdout, _stderr = p.communicate(timeout=30)
                outcomes.append(json.loads(stdout.strip().splitlines()[-1]))

            supervisor = Supervisor(sessions_dir)
            try:
                started = [o for o in outcomes if o["outcome"] == "started"]
                refused = [o for o in outcomes if o["outcome"] == "refused"]
                self.assertEqual(len(started), 1, outcomes)
                self.assertEqual(len(refused), 1, outcomes)
                self.assertEqual(refused[0]["error_type"], "alias_already_running")

                status = supervisor.status("raced-alias")
                self.assertEqual(status["state"], "running")
                self.assertEqual(status["pid"], started[0]["pid"])
            finally:
                supervisor.stop("raced-alias")


# ---------------------------------------------------------------------------
# R7 / regression 12: occupied gateway bind port starts no engine
# ---------------------------------------------------------------------------


class OccupiedGatewayBindPortStartsNoEngineTest(unittest.TestCase):
    def test_gateway_bind_conflict_leaves_no_engine_and_no_session_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "roots"
            model_root.mkdir()
            model_path = write_fixture_weight(model_root / "m.gguf")
            gateway_port = free_port()

            blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind(("127.0.0.1", gateway_port))
            blocker.listen(1)
            try:
                config_path = tmp_path / "config.json"
                config_path.write_text(json.dumps({
                    "model_roots": [str(model_root)],
                    "engines": {"llama_server": {"executable": str(FAKE_ENGINE_PATH)}},
                    "bind": {"host": "127.0.0.1", "port": gateway_port},
                    "security": {"auth_mode": "none"},
                }))
                engine_port = free_port()
                parser = cli_module.build_parser()
                args = parser.parse_args([
                    "serve", "--config", str(config_path), "--alias", "a1", "--model-path", str(model_path),
                    "--engine-port", str(engine_port), "--state-dir", str(tmp_path / "state"),
                    "--confirm-runtime-profile",
                ])
                with patch(
                    "plag_in.cli._model_load_safety",
                    return_value=(
                        {"memory.high": 1, "memory.max": 2, "memory.swap.max": 1},
                        {"available_ram_bytes": 4, "required_available_ram_bytes": 3,
                         "gpu": None, "required_free_vram_mib": None},
                    ),
                ):
                    with self.assertRaises(PortConflictError):
                        cli_module.cmd_serve(args)

                sessions_dir = tmp_path / "state" / "sessions"
                self.assertFalse(sessions_dir.exists() and any(sessions_dir.iterdir()))

                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    probe.bind(("127.0.0.1", engine_port))
                except OSError:
                    self.fail("the engine was started despite the gateway bind conflict")
                finally:
                    probe.close()
            finally:
                blocker.close()


# ---------------------------------------------------------------------------
# R7 / regression 13: forced receipt-store construction failure leaves
# no listener, engine or session record
# ---------------------------------------------------------------------------


class ForcedReceiptStoreConstructionFailureLeavesNothingTest(unittest.TestCase):
    def test_hmac_key_path_collision_leaves_no_listener_engine_or_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "roots"
            model_root.mkdir()
            model_path = write_fixture_weight(model_root / "m.gguf")
            gateway_port = free_port()

            state_dir = tmp_path / "state"
            state_dir.mkdir()
            # Force ReceiptStore construction to fail: its hmac key path is
            # already occupied by a directory, not a file.
            (state_dir / "receipts.jsonl.hmac_key").mkdir(parents=True)

            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "model_roots": [str(model_root)],
                "engines": {"llama_server": {"executable": str(FAKE_ENGINE_PATH)}},
                "bind": {"host": "127.0.0.1", "port": gateway_port},
                "security": {"auth_mode": "none"},
            }))
            engine_port = free_port()
            parser = cli_module.build_parser()
            args = parser.parse_args([
                "serve", "--config", str(config_path), "--alias", "a1", "--model-path", str(model_path),
                "--engine-port", str(engine_port), "--state-dir", str(state_dir),
                "--confirm-runtime-profile",
            ])
            with patch(
                "plag_in.cli._model_load_safety",
                return_value=(
                    {"memory.high": 1, "memory.max": 2, "memory.swap.max": 1},
                    {"available_ram_bytes": 4, "required_available_ram_bytes": 3,
                     "gpu": None, "required_free_vram_mib": None},
                ),
            ):
                # Was `OSError`, which is what the raw `IsADirectoryError` from
                # reading the key by name produced. The key is now opened with
                # O_NOFOLLOW and checked with `fstat`, so a directory at that
                # name is refused as a typed error before any read, per the
                # rule in `errors.py` that a refusal surfaces through these
                # types rather than a generic exception. What this test exists
                # to establish, that the forced failure leaves no listener,
                # engine or session, is unchanged and still asserted below.
                with self.assertRaises(ReceiptPersistenceError) as raised:
                    cli_module.cmd_serve(args)
            self.assertIn("not a regular file", str(raised.exception))

            sessions_dir = state_dir / "sessions"
            self.assertFalse(sessions_dir.exists() and any(sessions_dir.iterdir()))

            for port in (gateway_port, engine_port):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    self.fail(f"port {port} was left bound after the forced construction failure")
                finally:
                    probe.close()


# ---------------------------------------------------------------------------
# R5 / regression 14: bind port 0 resolves to the assigned port
# ---------------------------------------------------------------------------


class BindPortZeroResolvesToAssignedPortTest(unittest.TestCase):
    def test_status_and_receipt_report_the_effective_port_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                actual_port = stack.server._httpd.server_address[1]  # noqa: SLF001
                self.assertNotEqual(actual_port, 0)

                status, doc, _headers = _get(stack.server.base_url, "/plag-in/v1/status")
                self.assertEqual(status, 200)
                self.assertEqual(doc["bind"]["port"], actual_port)

                status, body, headers = _post(
                    stack.server.base_url, "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                )
                self.assertEqual(status, 200)
                request_id = headers.get("X-PLAG-IN-Request-ID")
                status, receipt, _headers = _get(stack.server.base_url, f"/plag-in/v1/receipts/{request_id}")
                self.assertEqual(status, 200)
                self.assertIn(f":{actual_port}", receipt["listen_address"])
                self.assertNotIn(":0", receipt["listen_address"])
            finally:
                stack.stop()


# ---------------------------------------------------------------------------
# R8 / regression 15: bounded connection-work admission before body read
# ---------------------------------------------------------------------------


class ConnectionAdmissionBoundedBeforeBodyReadTest(unittest.TestCase):
    def test_excess_slow_connections_receive_typed_overload_and_are_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            held = []
            try:
                host, port = stack.server._httpd.server_address[:2]  # noqa: SLF001
                limit = gateway_module.MAX_CONCURRENT_CONNECTIONS
                for _ in range(limit):
                    s = socket.create_connection((host, port), timeout=5)
                    held.append(s)
                time.sleep(0.3)  # let the server's handler threads start and block on the request line

                overflow = socket.create_connection((host, port), timeout=5)
                overflow.settimeout(5)
                response = b""
                try:
                    while b"\r\n\r\n" not in response:
                        chunk = overflow.recv(4096)
                        if not chunk:
                            break
                        response += chunk
                finally:
                    overflow.close()
                self.assertIn(b"429", response)
                self.assertIn(b"overloaded", response)
            finally:
                for s in held:
                    s.close()
                stack.stop()


if __name__ == "__main__":
    unittest.main()
