"""The typed 413 must reach the client that provoked it.

A request whose declared length exceeds the limit is refused before the body is
read. The refusal is the security property and it is unchanged here. What was
missing is that the client received it: the server closed while the client was
still writing, so the client's own write failed with a broken pipe and it never
read the response it had been sent. On the tree before this repair the failure
appeared in 4 of 30 immediate repetitions.

These probes drive one running gateway repeatedly, and bound what the repair
itself may cost: the discard reads a fixed-size chunk at a time, stops at a byte
cap, and stops at a deadline.
"""
import json
import socket
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import plag_in.gateway as gateway_module
from tests.support import build_gateway_stack

REPETITIONS = 25


def post_oversized(base_url: str, payload: bytes):
    """Return ('typed_413', code) or ('transport', reason) for one request."""
    request = urllib.request.Request(  # noqa: S310 - loopback fixture
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return "unrefused", response.status
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        finally:
            exc.close()
        return "typed_413", (exc.code, body["error"]["type"])
    except urllib.error.URLError as exc:
        return "transport", repr(exc.reason)


class OversizedBodyAnswerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = build_gateway_stack(Path(self.tmp.name), alias="fixture-alias")
        self.addCleanup(self.stack.stop)
        self.oversized = b"x" * (gateway_module.MAX_REQUEST_BODY_BYTES + 1)

    def test_every_repetition_receives_the_typed_refusal(self):
        """The regression: one lost answer in the run is a failure."""
        outcomes = [post_oversized(self.stack.server.base_url, self.oversized) for _ in range(REPETITIONS)]

        lost = [outcome for outcome in outcomes if outcome[0] != "typed_413"]
        self.assertEqual(lost, [], f"{len(lost)} of {REPETITIONS} clients did not receive the refusal")
        for _, detail in outcomes:
            self.assertEqual(detail, (413, "payload_too_large"))

    def test_no_oversized_body_reaches_the_store(self):
        """The property the refusal exists for, still true while the body is discarded."""
        before = (
            self.stack.receipts_path.read_text().count("\n")
            if self.stack.receipts_path.exists()
            else 0
        )
        # The transport outcome is deliberately not asserted here. Whether the
        # client received its answer is the regression above; this probe is
        # about the store, and it must hold on a tree where the answer is
        # sometimes lost, so it stays a guard rather than a second detector of
        # the same defect.
        for _ in range(5):
            post_oversized(self.stack.server.base_url, self.oversized)
        after = (
            self.stack.receipts_path.read_text().count("\n")
            if self.stack.receipts_path.exists()
            else 0
        )
        self.assertEqual(before, after, "an oversized body must never reach the backend")

    def test_the_discard_reads_in_bounded_chunks_and_stops_at_the_cap(self):
        """A declared length beyond the cap is not honoured, and nothing accumulates."""
        reads = []
        declared = gateway_module.MAX_DISCARD_BYTES * 4

        class CountingReader:
            """An endless sender: every read is satisfied in full."""

            def read(self, size=-1):
                reads.append(size)
                return b"x" * size if size and size > 0 else b""

        # The discard is driven directly, so the bound is measured rather than
        # inferred from a timing-dependent client.
        _StandInHandler(CountingReader())._discard_refused_body(declared)

        self.assertTrue(reads, "the discard must read something")
        self.assertLessEqual(max(reads), gateway_module.DISCARD_CHUNK_BYTES)
        self.assertLessEqual(
            sum(reads),
            gateway_module.MAX_DISCARD_BYTES,
            "a declared length beyond the cap must not be read whole",
        )
        self.assertLess(sum(reads), declared)

    def test_the_discard_stops_at_its_deadline(self):
        """A sender that stalls cannot hold the connection open through the discard."""
        class StallingReader:
            def read(self, size=-1):
                time.sleep(0.05)
                return b"x" * min(size, 1024)

        instance = _StandInHandler(StallingReader())
        started = time.monotonic()
        with patch.object(gateway_module, "DISCARD_TIMEOUT_S", 0.2):
            instance._discard_refused_body(gateway_module.MAX_DISCARD_BYTES)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 2.0, f"the discard ran for {elapsed:.2f}s past its deadline")

    def test_an_oversized_declared_length_with_a_short_body_still_answers(self):
        """The client stops early; the discard must not wait for the declared length."""
        host, port = self.stack.server.base_url.rsplit(":", 1)
        port = int(port)
        host = host.rsplit("/", 1)[-1]
        declared = gateway_module.MAX_REQUEST_BODY_BYTES * 64

        connection = socket.create_connection((host, port), timeout=10)
        self.addCleanup(connection.close)
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: " + host.encode("ascii") + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(declared).encode("ascii") + b"\r\n"
            b"\r\n"
        )
        connection.sendall(request + b"x" * 1024)
        connection.settimeout(10)

        # The headers arriving is not the response arriving. Reading only to the
        # header terminator saw `Content-Length: 126` and none of those 126
        # bytes, which made this probe itself intermittent.
        received = self.read_response(connection, deadline=time.monotonic() + 10)

        head, _, body = received.partition(b"\r\n\r\n")
        self.assertIn(b"413", head.split(b"\r\n", 1)[0], head[:200])
        self.assertEqual(json.loads(body)["error"]["type"], "payload_too_large")

    @staticmethod
    def read_response(connection, deadline: float) -> bytes:
        """Read one HTTP response whole, headers and the announced body."""
        received = b""
        while time.monotonic() < deadline:
            head, terminator, body = received.partition(b"\r\n\r\n")
            if terminator:
                declared = 0
                for line in head.split(b"\r\n")[1:]:
                    name, _, value = line.partition(b":")
                    if name.strip().lower() == b"content-length":
                        declared = int(value.strip())
                if len(body) >= declared:
                    return received
            chunk = connection.recv(65536)
            if not chunk:
                return received
            received += chunk
        return received


class _StandInHandler:
    """The discard, bound to a reader under test rather than to a live socket.

    `_discard_refused_body` is defined on the handler class built by
    `_make_handler`, so it is reached through one built here. Only `rfile` and
    `connection` are used by it.
    """

    def __init__(self, reader):
        self.rfile = reader
        self.connection = _StandInConnection()

    def __getattr__(self, name):
        handler_class = _handler_class()
        attribute = getattr(handler_class, name)
        return attribute.__get__(self, type(self))


class _StandInConnection:
    def __init__(self):
        self._timeout = None

    def gettimeout(self):
        return self._timeout

    def settimeout(self, value):
        self._timeout = value


def _handler_class():
    """The handler class the gateway builds, without starting a server."""
    return gateway_module._make_handler(_StandInContext())


class _StandInContext:
    def handle(self, *args, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("the discard probes never dispatch a request")


if __name__ == "__main__":
    unittest.main()
