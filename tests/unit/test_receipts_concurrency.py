"""Concurrency regressions from CODEX_REVIEW_MVP_2026-08-25.md F3.

50 threads appending to the same store must produce exactly 50 records and
a chain that verifies (regression 5). Two separate Linux processes writing
the same store concurrently must also preserve one valid chain
(regression 6), which is the part a thread-only lock cannot cover.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from plag_in.receipts import Receipt, ReceiptStore

_SRC_DIR = str(Path(__file__).parent.parent.parent / "src")


def _make_receipt(request_id: str) -> Receipt:
    return Receipt(
        request_id=request_id,
        timestamp="2026-08-24T00:00:00+00:00",
        gateway_version="0.1.0-mvp",
        engine="llama_server",
        engine_version="unknown",
        model_alias="fixture-alias",
        weight_digest="a" * 64,
        template_digest="b" * 64,
        config_digest="c" * 64,
        engine_executable_digest="d" * 64,
        argv_digest="e" * 64,
        locality_level="L1",
        route="local",
        listen_address="127.0.0.1:8080",
        backend_address="127.0.0.1:9000",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        status="completed",
        content_retained=False,
    )


class ConcurrentThreadAppendTest(unittest.TestCase):
    def test_fifty_concurrent_appends_produce_fifty_records_and_a_valid_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            n = 50
            errors = []

            def worker(i):
                try:
                    store.append(_make_receipt(f"req-{i}"))
                except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(errors, [])
            valid, count = store.verify_chain()
            self.assertEqual(count, n)
            self.assertTrue(valid)


_WORKER_SCRIPT = """
import sys
sys.path.insert(0, {src_dir!r})
from pathlib import Path
from plag_in.receipts import Receipt, ReceiptStore

store = ReceiptStore(Path({receipts_path!r}), hmac_key_path=Path({key_path!r}))
for i in range(int({count!r})):
    store.append(Receipt(
        request_id=f"{{sys.argv[1]}}-{{i}}",
        timestamp="2026-08-24T00:00:00+00:00",
        gateway_version="0.1.0-mvp",
        engine="llama_server",
        engine_version="unknown",
        model_alias="fixture-alias",
        weight_digest="a" * 64,
        template_digest="b" * 64,
        config_digest="c" * 64,
        engine_executable_digest="d" * 64,
        argv_digest="e" * 64,
        locality_level="L1",
        route="local",
        listen_address="127.0.0.1:8080",
        backend_address="127.0.0.1:9000",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        status="completed",
        content_retained=False,
    ))
"""


class ConcurrentProcessAppendTest(unittest.TestCase):
    def test_two_processes_writing_the_same_store_preserve_one_valid_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts_path = Path(tmp) / "receipts.jsonl"
            key_path = Path(tmp) / "key"
            count_per_process = 15
            script = _WORKER_SCRIPT.format(
                src_dir=_SRC_DIR, receipts_path=str(receipts_path), key_path=str(key_path), count=count_per_process
            )

            procs = [
                subprocess.Popen([sys.executable, "-c", script, label])  # noqa: S603
                for label in ("proc-a", "proc-b")
            ]
            for p in procs:
                self.assertEqual(p.wait(timeout=30), 0)

            store = ReceiptStore(receipts_path, hmac_key_path=key_path)
            valid, count = store.verify_chain()
            self.assertEqual(count, count_per_process * 2)
            self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
