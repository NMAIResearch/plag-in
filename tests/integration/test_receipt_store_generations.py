"""Legacy receipts stay evidence; new receipts go to a separate current store.

Work package A3 and A8. A store written under schema version 2, 3 or 4 is
historical evidence: it is readable and integrity-checkable under its own
rules, it is never rewritten or normalised, and it cannot accept a version 5
append. The operator creates a current-schema store beside it with an explicit
command, and both paths are reported with their roles.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from plag_in.errors import ReceiptMigrationRequiredError, ReceiptPersistenceError
from plag_in.identity import RUNTIME_SOURCE_SCOPE, canonical_line
from plag_in.paths import StateLayout
from plag_in.receipts import (
    CURRENT_STORE_FILENAME,
    INTEGRITY_MODE,
    RECEIPT_SCHEMA_VERSION,
    READABLE_SCHEMA_VERSIONS,
    Receipt,
    ReceiptStore,
    initialise_current_store,
)

TOOL = Path(__file__).resolve().parents[2] / "tools" / "verify_inference_receipts.py"


def make_receipt(request_id: str, **overrides) -> Receipt:
    base = dict(
        request_id=request_id,
        timestamp="2026-08-26T00:00:00+00:00",
        gateway_version="0.1.0a2",
        engine="llama_server",
        engine_version="unknown",
        model_alias="fixture-alias",
        weight_digest="a" * 64,
        template_digest="b" * 64,
        config_digest="c" * 64,
        locality_level="L1",
        route="local",
        listen_address="127.0.0.1:8080",
        backend_address="127.0.0.1:9000",
        input_tokens=7,
        output_tokens=5,
        latency_us=80929,
        status="completed",
        content_retained=False,
        schema_version=RECEIPT_SCHEMA_VERSION,
        runtime_source_digest="d" * 64,
        runtime_source_scope=RUNTIME_SOURCE_SCOPE,
    )
    base.update(overrides)
    return Receipt(**base)


def write_legacy_record(store: ReceiptStore, schema_version: str, request_id: str) -> bytes:
    """Write one authenticated record under an earlier schema generation.

    Signed through the store's own key so the chain and checkpoint verify: the
    record is legacy, not corrupt, which is the state the migration guard has
    to handle without touching it.
    """
    body = asdict(make_receipt(request_id, schema_version=schema_version))
    body["integrity_mode"] = INTEGRITY_MODE
    record = dict(body)
    record["prev_hmac"] = "0" * 64
    record["hmac"] = store._record_hmac(record["prev_hmac"], record)  # noqa: SLF001
    store.path.write_bytes(canonical_line(record))
    store._write_checkpoint(1, record["hmac"])  # noqa: SLF001
    return store.path.read_bytes()


class LegacyStoreRefusesCurrentAppendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl",
            hmac_key_path=self.tmp_path / "receipts.jsonl.hmac_key",
        )
        self.checkpoint = self.store.path.with_name(self.store.path.name + ".checkpoint")

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_pre_v5_generation_refuses_a_current_append(self):
        for legacy_version in ("2", "3", "4"):
            with self.subTest(schema_version=legacy_version):
                store = ReceiptStore(
                    self.tmp_path / f"legacy-{legacy_version}.jsonl",
                    hmac_key_path=self.tmp_path / f"legacy-{legacy_version}.key",
                )
                if legacy_version == "2":
                    body = asdict(make_receipt("req-legacy", schema_version="2"))
                    body["runtime_source_digest"] = None
                    body["runtime_source_scope"] = None
                    body["integrity_mode"] = INTEGRITY_MODE
                    record = dict(body)
                    record["prev_hmac"] = "0" * 64
                    record["hmac"] = store._record_hmac(record["prev_hmac"], record)  # noqa: SLF001
                    store.path.write_bytes(canonical_line(record))
                    store._write_checkpoint(1, record["hmac"])  # noqa: SLF001
                else:
                    write_legacy_record(store, legacy_version, "req-legacy")

                before_store = store.path.read_bytes()
                before_checkpoint = store.path.with_name(
                    store.path.name + ".checkpoint"
                ).read_bytes()

                with self.assertRaises(ReceiptMigrationRequiredError) as raised:
                    store.append(make_receipt("req-new"))

                self.assertEqual(raised.exception.fields["legacy_store_path"], str(store.path))
                self.assertEqual(raised.exception.fields["legacy_schema_version"], legacy_version)
                self.assertIn(
                    "plag-in receipt --initialise-current-store",
                    raised.exception.fields["operator_action"],
                )
                self.assertEqual(store.path.read_bytes(), before_store)
                self.assertEqual(
                    store.path.with_name(store.path.name + ".checkpoint").read_bytes(),
                    before_checkpoint,
                )

    def test_a_legacy_record_remains_readable_and_integrity_checkable(self):
        write_legacy_record(self.store, "3", "req-legacy")
        integrity_valid, count = self.store.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 1)
        self.assertEqual(self.store.get("req-legacy")["schema_version"], "3")

    def test_mixed_generations_cannot_be_produced_by_any_append(self):
        """A3: one store holds one field contract."""
        write_legacy_record(self.store, "4", "req-legacy")
        before = self.store.path.read_bytes()
        for attempt in range(3):
            with self.assertRaises(ReceiptMigrationRequiredError):
                self.store.append(make_receipt(f"req-attempt-{attempt}"))
        self.assertEqual(self.store.path.read_bytes(), before)

        versions = {
            json.loads(line)["schema_version"]
            for line in self.store.path.read_bytes().splitlines()
        }
        self.assertEqual(versions, {"4"})

    def test_a_current_store_accepts_a_current_append(self):
        self.store.append(make_receipt("req-0"))
        self.store.append(make_receipt("req-1"))
        integrity_valid, count = self.store.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 2)

    def test_readable_generations_are_declared(self):
        self.assertEqual(READABLE_SCHEMA_VERSIONS, ("2", "3", "4", "5"))
        self.assertEqual(RECEIPT_SCHEMA_VERSION, "5")


class InitialiseCurrentStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.legacy = ReceiptStore(
            self.tmp_path / "receipts.jsonl",
            hmac_key_path=self.tmp_path / "receipts.jsonl.hmac_key",
        )
        self.legacy_bytes = write_legacy_record(self.legacy, "3", "req-legacy")
        self.legacy_checkpoint = self.legacy.path.with_name(
            self.legacy.path.name + ".checkpoint"
        )
        self.legacy_checkpoint_bytes = self.legacy_checkpoint.read_bytes()
        self.destination = self.tmp_path / CURRENT_STORE_FILENAME

    def tearDown(self):
        self.tmp.cleanup()

    def assert_legacy_untouched(self):
        self.assertEqual(self.legacy.path.read_bytes(), self.legacy_bytes)
        self.assertEqual(self.legacy_checkpoint.read_bytes(), self.legacy_checkpoint_bytes)

    def test_a_fresh_store_accepts_verifies_and_anchors_one_record(self):
        created = initialise_current_store(self.destination)
        self.assertEqual(created["schema_version"], "5")
        self.assertEqual(created["record_count"], 0)
        self.assert_legacy_untouched()

        store = ReceiptStore(self.destination, hmac_key_path=Path(created["hmac_key_path"]))
        store.append(make_receipt("req-first"))
        integrity_valid, count = store.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 1)

        result = subprocess.run(
            [
                sys.executable, str(TOOL), str(self.destination),
                "--hmac-key", created["hmac_key_path"],
                "--checkpoint", created["checkpoint_path"],
            ],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("records_read=1", result.stdout)
        self.assertIn("conformance_assessed=core+anchor", result.stdout)
        self.assert_legacy_untouched()

    def test_the_new_store_has_its_own_key(self):
        created = initialise_current_store(self.destination)
        self.assertNotEqual(
            Path(created["hmac_key_path"]).read_bytes(),
            self.legacy.hmac_key_path.read_bytes(),
        )

    def test_an_existing_destination_is_refused(self):
        self.destination.write_bytes(b"")
        with self.assertRaises(ReceiptPersistenceError) as raised:
            initialise_current_store(self.destination)
        self.assertIn("refusing to initialise over an existing store", str(raised.exception))
        self.assert_legacy_untouched()

    def test_an_existing_key_or_checkpoint_is_refused(self):
        for suffix, role in ((".hmac_key", "HMAC key"), (".checkpoint", "checkpoint")):
            with self.subTest(role=role):
                destination = self.tmp_path / f"receipts{suffix}-case.jsonl"
                destination.with_name(destination.name + suffix).write_bytes(b"x")
                with self.assertRaises(ReceiptPersistenceError) as raised:
                    initialise_current_store(destination)
                self.assertIn(f"refusing to initialise over an existing {role}", str(raised.exception))
                self.assertFalse(destination.exists())

    def test_the_legacy_store_is_refused_as_a_destination(self):
        with self.assertRaises(ReceiptPersistenceError):
            initialise_current_store(self.legacy.path)
        self.assert_legacy_untouched()

    def test_a_failure_before_activation_removes_every_file_it_created(self):
        calls: list[Path] = []
        real = None

        def failing_create(path, data):
            calls.append(Path(path))
            if len(calls) == 3:  # the checkpoint, written last
                raise OSError("simulated failure writing the checkpoint")
            return real(path, data)

        from plag_in import receipts as receipts_module

        real = receipts_module._create_exclusive  # noqa: SLF001
        with patch.object(receipts_module, "_create_exclusive", failing_create):
            with self.assertRaises(OSError):
                initialise_current_store(self.destination)

        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_name(self.destination.name + ".hmac_key").exists())
        self.assertFalse(self.destination.with_name(self.destination.name + ".checkpoint").exists())
        self.assert_legacy_untouched()

    def test_a_second_initialisation_of_the_same_path_is_refused(self):
        initialise_current_store(self.destination)
        first = self.destination.read_bytes()
        with self.assertRaises(ReceiptPersistenceError):
            initialise_current_store(self.destination)
        self.assertEqual(self.destination.read_bytes(), first)


class StoreRolesReportedTests(unittest.TestCase):
    """A3: status reports both stores and what each is for."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = StateLayout(Path(self.tmp.name)).ensure()
        self.legacy = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        write_legacy_record(self.legacy, "3", "req-legacy")

    def tearDown(self):
        self.tmp.cleanup()

    def run_receipt(self, *flags) -> dict:
        result = subprocess.run(
            [
                sys.executable, "-m", "plag_in", "receipt", *flags,
                "--state-dir", str(self.state.base),
            ],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_before_initialisation_only_the_original_store_is_reported(self):
        roles = self.state.receipt_store_roles()
        self.assertIsNone(roles["current_store"])
        self.assertEqual(roles["active_store"], str(self.state.receipts_file))
        self.assertEqual(self.state.active_receipts_file, self.state.receipts_file)

    def test_after_initialisation_both_paths_and_roles_are_reported(self):
        payload = self.run_receipt("--initialise-current-store")
        self.assertFalse(payload["legacy_store_modified"])

        roles = payload["receipt_stores"]
        self.assertEqual(roles["legacy_store"]["path"], str(self.state.receipts_file))
        self.assertIn("historical evidence", roles["legacy_store"]["role"])
        self.assertEqual(
            roles["current_store"]["path"], str(self.state.base / CURRENT_STORE_FILENAME)
        )
        self.assertEqual(roles["current_store"]["schema_version"], "5")
        self.assertIn("new receipts are appended here", roles["current_store"]["role"])
        self.assertEqual(roles["active_store"], roles["current_store"]["path"])

        self.assertEqual(
            self.state.active_receipts_file, self.state.base / CURRENT_STORE_FILENAME
        )

    def test_the_command_reports_both_stores_on_every_check(self):
        self.run_receipt("--initialise-current-store")
        for flags in (("--check-chain",), ("--check-conformance",)):
            with self.subTest(flags=flags):
                payload = self.run_receipt(*flags)
                self.assertIsNotNone(payload["receipt_stores"]["current_store"])
                self.assertIn("legacy_store", payload["receipt_stores"])

    def test_conformance_evaluates_the_current_store_only(self):
        legacy_digest = hashlib.sha256(self.state.receipts_file.read_bytes()).hexdigest()
        self.run_receipt("--initialise-current-store")
        payload = self.run_receipt("--check-conformance")
        self.assertEqual(payload["store_path"], str(self.state.base / CURRENT_STORE_FILENAME))
        # An empty current store is not conforming, and the legacy record is not
        # counted towards it either way.
        self.assertEqual(payload["record_count"], 0)
        self.assertEqual(
            hashlib.sha256(self.state.receipts_file.read_bytes()).hexdigest(), legacy_digest
        )


if __name__ == "__main__":
    unittest.main()
