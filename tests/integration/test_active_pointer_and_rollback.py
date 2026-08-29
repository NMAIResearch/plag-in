"""The active-store pointer is confined, and a failed initialisation leaves nothing.

Every case here reproduces a probe from the independent review of work package A
(D1 and D2). The pointer is an ordinary file in the state directory, so an
operator, a stray script or anything running as the operator can edit it; before
this suite existed, a pointer naming a path outside the state directory, a
pointer naming a symlink and a dangling pointer symlink all changed where
receipts were written or read, and an interruption during initialisation left a
file behind at each of its three steps.

The probes run against scratch directories only. No test here touches the live
state directory.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plag_in import receipts as receipts_module
from plag_in.cli import cmd_receipt
from plag_in.errors import ConfigurationError, ReceiptPersistenceError
from plag_in.paths import StateLayout
from plag_in.receipts import (
    CURRENT_STORE_FILENAME,
    INIT_TRANSACTION_LOCK_FILENAME,
    RECEIPT_SCHEMA_VERSION,
    ReceiptStore,
    initialise_current_store,
    rollback_current_store,
)


class ActivePointerContainmentTests(unittest.TestCase):
    """D1: the pointer may name one path, and it may not be a link to it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = StateLayout(self.root / "state").ensure()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.legacy = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_pointer(self, current_store, schema_version: str = RECEIPT_SCHEMA_VERSION) -> None:
        """Write a pointer file directly, as an editor of that file would."""
        self.state.receipt_pointer_file.write_text(
            json.dumps(
                {
                    "current_store": str(current_store),
                    "legacy_store": str(self.state.receipts_file),
                    "schema_version": schema_version,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_a_pointer_naming_a_path_outside_the_state_directory_is_refused(self):
        target = self.outside / CURRENT_STORE_FILENAME
        self.write_pointer(target)

        with self.assertRaises(ConfigurationError) as raised:
            _ = self.state.active_receipts_file
        self.assertIn("directly beneath the state directory", str(raised.exception))
        self.assertFalse(target.exists(), "no store may be created at the escaping path")

    def test_a_pointer_naming_another_filename_in_the_state_directory_is_refused(self):
        self.write_pointer(self.state.base / "receipts.somewhere-else.jsonl")
        with self.assertRaises(ConfigurationError) as raised:
            _ = self.state.active_receipts_file
        self.assertIn("current-schema filename", str(raised.exception))

    def test_a_pointer_whose_store_is_a_symlink_is_refused_and_the_target_does_not_grow(self):
        outside_target = self.outside / "captured.jsonl"
        outside_target.write_bytes(b"")
        link = self.state.base / CURRENT_STORE_FILENAME
        link.symlink_to(outside_target)
        self.write_pointer(link)

        with self.assertRaises(ConfigurationError) as raised:
            _ = self.state.active_receipts_file
        self.assertIn("symbolic link", str(raised.exception))
        self.assertEqual(outside_target.read_bytes(), b"")

    def test_a_symlinked_key_or_checkpoint_beside_the_store_is_refused(self):
        for suffix in (".hmac_key", ".checkpoint", ".lock"):
            with self.subTest(suffix=suffix):
                state = StateLayout(self.root / f"state{suffix}").ensure()
                store = state.base / CURRENT_STORE_FILENAME
                store.write_bytes(b"")
                store.with_name(store.name + suffix).symlink_to(self.outside / "elsewhere")
                with self.assertRaises(ConfigurationError) as raised:
                    state.validate_active_store_path(store)
                self.assertIn("symbolic link", str(raised.exception))

    def test_a_dangling_pointer_symlink_does_not_fall_back_to_the_legacy_store(self):
        self.state.receipt_pointer_file.symlink_to(self.outside / "missing.json")
        with self.assertRaises(ConfigurationError) as raised:
            self.state.read_receipt_pointer()
        self.assertIn("symbolic link", str(raised.exception))

    def test_a_live_pointer_symlink_is_refused_rather_than_followed(self):
        planted = self.outside / "planted.json"
        planted.write_text(
            json.dumps({"current_store": str(self.outside / CURRENT_STORE_FILENAME)}),
            encoding="utf-8",
        )
        self.state.receipt_pointer_file.symlink_to(planted)
        with self.assertRaises(ConfigurationError):
            self.state.read_receipt_pointer()

    def test_a_pointer_naming_a_store_that_does_not_exist_is_refused(self):
        self.write_pointer(self.state.base / CURRENT_STORE_FILENAME)
        with self.assertRaises(ConfigurationError) as raised:
            _ = self.state.active_receipts_file
        self.assertIn("does not exist", str(raised.exception))

    def test_a_pointer_declaring_an_earlier_schema_version_is_refused(self):
        store = self.state.base / CURRENT_STORE_FILENAME
        store.write_bytes(b"")
        self.write_pointer(store, schema_version="4")
        with self.assertRaises(ConfigurationError) as raised:
            _ = self.state.active_receipts_file
        self.assertIn("current schema version", str(raised.exception))

    def test_a_malformed_pointer_is_an_error_not_an_absence(self):
        self.state.receipt_pointer_file.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            self.state.read_receipt_pointer()

    def test_an_absent_pointer_is_the_only_absence(self):
        self.assertIsNone(self.state.read_receipt_pointer())
        self.assertEqual(self.state.active_receipts_file, self.state.receipts_file)

    def test_the_writer_refuses_an_escaping_or_symlinked_destination(self):
        with self.assertRaises(ConfigurationError):
            self.state.write_receipt_pointer(
                self.outside / CURRENT_STORE_FILENAME, RECEIPT_SCHEMA_VERSION
            )
        self.assertFalse(self.state.receipt_pointer_file.exists())

        link = self.state.base / CURRENT_STORE_FILENAME
        link.symlink_to(self.outside / "captured.jsonl")
        with self.assertRaises(ConfigurationError):
            self.state.write_receipt_pointer(link, RECEIPT_SCHEMA_VERSION)
        self.assertFalse(self.state.receipt_pointer_file.exists())

    def test_the_writer_refuses_a_symlinked_pointer_name_and_leaves_the_target_alone(self):
        outside_target = self.outside / "pointer-target.json"
        outside_target.write_text("original", encoding="utf-8")
        self.state.receipt_pointer_file.symlink_to(outside_target)
        store = self.state.base / CURRENT_STORE_FILENAME
        store.write_bytes(b"")

        with self.assertRaises(ConfigurationError):
            self.state.write_receipt_pointer(store, RECEIPT_SCHEMA_VERSION)
        self.assertEqual(outside_target.read_text(encoding="utf-8"), "original")

    def test_the_writer_refuses_a_schema_version_that_is_not_the_current_one(self):
        store = self.state.base / CURRENT_STORE_FILENAME
        store.write_bytes(b"")
        with self.assertRaises(ConfigurationError):
            self.state.write_receipt_pointer(store, "4")

    def test_a_successful_write_leaves_no_temporary_file(self):
        store = self.state.base / CURRENT_STORE_FILENAME
        store.write_bytes(b"")
        self.state.write_receipt_pointer(store, RECEIPT_SCHEMA_VERSION)
        leftovers = [item.name for item in self.state.base.iterdir() if ".tmp" in item.name]
        self.assertEqual(leftovers, [])
        self.assertEqual(self.state.active_receipts_file, store)

    def test_a_planted_temporary_file_cannot_capture_the_write(self):
        """The temporary name is fresh per write and created exclusively."""
        store = self.state.base / CURRENT_STORE_FILENAME
        store.write_bytes(b"")
        planted = self.state.receipt_pointer_file.with_name(
            self.state.receipt_pointer_file.name + ".tmp"
        )
        planted.symlink_to(self.outside / "captured-tmp.json")

        self.state.write_receipt_pointer(store, RECEIPT_SCHEMA_VERSION)
        self.assertFalse((self.outside / "captured-tmp.json").exists())
        self.assertEqual(self.state.active_receipts_file, store)


class InitialisationRollbackTests(unittest.TestCase):
    """D2: nothing survives an interruption or a failed activation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = StateLayout(self.root / "state").ensure()
        self.legacy = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        self.destination = self.state.base / CURRENT_STORE_FILENAME
        self.before = self.directory_listing()

    def tearDown(self):
        self.tmp.cleanup()

    def directory_listing(self) -> set:
        return {item.name for item in self.state.base.iterdir()}

    def assert_nothing_new_remains(self):
        """No new store, key, checkpoint, pointer, temporary file or staging directory.

        Two lock files are permitted to appear and to stay. The initialisation
        transaction lock is persistent by design, and the store's own lock is
        never removed by a rollback, because another process may hold it open
        and its identity was never recorded (pass 2, D6). Both are zero-byte
        mutexes carrying no state.
        """
        permitted_locks = {
            INIT_TRANSACTION_LOCK_FILENAME,
            CURRENT_STORE_FILENAME + ".lock",
        }
        appeared = self.directory_listing() - self.before
        self.assertEqual(
            appeared - permitted_locks,
            set(),
            f"unexpected leftovers: {sorted(appeared - permitted_locks)}",
        )
        for name in appeared:
            self.assertEqual(
                (self.state.base / name).stat().st_size, 0, f"{name} is not an empty lock file"
            )

    def test_an_interruption_after_each_exclusive_creation_leaves_nothing(self):
        real = receipts_module._create_exclusive  # noqa: SLF001

        for stop_after in (1, 2, 3):
            with self.subTest(interrupted_after=stop_after):
                calls: list[Path] = []

                def interrupting_create(path, data, _stop=stop_after, _calls=calls):
                    _calls.append(Path(path))
                    real(path, data)
                    if len(_calls) == _stop:
                        # The file now exists on disk and the call has not
                        # returned to its caller: the exact window that left one
                        # file behind at every step.
                        raise KeyboardInterrupt("interrupted after creation")

                with patch.object(receipts_module, "_create_exclusive", interrupting_create):
                    with self.assertRaises(KeyboardInterrupt):
                        initialise_current_store(self.destination)

                self.assert_nothing_new_remains()

    def test_a_failure_inside_a_creation_leaves_nothing(self):
        real = receipts_module._create_exclusive  # noqa: SLF001
        for stop_after in (1, 2, 3):
            with self.subTest(failed_on=stop_after):
                calls: list[Path] = []

                def failing_create(path, data, _stop=stop_after, _calls=calls):
                    _calls.append(Path(path))
                    if len(_calls) == _stop:
                        raise OSError("simulated write failure")
                    real(path, data)

                with patch.object(receipts_module, "_create_exclusive", failing_create):
                    with self.assertRaises(OSError):
                        initialise_current_store(self.destination)

                self.assert_nothing_new_remains()

    def test_a_verification_failure_after_creation_leaves_nothing(self):
        with patch.object(ReceiptStore, "verify_chain", return_value=(False, 0)):
            with self.assertRaises(ReceiptPersistenceError):
                initialise_current_store(self.destination)
        self.assert_nothing_new_remains()

    def test_an_activation_failure_removes_the_whole_unactivated_store(self):
        args = argparse.Namespace(
            state_dir=str(self.state.base),
            initialise_current_store=True,
            store_path=None,
        )
        with patch.object(
            StateLayout, "write_receipt_pointer", side_effect=OSError("simulated pointer failure")
        ):
            with self.assertRaises(OSError):
                cmd_receipt(args)

        self.assert_nothing_new_remains()
        self.assertFalse(self.state.receipt_pointer_file.exists())
        self.assertEqual(self.state.active_receipts_file, self.state.receipts_file)

    def test_an_interruption_never_removes_a_file_another_process_created(self):
        """The ownership race the repair review rejected the first repair for.

        Initialiser A plans a destination, initialiser B creates that
        destination, A is interrupted before it reaches that name, and A's
        cleanup must leave B's file alone. Removal is decided by file identity,
        so a name A never linked is never A's to remove.
        """
        competing_bytes = b"created by another initialiser\n"

        def interrupting_link(staged, final, _first=[]):  # noqa: B006
            if not _first:
                _first.append(True)
                Path(final).write_bytes(competing_bytes)
                raise KeyboardInterrupt("interrupted before this name was linked")
            raise AssertionError("the first link should have interrupted")

        with patch.object(receipts_module, "_link_into_place", interrupting_link):
            with self.assertRaises(KeyboardInterrupt):
                initialise_current_store(self.destination)

        unowned = self.state.base / (CURRENT_STORE_FILENAME + ".hmac_key")
        self.assertTrue(unowned.exists(), "the other initialiser's file must survive")
        self.assertEqual(unowned.read_bytes(), competing_bytes)

        # Nothing of this call's own survives beside it.
        self.assertFalse(self.destination.exists())
        self.assertFalse(
            self.destination.with_name(self.destination.name + ".checkpoint").exists()
        )
        staging = [item.name for item in self.state.base.iterdir() if item.name.startswith(".plag-in-init-")]
        self.assertEqual(staging, [])

    def test_a_destination_taken_mid_transaction_is_refused_and_left_intact(self):
        competing_bytes = b"not this call's file\n"
        real_link = receipts_module._link_into_place  # noqa: SLF001

        def racing_link(staged, final, _first=[]):  # noqa: B006
            if not _first:
                _first.append(True)
                Path(final).write_bytes(competing_bytes)
            return real_link(staged, final)

        with patch.object(receipts_module, "_link_into_place", racing_link):
            with self.assertRaises(ReceiptPersistenceError) as raised:
                initialise_current_store(self.destination)

        self.assertIn("refusing to initialise over an existing", str(raised.exception))
        taken = self.state.base / (CURRENT_STORE_FILENAME + ".hmac_key")
        self.assertEqual(taken.read_bytes(), competing_bytes)
        self.assertFalse(self.destination.exists())

    def test_rollback_leaves_a_destination_that_is_no_longer_its_file(self):
        created = initialise_current_store(self.destination)
        self.destination.unlink()
        self.destination.write_bytes(b"")  # same name, different file

        removed = rollback_current_store(created)

        self.assertTrue(self.destination.exists(), "a replaced destination is not ours")
        self.assertNotIn(str(self.destination), removed)
        self.assertFalse(Path(created["hmac_key_path"]).exists())

    def test_rollback_refuses_a_record_without_file_identities(self):
        created = initialise_current_store(self.destination)
        without = {k: v for k, v in created.items() if k != "created_identities"}
        with self.assertRaises(ReceiptPersistenceError) as raised:
            rollback_current_store(without)
        self.assertIn("identities that prove ownership", str(raised.exception))
        self.assertTrue(self.destination.exists())

    def test_the_command_output_carries_no_file_identities(self):
        args = argparse.Namespace(
            state_dir=str(self.state.base),
            initialise_current_store=True,
            store_path=None,
        )
        payload = cmd_receipt(args)
        self.assertNotIn("created_identities", payload["initialised"])

    def test_rollback_refuses_to_remove_a_store_that_holds_bytes(self):
        created = initialise_current_store(self.destination)
        self.destination.write_bytes(b"{}\n")
        with self.assertRaises(ReceiptPersistenceError) as raised:
            rollback_current_store(created)
        self.assertIn("already holds bytes", str(raised.exception))
        self.assertTrue(self.destination.exists())

    def test_rollback_removes_the_store_key_and_checkpoint_and_keeps_the_lock(self):
        created = initialise_current_store(self.destination)
        ReceiptStore(self.destination, hmac_key_path=Path(created["hmac_key_path"])).verify_chain()
        lock = self.destination.with_name(self.destination.name + ".lock")
        self.assertTrue(lock.exists(), "the lock file is created by a verification pass")

        removed = rollback_current_store(created)

        self.assertEqual(len(removed), 3)
        self.assertNotIn(str(lock), removed)
        self.assertTrue(lock.exists(), "a lock another process may hold is never removed")
        self.assert_nothing_new_remains()

    def test_an_unowned_lock_survives_an_interrupted_initialisation(self):
        """D6: the lock is another process's, and this call never records its identity."""
        lock = self.destination.with_name(self.destination.name + ".lock")
        lock_bytes = b""

        def interrupting_verify(self_store):
            lock.write_bytes(lock_bytes)  # another process creates the lock
            raise KeyboardInterrupt("interrupted during verification")

        with patch.object(ReceiptStore, "verify_chain", interrupting_verify):
            with self.assertRaises(KeyboardInterrupt):
                initialise_current_store(self.destination)

        self.assertTrue(lock.exists(), "the other process's lock must survive")
        self.assertEqual(lock.read_bytes(), lock_bytes)
        self.assertFalse(self.destination.exists())
        self.assertFalse(Path(str(self.destination) + ".hmac_key").exists())
        self.assertFalse(Path(str(self.destination) + ".checkpoint").exists())

    def test_initialisation_and_rollback_serialise_on_the_transaction_lock(self):
        """The declared boundary: PLAG IN processes take this lock before they act."""
        lock_path = self.state.base / INIT_TRANSACTION_LOCK_FILENAME
        holder = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            command = [
                sys.executable, "-m", "plag_in", "receipt",
                "--initialise-current-store", "--state-dir", str(self.state.base),
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
            )
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.communicate(timeout=3)
                self.assertFalse(
                    self.destination.exists(),
                    "no store may be created while another holder has the transaction lock",
                )
            finally:
                fcntl.flock(holder, fcntl.LOCK_UN)
            stdout, stderr = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertTrue(self.destination.exists())
            self.assertIn("active_pointer", stdout)
        finally:
            os.close(holder)

    def test_the_command_refuses_a_store_path_outside_the_state_directory(self):
        outside = self.root / "outside"
        outside.mkdir()
        args = argparse.Namespace(
            state_dir=str(self.state.base),
            initialise_current_store=True,
            store_path=str(outside / CURRENT_STORE_FILENAME),
        )
        with self.assertRaises(ConfigurationError):
            cmd_receipt(args)

        self.assertFalse((outside / CURRENT_STORE_FILENAME).exists())
        self.assert_nothing_new_remains()

    def test_a_normal_initialisation_still_activates_and_serves(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "plag_in", "receipt",
                "--initialise-current-store", "--state-dir", str(self.state.base),
            ],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["active_pointer"]["current_store"], str(self.destination))
        self.assertEqual(self.state.active_receipts_file, self.destination)
        self.assertEqual(
            self.state.active_receipts_hmac_key_file,
            self.destination.with_name(self.destination.name + ".hmac_key"),
        )


if __name__ == "__main__":
    unittest.main()
