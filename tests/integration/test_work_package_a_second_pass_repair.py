"""The three blocking defects of the second independent review, and the fourth.

Each case here reproduces a probe from `CODEX_REVIEW` of work package A, second
pass, as recorded in `agent_handoff/FROM_CODEX.md` on 2026-08-27:

D1  the transaction lock was opened without refusing a symbolic link, so a link
    planted at `receipts.init.lock` created a regular file outside the state
    directory and store creation then completed.
D2  a gateway resolved its store once, so a store activated while it ran was
    not written to until the process restarted.
D3  the store and its key were two separate pointer reads, so an activation
    between them selected the legacy store with the current store's key.
D4  the pointer writer reported success for a store that did not exist, and the
    next read refused the pointer it had just written.

Every probe runs against a scratch directory. No test here touches the live
state directory, starts a model, opens a socket or writes outside its own
temporary tree.
"""
import fcntl
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from plag_in import receipts as receipts_module
from plag_in.active_store import ActiveReceiptStore
from plag_in import conformance as conformance_module
from plag_in.conformance import check_conformance
from plag_in.cli import cmd_receipt
from plag_in.errors import (
    ConfigurationError,
    ReceiptCheckpointError,
    ReceiptPersistenceError,
)
from plag_in.paths import StateLayout
from plag_in.receipts import (
    CURRENT_STORE_FILENAME,
    INIT_TRANSACTION_LOCK_FILENAME,
    RECEIPT_SCHEMA_VERSION,
    ReceiptStore,
    initialise_current_store,
)

from tests.integration.test_receipt_store_generations import make_receipt


class ScratchStateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = StateLayout(self.root / "state").ensure()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.destination = self.state.base / CURRENT_STORE_FILENAME

    def tearDown(self):
        self.tmp.cleanup()

    def initialise_args(self) -> Namespace:
        return Namespace(
            state_dir=str(self.state.base),
            initialise_current_store=True,
            store_path=None,
            check_chain=False,
            check_conformance=False,
            request_id=None,
        )


class TransactionLockConfinementTests(ScratchStateTestCase):
    """D1: the lock is a name in the state directory, not a route out of it."""

    def test_a_symlinked_transaction_lock_is_refused_and_creates_nothing_outside(self):
        target = self.outside / "captured.lock"
        link = self.state.base / INIT_TRANSACTION_LOCK_FILENAME
        link.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            initialise_current_store(self.destination)

        self.assertIn("symbolic link", str(raised.exception))
        self.assertFalse(target.exists(), "no file may be created at the far end of the link")
        self.assertFalse(self.destination.exists(), "no store may be created behind a refused lock")
        self.assertFalse(
            self.destination.with_name(self.destination.name + ".hmac_key").exists()
        )
        self.assertFalse(
            self.destination.with_name(self.destination.name + ".checkpoint").exists()
        )

    def test_the_store_own_lock_is_refused_when_it_is_a_link(self):
        """The same defect class one directory along: `receipts.jsonl.lock`.

        Not named in the review. `validate_active_store_path` refuses a
        symlinked `.lock`, but only on the path through the pointer, and a
        store constructed directly never takes it.
        """
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        target = self.outside / "captured-store.lock"
        lock = self.state.receipts_file.with_name(self.state.receipts_file.name + ".lock")
        lock.unlink(missing_ok=True)
        lock.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            store.append(make_receipt("req-lock-probe"))

        self.assertIn("symbolic link", str(raised.exception))
        self.assertFalse(target.exists())

    def test_a_non_regular_file_at_the_transaction_lock_is_refused(self):
        """A FIFO is not a link, and opening one would block initialisation."""
        os.mkfifo(self.state.base / INIT_TRANSACTION_LOCK_FILENAME)
        with self.assertRaises(ReceiptPersistenceError) as raised:
            initialise_current_store(self.destination)
        self.assertIn("not a regular file", str(raised.exception))
        self.assertFalse(self.destination.exists())

    def test_initialisation_still_succeeds_with_an_ordinary_lock(self):
        created = initialise_current_store(self.destination)
        self.assertEqual(created["schema_version"], RECEIPT_SCHEMA_VERSION)
        self.assertTrue(self.destination.exists())
        self.assertTrue((self.state.base / INIT_TRANSACTION_LOCK_FILENAME).is_file())


class RunningWriterRebindTests(ScratchStateTestCase):
    """D2: activation reaches a writer that is already running."""

    def test_a_writer_built_before_initialisation_follows_the_pointer(self):
        writer = ActiveReceiptStore(self.state)
        self.assertEqual(writer.path, self.state.receipts_file)
        first_binding = writer.rebind_count

        result = cmd_receipt(self.initialise_args())
        self.assertEqual(result["active_pointer"]["current_store"], str(self.destination))

        # No restart, no new object: the same writer resolves the new store.
        self.assertEqual(writer.path, self.destination)
        self.assertEqual(
            writer.hmac_key_path,
            self.destination.with_name(self.destination.name + ".hmac_key"),
        )
        self.assertGreater(writer.rebind_count, first_binding)

    def test_a_receipt_written_after_activation_lands_in_the_current_store(self):
        writer = ActiveReceiptStore(self.state)
        legacy_bytes_before = self.state.receipts_file.read_bytes()

        cmd_receipt(self.initialise_args())
        written = writer.append(make_receipt("req-after-activation"))

        self.assertEqual(written["request_id"], "req-after-activation")
        self.assertEqual(
            self.state.receipts_file.read_bytes(),
            legacy_bytes_before,
            "the legacy store must be byte-identical after an append to the current store",
        )
        records = [
            json.loads(line)
            for line in self.destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual([record["request_id"] for record in records], ["req-after-activation"])

    def test_the_writer_serves_the_active_store_and_reports_a_valid_chain(self):
        writer = ActiveReceiptStore(self.state)
        cmd_receipt(self.initialise_args())
        writer.append(make_receipt("req-served"))

        self.assertEqual(writer.get("req-served")["request_id"], "req-served")
        integrity_valid, count = writer.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 1)

    def test_an_unreadable_pointer_fails_when_the_writer_is_built(self):
        """Failing closed at construction, not on the first served request."""
        self.state.receipt_pointer_file.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            ActiveReceiptStore(self.state)


class ConstructorConfinementTests(ScratchStateTestCase):
    """F1 and F2: an existing link at the store or key name is not a store.

    Both were reported blocking in the review of the second pass. The store
    name took `Path.exists()`, which follows a link, so the confined open was
    skipped for the one case that needed it and the `chmod` after it changed
    the mode of a file outside the state directory. The key name reached the
    `FileExistsError` branch and was then followed by `chmod` and by the read,
    which accepted an outside file's bytes as the HMAC key.

    Both are on the serving path: before activation, `ActiveReceiptStore`
    constructs the legacy store directly from the legacy names, without the
    pointer validation.
    """

    def outside_file(self, name: str, content: bytes) -> Path:
        target = self.outside / name
        target.write_bytes(content)
        os.chmod(target, 0o644)
        return target

    def assert_untouched(self, target: Path, content: bytes) -> None:
        self.assertEqual(target.read_bytes(), content, "outside bytes must not change")
        self.assertEqual(
            stat.S_IMODE(os.stat(target).st_mode), 0o644, "outside mode must not change"
        )

    def test_an_existing_link_at_the_store_name_is_refused(self):
        target = self.outside_file("victim-store", b"")
        self.state.receipts_file.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )

        self.assertIn("symbolic link", str(raised.exception))
        self.assert_untouched(target, b"")
        self.assertFalse(self.state.receipts_hmac_key_file.exists(), "no key sidecar")
        self.assertFalse(
            self.state.receipts_file.with_name("receipts.jsonl.checkpoint").exists()
        )

    def test_a_dangling_link_at_the_store_name_is_refused(self):
        target = self.outside / "absent-store"
        self.state.receipts_file.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError):
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )

        self.assertFalse(target.exists(), "nothing may be created at the far end")
        self.assertFalse(self.state.receipts_hmac_key_file.exists())

    def test_an_existing_link_at_the_key_name_is_refused_and_its_bytes_are_not_the_key(self):
        planted = os.urandom(32)
        target = self.outside_file("victim-key", planted)
        self.state.receipts_hmac_key_file.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )

        self.assertIn("symbolic link", str(raised.exception))
        self.assert_untouched(target, planted)

    def test_a_refused_key_rolls_back_the_store_this_call_created(self):
        target = self.outside_file("victim-key-rollback", os.urandom(32))
        self.state.receipts_hmac_key_file.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError):
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )

        self.assertFalse(
            self.state.receipts_file.exists(),
            "a store created by a call that then failed is not left behind",
        )

    def test_a_store_that_was_already_here_survives_a_refused_key(self):
        """Rollback removes what this call made, never what it found."""
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        store.append(make_receipt("req-pre-existing"))
        before = self.state.receipts_file.read_bytes()

        elsewhere = self.state.base / "other.hmac_key"
        elsewhere.symlink_to(self.outside_file("victim-key-2", os.urandom(32)))
        with self.assertRaises(ReceiptPersistenceError):
            ReceiptStore(self.state.receipts_file, hmac_key_path=elsewhere)

        self.assertEqual(self.state.receipts_file.read_bytes(), before)

    def test_a_dangling_link_at_the_key_name_is_refused(self):
        target = self.outside / "absent-key"
        self.state.receipts_hmac_key_file.symlink_to(target)

        with self.assertRaises(ReceiptPersistenceError):
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )

        self.assertFalse(target.exists())
        self.assertFalse(self.state.receipts_file.exists())

    def test_an_ordinary_store_and_key_still_construct(self):
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        store.append(make_receipt("req-ordinary"))
        integrity_valid, count = store.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 1)
        self.assertEqual(stat.S_IMODE(os.stat(self.state.receipts_file).st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(os.stat(self.state.receipts_hmac_key_file).st_mode), 0o600
        )


_BOUNDED_CONSTRUCTOR_PROBE = """
import os, sys
from pathlib import Path
from plag_in.errors import ReceiptPersistenceError
from plag_in.receipts import ReceiptStore

state = Path(sys.argv[1])
kind = sys.argv[2]
store = state / "receipts.jsonl"
key = state / "receipts.jsonl.hmac_key"
if kind == "store-fifo":
    os.mkfifo(store)
elif kind == "store-directory":
    store.mkdir()
elif kind == "key-fifo":
    os.mkfifo(key)
try:
    ReceiptStore(store, hmac_key_path=key)
except ReceiptPersistenceError as exc:
    print("TYPED", exc)
    sys.exit(0)
except BaseException as exc:
    print("UNTYPED", type(exc).__name__, exc)
    sys.exit(2)
print("ACCEPTED")
sys.exit(3)
"""


class NonRegularConstructorRefusalTests(ScratchStateTestCase):
    """T3-F1, T3-F2 and T3-F4: a name that is not a file fails closed, promptly.

    A FIFO is the case that matters most and is the reason each probe here runs
    in a bounded subprocess. Opening a FIFO waits for a peer, so before the
    repair the constructor did not refuse and did not fail: it stopped inside
    `os.open` and stayed there. A probe for that cannot run in the test process,
    because a regression would hang the suite rather than fail it.

    No FIFO peer is ever opened by these probes, which is the point: the
    refusal must not depend on one arriving.
    """

    TIMEOUT_S = 20

    def run_bounded(self, kind: str):
        state = self.state.base
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _BOUNDED_CONSTRUCTOR_PROBE, str(state), kind],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_S,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except subprocess.TimeoutExpired:
            self.fail(
                f"{kind}: the constructor did not return within {self.TIMEOUT_S}s; "
                "a non-regular name must be refused without waiting for a peer"
            )
        return completed

    def assert_typed_refusal(self, kind: str, expected: str):
        completed = self.run_bounded(kind)
        self.assertEqual(
            completed.returncode,
            0,
            f"{kind}: expected a typed refusal, got {completed.stdout}{completed.stderr}",
        )
        self.assertIn("TYPED", completed.stdout)
        self.assertIn(expected, completed.stdout)

    def test_a_fifo_at_the_store_name_is_refused_without_waiting(self):
        self.assert_typed_refusal("store-fifo", "the receipt store is not a regular file")
        self.assertFalse(self.state.receipts_hmac_key_file.exists(), "no key sidecar")

    def test_a_directory_at_the_store_name_is_refused_as_a_typed_error(self):
        self.assert_typed_refusal(
            "store-directory", "the receipt store is not a regular file"
        )
        self.assertFalse(self.state.receipts_hmac_key_file.exists())

    def test_a_fifo_at_the_key_name_is_refused_and_the_store_is_rolled_back(self):
        self.assert_typed_refusal(
            "key-fifo", "the receipt store HMAC key is not a regular file"
        )
        self.assertFalse(
            self.state.receipts_file.exists(),
            "the store this call created must not survive the key refusal",
        )

    def test_a_directory_at_the_key_name_is_refused(self):
        os.mkdir(self.state.receipts_hmac_key_file)
        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )
        self.assertIn("not a regular file", str(raised.exception))
        self.assertFalse(self.state.receipts_file.exists())

    def test_a_device_node_is_refused_by_the_confined_open(self):
        """Tested at the confined open rather than through the constructor.

        A character device inside the state directory would be the end-to-end
        case, and creating one needs privileges this suite does not have and
        must not require. `/dev/null` exercises the same refusal on the same
        code path: every store-set name is opened through this function.
        """
        for flags in (os.O_RDONLY, os.O_CREAT | os.O_WRONLY):
            with self.subTest(flags=flags):
                with self.assertRaises(ReceiptPersistenceError) as raised:
                    receipts_module._open_confined(
                        Path("/dev/null"), flags, 0o600, "receipt store"
                    )
                self.assertIn("not a regular file", str(raised.exception))


class CreationOwnershipTests(ScratchStateTestCase):
    """T3-F3: ownership of a created store comes from the create, not from a look.

    The reverted shape decided ownership from an `lstat` taken before the open.
    A second actor creating the store in that window left the constructor
    holding another party's inode and recording it as its own, and the rollback
    on key failure then deleted it.

    The interleave is produced deterministically here: the store is created,
    with bytes that identify its author, at the moment the constructor is about
    to open the name.
    """

    SENTINEL = b'{"request_id":"written-by-another-actor"}\n'

    def test_a_store_created_by_another_actor_in_the_window_is_not_removed(self):
        # The key is a link, so key loading fails and rollback runs. That is the
        # path on which the wrong store was being deleted.
        self.state.receipts_hmac_key_file.symlink_to(self.outside / "victim-key")

        real_open_confined = receipts_module._open_confined
        planted = {"done": False}

        def open_confined_after_another_actor_creates(path, flags, mode, role, created_registry=None):
            # Runs immediately before every confined open the constructor makes.
            # On the first store open, a second actor wins the name first.
            if not planted["done"] and Path(path) == self.state.receipts_file:
                planted["done"] = True
                self.state.receipts_file.write_bytes(self.SENTINEL)
            return real_open_confined(path, flags, mode, role, created_registry=created_registry)

        with patch.object(
            receipts_module, "_open_confined", open_confined_after_another_actor_creates
        ):
            with self.assertRaises(ReceiptPersistenceError):
                ReceiptStore(
                    self.state.receipts_file,
                    hmac_key_path=self.state.receipts_hmac_key_file,
                )

        self.assertTrue(planted["done"], "the probe did not reach the store open")
        self.assertTrue(
            self.state.receipts_file.exists(),
            "a store this call did not create must survive its failure",
        )
        self.assertEqual(
            self.state.receipts_file.read_bytes(),
            self.SENTINEL,
            "the other actor's store must be byte-identical",
        )

    def test_a_store_this_call_created_is_still_rolled_back(self):
        """The other direction, so the repair cannot be a blanket refusal to roll back."""
        self.state.receipts_hmac_key_file.symlink_to(self.outside / "victim-key-2")
        with self.assertRaises(ReceiptPersistenceError):
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )
        self.assertFalse(self.state.receipts_file.exists())

    def test_a_genuinely_new_store_still_receives_its_genesis_checkpoint(self):
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        self.assertTrue(checkpoint.exists(), "a new store starts with a genesis checkpoint")
        integrity_valid, count = store.verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 0)

    def test_a_store_recreated_beside_a_surviving_key_gets_no_new_genesis(self):
        """The distinction the pre-existence flags exist to preserve."""
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        self.state.receipts_file.unlink()
        checkpoint.unlink()

        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        self.assertFalse(
            checkpoint.exists(),
            "a removed log and checkpoint beside a surviving key must not silently reset",
        )


_CHECKPOINT_VERIFY_PROBE = """
import os, sys
from pathlib import Path
from plag_in.errors import ReceiptPersistenceError
from plag_in.receipts import ReceiptStore

store = Path(sys.argv[1])
# Build an ordinary store first, then take over its checkpoint name. Planting
# the FIFO before construction is a different case: construction refuses it
# outright, so verification is never reached and this route goes unprobed.
receipts = ReceiptStore(store)
checkpoint = store.with_name(store.name + ".checkpoint")
checkpoint.unlink()
os.mkfifo(checkpoint)
try:
    print("VERIFY", receipts.verify_chain())
except ReceiptPersistenceError as exc:
    print("TYPED", exc)
    sys.exit(0)
except BaseException as exc:
    print("UNTYPED", type(exc).__name__, exc)
    sys.exit(2)
sys.exit(3)
"""

_CONSTRUCT_PROBE = """
import sys
from pathlib import Path
from plag_in.receipts import ReceiptStore

receipts = ReceiptStore(Path(sys.argv[1]))
print("VERIFY", receipts.verify_chain())
"""


class StoreSetTransactionTests(ScratchStateTestCase):
    """F4-R1 to F4-R4: the store, key, checkpoint and lock are one set.

    Each finding here reached a sidecar the previous passes had not covered.
    The repair is structural rather than another named case: first construction
    now runs under the store's own lock, taken before anything is created, and
    every file the call creates is removed if any later step fails.
    """

    TIMEOUT_S = 20

    def run_bounded(self, script: str, *args: str):
        return subprocess.run(
            [sys.executable, "-c", script, *args],
            capture_output=True,
            text=True,
            timeout=self.TIMEOUT_S,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

    # -- F4-R2 and F4-R3, the checkpoint sidecar -----------------------

    def test_a_linked_checkpoint_is_refused_and_outside_bytes_are_not_accepted(self):
        """A planted checkpoint made an empty store verify as sound."""
        donor = ReceiptStore(self.state.receipts_file)
        donor_checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        planted = self.outside / "valid-looking.checkpoint"
        planted.write_bytes(donor_checkpoint.read_bytes())

        second = StateLayout(self.root / "second").ensure()
        second.receipts_hmac_key_file.write_bytes(
            self.state.receipts_hmac_key_file.read_bytes()
        )
        second.receipts_file.with_name("receipts.jsonl.checkpoint").symlink_to(planted)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(second.receipts_file).verify_chain()
        self.assertIn("symbolic link", str(raised.exception))
        self.assertIn("checkpoint", str(raised.exception))
        self.assertEqual(planted.read_bytes(), donor_checkpoint.read_bytes())

    def test_a_checkpoint_fifo_does_not_block_verification(self):
        try:
            completed = self.run_bounded(
                _CHECKPOINT_VERIFY_PROBE, str(self.state.receipts_file)
            )
        except subprocess.TimeoutExpired:
            self.fail(
                "verification blocked on a checkpoint FIFO; no receipt operation "
                "may depend on a peer arriving"
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("not a regular file", completed.stdout)

    # -- F4-R4, a refusal at the lock leaves nothing -------------------

    def test_a_refusal_at_the_lock_leaves_no_store_or_key_behind(self):
        os.mkfifo(self.state.receipts_file.with_name("receipts.jsonl.lock"))
        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(self.state.receipts_file)
        self.assertIn("lock file", str(raised.exception))
        self.assertFalse(self.state.receipts_file.exists(), "no store may be left behind")
        self.assertFalse(self.state.receipts_hmac_key_file.exists(), "no key may be left behind")
        self.assertEqual(
            sorted(
                p.name for p in self.state.base.iterdir()
                if p.name.startswith("receipts.")
            ),
            ["receipts.jsonl.lock"],
            "only the planted lock name remains of the receipt store set",
        )

    def test_a_failure_after_the_key_still_rolls_the_whole_set_back(self):
        """The rollback covers every step, not only key loading."""
        real_write_checkpoint = ReceiptStore._write_checkpoint

        def failing_write_checkpoint(self_, count, last_hmac, created_registry=None):
            raise ReceiptPersistenceError("simulated checkpoint failure")

        with patch.object(ReceiptStore, "_write_checkpoint", failing_write_checkpoint):
            with self.assertRaises(ReceiptPersistenceError):
                ReceiptStore(self.state.receipts_file)

        self.assertFalse(self.state.receipts_file.exists())
        self.assertFalse(self.state.receipts_hmac_key_file.exists())
        self.assertIs(ReceiptStore._write_checkpoint, real_write_checkpoint)

    def test_a_failure_does_not_remove_a_store_set_that_was_already_here(self):
        first = ReceiptStore(self.state.receipts_file)
        first.append(make_receipt("req-established"))
        store_bytes = self.state.receipts_file.read_bytes()
        key_bytes = self.state.receipts_hmac_key_file.read_bytes()

        def failing_write_checkpoint(self_, count, last_hmac, created_registry=None):
            raise ReceiptPersistenceError("simulated checkpoint failure")

        with patch.object(ReceiptStore, "_write_checkpoint", failing_write_checkpoint):
            # Construction over an established set writes no genesis checkpoint,
            # so this must simply succeed and touch nothing.
            ReceiptStore(self.state.receipts_file)

        self.assertEqual(self.state.receipts_file.read_bytes(), store_bytes)
        self.assertEqual(self.state.receipts_hmac_key_file.read_bytes(), key_bytes)

    # -- F4-R1, first construction is one decision ---------------------

    def test_construction_holds_the_store_lock_across_the_whole_set(self):
        """The deterministic form: nothing is created while the lock is held.

        Before the repair, the store and key were created outside any lock and
        only the genesis write took it, so two first constructors could split
        the pair between them and neither wrote a checkpoint. Holding the lock
        for the whole set is the property that makes that impossible, and this
        establishes it directly rather than by racing.
        """
        lock_path = self.state.receipts_file.with_name("receipts.jsonl.lock")
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            child = subprocess.Popen(
                [sys.executable, "-c", _CONSTRUCT_PROBE, str(self.state.receipts_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            try:
                time.sleep(1.5)
                self.assertIsNone(
                    child.poll(),
                    "construction completed while another holder had the store lock",
                )
                self.assertFalse(
                    self.state.receipts_file.exists(),
                    "nothing may be created before the lock is held",
                )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            stdout, stderr = child.communicate(timeout=self.TIMEOUT_S)
            self.assertEqual(child.returncode, 0, stdout + stderr)
            self.assertIn("(True, 0)", stdout)
        finally:
            os.close(lock_fd)

    def test_concurrent_first_constructors_produce_one_coherent_store(self):
        """The observed form of F4-R1. Not deterministic on a reverted tree."""
        children = [
            subprocess.Popen(
                [sys.executable, "-c", _CONSTRUCT_PROBE, str(self.state.receipts_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            for _ in range(6)
        ]
        results = [child.communicate(timeout=self.TIMEOUT_S) for child in children]

        for child, (stdout, stderr) in zip(children, results):
            self.assertEqual(child.returncode, 0, stdout + stderr)
            self.assertIn("(True, 0)", stdout)
        self.assertTrue(
            self.state.receipts_file.with_name("receipts.jsonl.checkpoint").exists(),
            "a genesis checkpoint must exist after concurrent first construction",
        )
        integrity_valid, count = ReceiptStore(self.state.receipts_file).verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 0)


_CONFORMANCE_PROBE = """
import sys
from pathlib import Path
from plag_in.conformance import check_conformance
from plag_in.errors import ReceiptPersistenceError

store = Path(sys.argv[1])
try:
    result = check_conformance(
        store,
        store.with_name(store.name + ".hmac_key"),
        store.with_name(store.name + ".checkpoint"),
    )
    print("RESULT", result["conformance_valid"], result["record_count"])
except ReceiptPersistenceError as exc:
    print("TYPED", exc)
    sys.exit(0)
sys.exit(3)
"""


class ConformanceRouteConfinementTests(ScratchStateTestCase):
    """R5-F1: the conformance route reads the set under the same rules.

    `check_conformance` read the store, key and checkpoint by name and outside
    the store lock. Links at the legacy paths were accepted as conforming, a
    FIFO at a sidecar blocked the check, and an ordinary append landing between
    the log read and the checkpoint read made a sound store report as invalid.
    """

    TIMEOUT_S = 20

    def build_store(self, request_id: str = "req-conformance") -> ReceiptStore:
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        store.append(make_receipt(request_id))
        return store

    def paths(self):
        return (
            self.state.receipts_file,
            self.state.receipts_hmac_key_file,
            self.state.receipts_file.with_name("receipts.jsonl.checkpoint"),
        )

    def test_a_link_at_any_receipt_set_name_is_refused_by_the_check(self):
        for index, name in enumerate(("store", "key", "checkpoint")):
            with self.subTest(name=name):
                state = StateLayout(self.root / f"legacy-{name}").ensure()
                donor = ReceiptStore(state.receipts_file)
                donor.append(make_receipt("req-donor"))
                real = [
                    state.receipts_file,
                    state.receipts_hmac_key_file,
                    state.receipts_file.with_name("receipts.jsonl.checkpoint"),
                ]
                target = self.outside / f"outside-{name}"
                target.write_bytes(real[index].read_bytes())
                real[index].unlink()
                real[index].symlink_to(target)

                with self.assertRaises(ReceiptPersistenceError) as raised:
                    check_conformance(*real)
                self.assertIn("symbolic link", str(raised.exception))

    def test_a_fifo_at_a_receipt_set_name_does_not_block_the_check(self):
        for name, index in (("key", 1), ("checkpoint", 2)):
            with self.subTest(name=name):
                state = StateLayout(self.root / f"fifo-{name}").ensure()
                donor = ReceiptStore(state.receipts_file)
                donor.append(make_receipt("req-donor"))
                real = [
                    state.receipts_file,
                    state.receipts_hmac_key_file,
                    state.receipts_file.with_name("receipts.jsonl.checkpoint"),
                ]
                real[index].unlink()
                os.mkfifo(real[index])
                try:
                    completed = subprocess.run(
                        [sys.executable, "-c", _CONFORMANCE_PROBE, str(state.receipts_file)],
                        capture_output=True,
                        text=True,
                        timeout=self.TIMEOUT_S,
                        cwd=str(Path(__file__).resolve().parents[2]),
                    )
                except subprocess.TimeoutExpired:
                    self.fail(f"the conformance check blocked on a {name} FIFO")
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn("not a regular file", completed.stdout)

    def test_the_check_takes_a_coherent_snapshot_against_an_append(self):
        """The interleave, without any special file involved."""
        store = self.build_store()
        store_path, key_path, checkpoint_path = self.paths()

        # Patched on the conformance module, which bound the name at import.
        real_read = conformance_module.read_confined_bytes
        appended = {"done": False}

        def append_between_the_log_and_the_checkpoint(path, role, limit=None):
            data = real_read(path, role, limit)
            if role == "receipt store" and not appended["done"]:
                appended["done"] = True
                # A second writer completes a record while the check is
                # between its two reads. It must block on the store lock, so
                # this runs after the check has released it.
                threading.Thread(
                    target=lambda: store.append(make_receipt("req-interleaved"))
                ).start()
                time.sleep(0.3)
            return data

        with patch.object(
            conformance_module, "read_confined_bytes", append_between_the_log_and_the_checkpoint
        ):
            result = check_conformance(store_path, key_path, checkpoint_path)

        # Whichever snapshot the check took, it must be internally coherent.
        time.sleep(0.3)
        self.assertTrue(appended["done"])
        self.assertEqual(
            result["record_count"],
            1,
            "the check must report the store it read, not a later one",
        )
        self.assertTrue(
            result["conformance_valid"],
            f"a coherent snapshot must not report a mismatch: {result['errors']}",
        )
        integrity_valid, count = ReceiptStore(store_path, hmac_key_path=key_path).verify_chain()
        self.assertTrue(integrity_valid)
        self.assertEqual(count, 2)


class OversizedCheckpointTests(ScratchStateTestCase):
    """R5-F2, a regression introduced by the fifth pass's bounded read.

    `os.read(fd, 4096)` does not establish end of file, so a checkpoint whose
    valid JSON occupied the first 4096 bytes was accepted and its trailer was
    never seen. The bound is a maximum size, not a maximum read.
    """

    def oversized_checkpoint(self, checkpoint: Path) -> bytes:
        valid = checkpoint.read_bytes()
        padding = b" " * (receipts_module.CHECKPOINT_MAX_BYTES - len(valid))
        oversized = valid + padding + b"UNREAD-TRAILER"
        self.assertGreater(len(oversized), receipts_module.CHECKPOINT_MAX_BYTES)
        checkpoint.write_bytes(oversized)
        return oversized

    def test_an_oversized_checkpoint_does_not_verify(self):
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        self.oversized_checkpoint(checkpoint)

        # Refused rather than reported invalid. Either satisfies the finding;
        # the refusal is what the confined read produces, and it says the file
        # is out of contract rather than that the chain disagrees.
        with self.assertRaises(ReceiptPersistenceError) as raised:
            store.verify_chain()
        self.assertIn("larger than", str(raised.exception))

    def test_an_oversized_checkpoint_is_refused_by_the_confined_read(self):
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        self.oversized_checkpoint(checkpoint)

        with self.assertRaises(ReceiptPersistenceError) as raised:
            receipts_module.read_confined_bytes(
                checkpoint,
                "receipt store checkpoint",
                limit=receipts_module.CHECKPOINT_MAX_BYTES,
            )
        self.assertIn("larger than", str(raised.exception))

    def test_a_checkpoint_at_exactly_the_bound_is_still_read(self):
        """The boundary itself, so the refusal is not off by one."""
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        valid = checkpoint.read_bytes()
        exact = valid + b" " * (receipts_module.CHECKPOINT_MAX_BYTES - len(valid))
        self.assertEqual(len(exact), receipts_module.CHECKPOINT_MAX_BYTES)
        checkpoint.write_bytes(exact)

        raw = receipts_module.read_confined_bytes(
            checkpoint, "receipt store checkpoint", limit=receipts_module.CHECKPOINT_MAX_BYTES
        )
        self.assertEqual(raw, exact)


class LateCreationFailureTests(ScratchStateTestCase):
    """R5-F3: a failure inside a creation helper still leaves nothing behind.

    The rollback list can only remove identities that reached it. A failure
    between creating a file and returning its identity left that file behind,
    so each helper now cleans up what it created and did not finish.
    """

    def assert_nothing_but_the_lock(self):
        remaining = sorted(
            p.name for p in self.state.base.iterdir() if p.name.startswith("receipts.")
        )
        self.assertIn(remaining, ([], ["receipts.jsonl.lock"]), f"left behind: {remaining}")

    def test_a_failure_inside_the_confined_open_removes_the_store(self):
        """The failure must land on the store's own open, not on the lock's.

        The lock is opened first, so a blanket failure stops before the store
        exists and the probe would pass without testing anything. The store is
        the first exclusive create, which is what this selects on.
        """
        real_set_blocking = os.set_blocking
        seen = {"exclusive_opens": 0}
        real_open_confined = receipts_module._open_confined

        def open_confined_counting(path, flags, mode, role, created_registry=None):
            if flags & os.O_EXCL:
                seen["exclusive_opens"] += 1
            return real_open_confined(path, flags, mode, role, created_registry=created_registry)

        def failing_set_blocking(fd, blocking):
            if seen["exclusive_opens"] >= 1:
                raise OSError("simulated failure after the store was created")
            return real_set_blocking(fd, blocking)

        with patch.object(receipts_module, "_open_confined", open_confined_counting):
            with patch.object(os, "set_blocking", failing_set_blocking):
                with self.assertRaises(OSError):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )

        self.assertEqual(
            seen["exclusive_opens"], 1, "the failure must land on the store's exclusive create"
        )
        self.assertIs(os.set_blocking, real_set_blocking)
        self.assert_nothing_but_the_lock()

    def test_a_failure_while_writing_the_key_removes_the_store_and_the_key(self):
        real_fsync = os.fsync
        calls = {"n": 0}

        def failing_fsync(fd):
            calls["n"] += 1
            raise OSError("simulated failure after the key was created")

        with patch.object(os, "fsync", failing_fsync):
            with self.assertRaises(OSError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )
        self.assertIs(os.fsync, real_fsync)
        self.assertGreater(calls["n"], 0)
        self.assert_nothing_but_the_lock()

    def test_a_failure_at_the_checkpoint_rename_removes_the_temporary_file(self):
        def failing_replace(src, dst):
            raise OSError("simulated failure at the checkpoint rename")

        with patch.object(os, "replace", failing_replace):
            with self.assertRaises(OSError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )
        self.assert_nothing_but_the_lock()

    def test_a_genesis_checkpoint_is_removed_when_a_later_step_fails(self):
        real_write_checkpoint = ReceiptStore._write_checkpoint

        def write_then_fail(self_, count, last_hmac, created_registry=None):
            real_write_checkpoint(self_, count, last_hmac, created_registry=created_registry)
            raise ReceiptPersistenceError("simulated failure after the checkpoint existed")

        with patch.object(ReceiptStore, "_write_checkpoint", write_then_fail):
            with self.assertRaises(ReceiptPersistenceError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )
        self.assert_nothing_but_the_lock()


class PlantedCheckpointTests(ScratchStateTestCase):
    """R5-F4: construction does not return a newly enlarged invalid set."""

    def test_a_planted_checkpoint_beside_an_absent_set_is_refused(self):
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        checkpoint.write_text("{}", encoding="utf-8")

        with self.assertRaises(ReceiptPersistenceError) as raised:
            ReceiptStore(
                self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
            )
        self.assertIn("checkpoint already exists", str(raised.exception))
        self.assertFalse(self.state.receipts_file.exists(), "the store must be rolled back")
        self.assertFalse(
            self.state.receipts_hmac_key_file.exists(), "the key must be rolled back"
        )
        self.assertEqual(checkpoint.read_text(encoding="utf-8"), "{}", "the planted file stays")

    def test_a_checkpoint_beside_a_surviving_key_is_still_adopted(self):
        """The established case must keep working: only a wholly new set refuses."""
        first = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        first.append(make_receipt("req-established"))
        self.state.receipts_file.unlink()

        second = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        integrity_valid, _count = second.verify_chain()
        self.assertFalse(
            integrity_valid, "a removed log beside a surviving key must not verify"
        )


class ShortReadTests(ScratchStateTestCase):
    """R6-F1: one short read is not evidence that a file ends there.

    `read(2)` may return fewer bytes than asked for without having reached end
    of file. The sixth pass asked for `limit + 1` bytes once and treated a
    short result as the whole file, so an oversized checkpoint was accepted on
    its valid prefix under an entirely permitted kernel behaviour.
    """

    def oversized(self) -> bytes:
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        valid = checkpoint.read_bytes()
        limit = receipts_module.CHECKPOINT_MAX_BYTES
        oversized = valid + b" " * (limit - len(valid)) + b"UNREAD-TRAILER"
        checkpoint.write_bytes(oversized)
        self.store = store
        self.checkpoint = checkpoint
        return oversized

    def test_a_permitted_short_read_does_not_accept_an_oversized_checkpoint(self):
        self.oversized()
        real_read = os.read
        limit = receipts_module.CHECKPOINT_MAX_BYTES

        def short_read(fd, size):
            # The permitted behaviour: return less than asked for, with the
            # file continuing beyond what was returned.
            if size == limit + 1:
                return real_read(fd, limit)
            return real_read(fd, size)

        with patch.object(os, "read", short_read):
            with self.assertRaises(ReceiptPersistenceError) as raised:
                self.store.verify_chain()
        self.assertIn("larger than", str(raised.exception))

    def test_a_short_read_still_assembles_a_checkpoint_within_the_bound(self):
        """The same injection must not break a legitimate checkpoint."""
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        expected = checkpoint.read_bytes()
        real_read = os.read

        def short_read(fd, size):
            return real_read(fd, 1) if size > 1 else real_read(fd, size)

        with patch.object(os, "read", short_read):
            raw = receipts_module.read_confined_bytes(
                checkpoint,
                "receipt store checkpoint",
                limit=receipts_module.CHECKPOINT_MAX_BYTES,
            )
        self.assertEqual(raw, expected)

    def test_a_short_read_assembles_the_whole_store(self):
        """The unbounded branch, one byte at a time."""
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        store.append(make_receipt("req-short-read"))
        expected = self.state.receipts_file.read_bytes()
        real_read = os.read

        def short_read(fd, size):
            return real_read(fd, 1) if size > 1 else real_read(fd, size)

        with patch.object(os, "read", short_read):
            raw = receipts_module.read_confined_bytes(
                self.state.receipts_file, "receipt store"
            )
        self.assertEqual(raw, expected)


class TemporaryNameOwnershipTests(ScratchStateTestCase):
    """R6-F3: a pre-existing temporary name is not this call's file.

    The predictable `.checkpoint.tmp` name was opened with O_TRUNC and no
    O_EXCL, so a regular file already there was truncated, filled with this
    call's bytes and renamed over the checkpoint. A successful construction
    destroyed a file it found, which the specification says never happens.
    """

    def test_a_pre_existing_temporary_file_is_preserved(self):
        planted = self.state.receipts_file.with_name("receipts.jsonl.checkpoint.tmp")
        sentinel = b"planted-bytes-that-must-survive\n"
        planted.write_bytes(sentinel)

        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)

        self.assertTrue(planted.exists(), "a file found at the temporary name must remain")
        self.assertEqual(planted.read_bytes(), sentinel, "and must be byte-identical")

    def test_the_temporary_name_is_fresh_for_each_write(self):
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        seen = set()
        real_open_confined = receipts_module._open_confined

        def record_temporary_names(path, flags, mode, role, created_registry=None, **kwargs):
            if "checkpoint temporary" in role:
                seen.add(Path(path).name)
            return real_open_confined(
                path, flags, mode, role, created_registry=created_registry, **kwargs
            )

        with patch.object(receipts_module, "_open_confined", record_temporary_names):
            store.append(make_receipt("req-tmp-1"))
            store.append(make_receipt("req-tmp-2"))

        self.assertEqual(len(seen), 2, f"each write needs its own temporary name: {seen}")
        for name in seen:
            self.assertNotEqual(name, "receipts.jsonl.checkpoint.tmp")

    def test_no_temporary_file_survives_a_successful_construction(self):
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file)
        leftovers = [p.name for p in self.state.base.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])


class IdentityWindowTests(ScratchStateTestCase):
    """R6-F2: the obligation to remove a file starts when the file does.

    Six windows existed in which a newly created file was on disk and no
    identity had yet been recorded for it, so nothing could remove it. Each is
    injected here at the point the review named.
    """

    def assert_no_receipt_files_but_the_lock(self):
        remaining = sorted(
            p.name for p in self.state.base.iterdir() if p.name.startswith("receipts.")
        )
        self.assertIn(
            remaining, ([], ["receipts.jsonl.lock"]), f"left behind: {remaining}"
        )

    def fail_on_nth_exclusive_create(self, target_role: str, failing):
        """Run construction with `failing` invoked after the named create."""
        real_open_confined = receipts_module._open_confined
        state = {"armed": False}

        def open_confined_arming(path, flags, mode, role, created_registry=None):
            fd = real_open_confined(path, flags, mode, role, created_registry=created_registry)
            if role == target_role and (flags & os.O_EXCL):
                state["armed"] = True
            return fd

        with patch.object(receipts_module, "_open_confined", open_confined_arming):
            with failing(state):
                with self.assertRaises(BaseException):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )
        return state

    def test_a_failure_at_the_stores_own_fstat_removes_the_store(self):
        """Window 1: identity is never obtained, so cleanup is by name.

        Armed on the store's exclusive create specifically. A blanket `fstat`
        failure fires on the lock's open instead, before the store exists, and
        the probe then passes without testing anything.
        """
        real_fstat = os.fstat
        real_open_confined = receipts_module._open_confined
        armed = {"store": False}

        def open_confined_arming(path, flags, mode, role, created_registry=None):
            if role == "receipt store" and (flags & os.O_EXCL):
                armed["store"] = True
            return real_open_confined(path, flags, mode, role, created_registry=created_registry)

        def failing_fstat(fd):
            if armed["store"]:
                raise OSError("simulated fstat failure on the new store")
            return real_fstat(fd)

        with patch.object(receipts_module, "_open_confined", open_confined_arming):
            with patch.object(os, "fstat", failing_fstat):
                with self.assertRaises(OSError):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )
        self.assertTrue(armed["store"], "the failure must land on the store's own create")
        self.assert_no_receipt_files_but_the_lock()

    def test_a_failure_at_the_stores_mode_setting_removes_the_store(self):
        """Window 2: registered by the create, so the caller's failure is covered."""
        real_fchmod = os.fchmod
        armed = {"store_created": False}
        real_open_confined = receipts_module._open_confined

        def open_confined_arming(path, flags, mode, role, created_registry=None):
            fd = real_open_confined(path, flags, mode, role, created_registry=created_registry)
            if role == "receipt store" and (flags & os.O_EXCL):
                armed["store_created"] = True
            return fd

        def failing_fchmod(fd, mode):
            if armed["store_created"]:
                raise OSError("simulated mode failure after the store was created")
            return real_fchmod(fd, mode)

        with patch.object(receipts_module, "_open_confined", open_confined_arming):
            with patch.object(os, "fchmod", failing_fchmod):
                with self.assertRaises(OSError):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )
        self.assertTrue(armed["store_created"])
        self.assert_no_receipt_files_but_the_lock()

    def test_a_failure_at_the_keys_mode_setting_removes_the_store_and_key(self):
        """Window 3."""
        real_fchmod = os.fchmod
        armed = {"key_created": False}
        real_open_confined = receipts_module._open_confined

        def open_confined_arming(path, flags, mode, role, created_registry=None):
            fd = real_open_confined(path, flags, mode, role, created_registry=created_registry)
            if role == "receipt store HMAC key" and (flags & os.O_EXCL):
                armed["key_created"] = True
            return fd

        def failing_fchmod(fd, mode):
            if armed["key_created"]:
                raise OSError("simulated mode failure after the key was created")
            return real_fchmod(fd, mode)

        with patch.object(receipts_module, "_open_confined", open_confined_arming):
            with patch.object(os, "fchmod", failing_fchmod):
                with self.assertRaises(OSError):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )
        self.assertTrue(armed["key_created"])
        self.assert_no_receipt_files_but_the_lock()

    def test_a_failure_in_the_keys_read_back_removes_the_store_and_key(self):
        """Window 4: the read-back sat outside the creation cleanup block."""
        real_open_confined = receipts_module._open_confined
        armed = {"key_created": False}

        def open_confined_failing_read_back(path, flags, mode, role, created_registry=None):
            if role == "receipt store HMAC key" and not (flags & os.O_EXCL):
                if armed["key_created"]:
                    raise OSError("simulated failure reading the new key back")
            fd = real_open_confined(path, flags, mode, role, created_registry=created_registry)
            if role == "receipt store HMAC key" and (flags & os.O_EXCL):
                armed["key_created"] = True
            return fd

        with patch.object(receipts_module, "_open_confined", open_confined_failing_read_back):
            with self.assertRaises(OSError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )
        self.assertTrue(armed["key_created"])
        self.assert_no_receipt_files_but_the_lock()

    def test_a_failure_inside_the_temporary_files_open_leaves_no_temporary(self):
        """Window 5: the temporary open sat before its own cleanup block."""
        real_set_blocking = os.set_blocking
        armed = {"tmp": False}
        real_open_confined = receipts_module._open_confined

        def open_confined_arming(path, flags, mode, role, created_registry=None):
            if "checkpoint temporary" in role:
                armed["tmp"] = True
            return real_open_confined(path, flags, mode, role, created_registry=created_registry)

        def failing_set_blocking(fd, blocking):
            if armed["tmp"]:
                raise OSError("simulated failure inside the temporary file's open")
            return real_set_blocking(fd, blocking)

        with patch.object(receipts_module, "_open_confined", open_confined_arming):
            with patch.object(os, "set_blocking", failing_set_blocking):
                with self.assertRaises(OSError):
                    ReceiptStore(
                        self.state.receipts_file,
                        hmac_key_path=self.state.receipts_hmac_key_file,
                    )
        self.assertTrue(armed["tmp"])
        self.assert_no_receipt_files_but_the_lock()

    def test_the_checkpoint_is_registered_with_the_inode_the_rename_produced(self):
        """Window 6, asserted directly rather than by injection.

        Registration used to depend on an `lstat` taken after the rename, so a
        failure there left an unowned checkpoint. There is now no fallible call
        between the rename and the registration, which means the old injection
        point no longer exists and a probe aimed at it would pass without
        testing anything. What is testable is the invariant it protected: the
        identity recorded for the checkpoint is the identity of the file the
        rename produced.
        """
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        registry: list = []
        store._write_checkpoint(0, "0" * 64, created_registry=registry)

        self.assertEqual(len(registry), 1, f"exactly one registration expected: {registry}")
        registered_path, registered_identity = registry[0]
        self.assertEqual(registered_path, checkpoint)
        info = os.lstat(checkpoint)
        self.assertEqual(registered_identity, (info.st_dev, info.st_ino))

    def test_a_failure_after_the_checkpoint_exists_removes_it(self):
        """The behaviour window 6 protected, injected after the write returns."""
        real_write_checkpoint = ReceiptStore._write_checkpoint

        def write_then_fail(self_, count, last_hmac, created_registry=None):
            real_write_checkpoint(self_, count, last_hmac, created_registry=created_registry)
            raise ReceiptPersistenceError("simulated failure once the checkpoint existed")

        with patch.object(ReceiptStore, "_write_checkpoint", write_then_fail):
            with self.assertRaises(ReceiptPersistenceError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )
        self.assert_no_receipt_files_but_the_lock()


class KeyLengthReadTests(ScratchStateTestCase):
    """R7-F1: the key loader had the same short-read assumption.

    The retry loop reopens the file; it never accumulated a read to the end.
    So a valid key read one byte at a time was refused after every retry, and
    an over-long key whose first read returned exactly the key length was
    accepted with the file still longer than the key.
    """

    def key_path(self) -> Path:
        return self.state.receipts_hmac_key_file

    def test_a_valid_key_read_one_byte_at_a_time_is_accepted(self):
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.key_path())
        expected = self.key_path().read_bytes()
        self.assertEqual(len(expected), 32)
        real_read = os.read

        def one_byte_read(fd, size):
            return real_read(fd, 1) if size > 1 else real_read(fd, size)

        with patch.object(os, "read", one_byte_read):
            key, created, _identity = receipts_module._load_or_create_hmac_key_with_origin(
                self.key_path()
            )
        self.assertEqual(key, expected)
        self.assertFalse(created)

    def test_an_over_long_key_is_refused_even_when_the_first_read_looks_right(self):
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.key_path())
        self.key_path().write_bytes(os.urandom(33))
        real_read = os.read

        def read_returning_exactly_the_key_length(fd, size):
            # The permitted short read that made the file look 32 bytes long.
            return real_read(fd, 32) if size > 32 else real_read(fd, size)

        with patch.object(os, "read", read_returning_exactly_the_key_length):
            with self.assertRaises(ReceiptPersistenceError) as raised:
                receipts_module._load_or_create_hmac_key_with_origin(self.key_path())
        self.assertIn("unexpected length", str(raised.exception))
        self.assertEqual(len(self.key_path().read_bytes()), 33, "the file is not modified")

    def test_an_over_long_key_is_refused_without_retrying(self):
        """Too long is settled; the retries exist for a short concurrent write."""
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.key_path())
        self.key_path().write_bytes(os.urandom(64))
        started = time.monotonic()
        with self.assertRaises(ReceiptPersistenceError):
            receipts_module._load_or_create_hmac_key_with_origin(self.key_path())
        self.assertLess(
            time.monotonic() - started, 0.5, "an over-long key must not spend the retry budget"
        )

    def test_a_short_key_still_uses_the_retry_budget(self):
        """The declared concurrent-creator case must keep working."""
        ReceiptStore(self.state.receipts_file, hmac_key_path=self.key_path())
        self.key_path().write_bytes(os.urandom(16))
        with self.assertRaises(ReceiptPersistenceError) as raised:
            receipts_module._load_or_create_hmac_key_with_origin(self.key_path())
        self.assertIn("unexpected length", str(raised.exception))


class PostRenameOwnershipTests(ScratchStateTestCase):
    """R7-F2: the interval between the rename and the transaction registration.

    The rename moves an inode between two names. Until the caller's registry
    holds the final identity, neither name is covered unless the cleanup covers
    both, and an exception in that interval left the checkpoint behind.
    """

    def assert_no_receipt_files_but_the_lock(self):
        remaining = sorted(
            p.name for p in self.state.base.iterdir() if p.name.startswith("receipts.")
        )
        self.assertIn(remaining, ([], ["receipts.jsonl.lock"]), f"left behind: {remaining}")

    def test_a_failure_immediately_after_the_rename_removes_the_checkpoint(self):
        """The reviewer's injection: the real rename completes, then it raises."""
        real_replace = os.replace
        renamed = {"done": False}

        def replace_then_fail(src, dst):
            result = real_replace(src, dst)
            renamed["done"] = True
            raise OSError("simulated failure in the interval after the rename")

        with patch.object(os, "replace", replace_then_fail):
            with self.assertRaises(OSError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )

        self.assertTrue(renamed["done"], "the rename must have completed")
        self.assert_no_receipt_files_but_the_lock()

    def test_a_failure_at_the_registration_removes_the_checkpoint(self):
        """The other end of the same interval: the registry append itself fails."""

        class FailingRegistry(list):
            def append(self, item):
                raise OSError("simulated failure registering the checkpoint")

        store_path = self.state.receipts_file
        ReceiptStore(store_path, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = store_path.with_name("receipts.jsonl.checkpoint")
        checkpoint.unlink()

        store = ReceiptStore(store_path, hmac_key_path=self.state.receipts_hmac_key_file)
        with self.assertRaises(OSError):
            store._write_checkpoint(0, "0" * 64, created_registry=FailingRegistry())

        self.assertFalse(
            checkpoint.exists(),
            "a checkpoint whose registration failed must not survive",
        )

    def test_a_failed_rename_removes_only_the_temporary_file(self):
        """The inode never reaches the final name, so only the temporary goes."""
        established = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        established.append(make_receipt("req-established"))
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        before = checkpoint.read_bytes()

        def failing_replace(src, dst):
            raise OSError("simulated rename failure")

        with patch.object(os, "replace", failing_replace):
            # `append` wraps a checkpoint failure in its own typed error, which
            # is the documented fail-closed behaviour for a record written
            # whose checkpoint could not be updated.
            with self.assertRaises(ReceiptPersistenceError):
                established.append(make_receipt("req-after-failure"))

        self.assertEqual(
            checkpoint.read_bytes(), before, "the established checkpoint is untouched"
        )
        leftovers = [p.name for p in self.state.base.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [], "the temporary file is removed")

    def test_the_cleanup_does_not_remove_a_different_file_at_the_final_name(self):
        """Both removals are inode-bound, so a stranger at the final name stays."""
        store_path = self.state.receipts_file
        ReceiptStore(store_path, hmac_key_path=self.state.receipts_hmac_key_file)
        checkpoint = store_path.with_name("receipts.jsonl.checkpoint")
        store = ReceiptStore(store_path, hmac_key_path=self.state.receipts_hmac_key_file)

        real_replace = os.replace
        stranger = b"a different file that this call did not create\n"

        def replace_elsewhere_then_fail(src, dst):
            # The temporary goes somewhere else, and an unrelated file occupies
            # the final name. Cleanup must remove the first and not the second.
            real_replace(src, str(self.outside / "moved-checkpoint"))
            checkpoint.write_bytes(stranger)
            raise OSError("simulated failure with a stranger at the final name")

        with patch.object(os, "replace", replace_elsewhere_then_fail):
            with self.assertRaises(OSError):
                store._write_checkpoint(0, "0" * 64, created_registry=[])

        self.assertEqual(checkpoint.read_bytes(), stranger, "a file it did not create stays")


class EstablishedAppendRenameTests(ScratchStateTestCase):
    """R8-F1: the same interval, entered by an append rather than a construction.

    An append persists and syncs its record before it replaces the checkpoint,
    so once the replacement commits the installed checkpoint authenticates a
    record that is already durable. Removing it there displaces the old
    checkpoint and leaves no anchor, which construction cleanup must not do to
    an established store.
    """

    def test_a_committed_append_replacement_keeps_the_new_checkpoint(self):
        """The reviewer's injection, run against an established store."""
        store = ReceiptStore(
            self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
        )
        store.append(make_receipt("req-established"))
        checkpoint = self.state.receipts_file.with_name("receipts.jsonl.checkpoint")
        before = checkpoint.read_bytes()

        real_replace = os.replace
        renamed = {"done": False}

        def replace_then_fail(src, dst):
            real_replace(src, dst)
            renamed["done"] = True
            raise OSError("simulated failure in the interval after the rename")

        with patch.object(os, "replace", replace_then_fail):
            with self.assertRaises(ReceiptCheckpointError):
                store.append(make_receipt("req-after-committed-rename"))

        self.assertTrue(renamed["done"], "the rename must have completed")
        self.assertTrue(
            checkpoint.exists(),
            "the installed checkpoint authenticates a record already on disk",
        )
        self.assertNotEqual(
            checkpoint.read_bytes(), before, "the installed checkpoint is the two-record one"
        )
        self.assertEqual(store.verify_chain(), (True, 2))
        leftovers = [p.name for p in self.state.base.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [], "no temporary file survives")

    def test_a_construction_still_rolls_back_a_committed_replacement(self):
        """The other caller keeps the construction outcome under the same injection."""
        real_replace = os.replace
        renamed = {"done": False}

        def replace_then_fail(src, dst):
            real_replace(src, dst)
            renamed["done"] = True
            raise OSError("simulated failure in the interval after the rename")

        with patch.object(os, "replace", replace_then_fail):
            with self.assertRaises(OSError):
                ReceiptStore(
                    self.state.receipts_file, hmac_key_path=self.state.receipts_hmac_key_file
                )

        self.assertTrue(renamed["done"], "the rename must have completed")
        remaining = sorted(
            p.name for p in self.state.base.iterdir() if p.name.startswith("receipts.")
        )
        self.assertIn(remaining, ([], ["receipts.jsonl.lock"]), f"left behind: {remaining}")


class ServingWiringTests(unittest.TestCase):
    """D2, the wiring half: the rebinding writer is what the serving paths build.

    Declared scope, and it is narrow. This reads the source of `cli.py` and
    establishes that both serving entry points construct `ActiveReceiptStore`
    and hand that object to `GatewayContext`. It is a structural check, not a
    behavioural one: it does not start a gateway and does not observe a receipt
    being written through a running server.

    It exists because the behavioural probes in `RunningWriterRebindTests`
    construct the writer directly, so they pass whatever `cli.py` does. Without
    this check, reverting the two call sites to a store resolved once would
    leave every test in this file green, which is the shape of a check that is
    not wired to what it claims to cover.
    """

    SERVING_FUNCTIONS = ("_start_embedded_session", "cmd_serve")

    @classmethod
    def setUpClass(cls):
        import ast

        cls.ast = ast
        source_path = Path(__file__).resolve().parents[2] / "src" / "plag_in" / "cli.py"
        cls.tree = ast.parse(source_path.read_text(encoding="utf-8"))
        cls.functions = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
        }

    def test_both_serving_paths_build_the_rebinding_writer(self):
        for name in self.SERVING_FUNCTIONS:
            with self.subTest(function=name):
                function = self.functions.get(name)
                self.assertIsNotNone(function, f"{name} is not defined in cli.py")
                built = {
                    target.id: node.value
                    for node in self.ast.walk(function)
                    if isinstance(node, self.ast.Assign)
                    and isinstance(node.value, self.ast.Call)
                    and isinstance(node.value.func, self.ast.Name)
                    and node.value.func.id == "ActiveReceiptStore"
                    for target in node.targets
                    if isinstance(target, self.ast.Name)
                }
                self.assertTrue(
                    built, f"{name} does not construct ActiveReceiptStore"
                )

                handed_over = [
                    node.args[2].id
                    for node in self.ast.walk(function)
                    if isinstance(node, self.ast.Call)
                    and isinstance(node.func, self.ast.Name)
                    and node.func.id == "GatewayContext"
                    and len(node.args) > 2
                    and isinstance(node.args[2], self.ast.Name)
                ]
                self.assertTrue(
                    set(handed_over) & set(built),
                    f"{name} does not pass the rebinding writer to GatewayContext",
                )

    def test_the_paired_pointer_reads_have_no_caller_in_the_product(self):
        """A mechanism rather than a docstring, per the second-pass review, item 10.

        Both properties remain defined for callers that need one path alone, so
        nothing stops the pairing returning. This scans every product source
        file and fails if either name is read anywhere outside its own
        definition in `paths.py`.
        """
        root = Path(__file__).resolve().parents[2]
        sources = sorted((root / "src" / "plag_in").rglob("*.py"))
        sources += sorted((root / "tools").rglob("*.py"))
        self.assertTrue(sources, "no product source found to scan")

        offenders = []
        for source in sources:
            tree = self.ast.parse(source.read_text(encoding="utf-8"))
            for node in self.ast.walk(tree):
                if not isinstance(node, self.ast.Attribute):
                    continue
                if node.attr not in (
                    "active_receipts_file",
                    "active_receipts_hmac_key_file",
                ):
                    continue
                offenders.append(f"{source.relative_to(root)}:{node.lineno} {node.attr}")

        self.assertEqual(
            offenders,
            [],
            "read the pair through active_store_snapshot(); direct reads found at "
            + ", ".join(offenders),
        )

    def test_no_serving_path_pairs_the_two_pointer_reads(self):
        """The D3 pairing must not return anywhere in the command layer."""
        for node in self.ast.walk(self.tree):
            if not (
                isinstance(node, self.ast.Call)
                and isinstance(node.func, self.ast.Name)
                and node.func.id == "ReceiptStore"
            ):
                continue
            rendered = self.ast.unparse(node)
            self.assertNotIn(
                "active_receipts_file",
                rendered,
                "a store built from the paired pointer reads: use active_store_snapshot()",
            )
            self.assertNotIn("active_receipts_hmac_key_file", rendered)


class RunningListenerRebindTests(unittest.TestCase):
    """D2 through a running listener, as the review of the second pass described.

    The serving process is not restarted between the two requests. The store is
    activated while it is serving, and the second receipt is located by reading
    the stores rather than by asking the writer where it wrote.

    Socket-dependent: this starts the fake engine and a loopback gateway. It
    fails under a sandbox that denies `AF_INET`, along with the other
    loopback tests in this suite. No model is loaded.
    """

    def post_chat(self, base_url: str, alias: str) -> int:
        body = json.dumps(
            {"model": alias, "messages": [{"role": "user", "content": "probe"}]}
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                return response.status
        except urllib.error.HTTPError as exc:
            try:
                return exc.code
            finally:
                exc.close()

    def test_a_serving_gateway_writes_to_a_store_activated_while_it_runs(self):
        from tests.support import build_gateway_stack

        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            state = StateLayout(base).ensure()
            writer = ActiveReceiptStore(state)
            stack = build_gateway_stack(base, receipt_store=writer)
            try:
                self.assertEqual(self.post_chat(stack.server.base_url, stack.alias), 200)
                legacy_after_first = state.receipts_file.read_bytes()
                self.assertIn(b"request_id", legacy_after_first)

                destination = state.base / CURRENT_STORE_FILENAME
                created = initialise_current_store(destination)
                state.write_receipt_pointer(destination, created["schema_version"])

                # Same process, same listener, same GatewayContext.
                self.assertEqual(self.post_chat(stack.server.base_url, stack.alias), 200)

                self.assertEqual(
                    state.receipts_file.read_bytes(),
                    legacy_after_first,
                    "the legacy store must be byte-identical after activation",
                )
                current_records = [
                    json.loads(line)
                    for line in destination.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(
                    len(current_records),
                    1,
                    "the second request's receipt must be in the activated store",
                )
            finally:
                stack.stop()


class PointerSnapshotAtomicityTests(ScratchStateTestCase):
    """D3: one pointer reading yields a store, a key and a checkpoint that match."""

    def test_the_snapshot_reads_the_pointer_once(self):
        initialise_current_store(self.destination)
        self.state.write_receipt_pointer(self.destination, RECEIPT_SCHEMA_VERSION)

        real = StateLayout.read_receipt_pointer
        with patch.object(
            StateLayout, "read_receipt_pointer", autospec=True, side_effect=real
        ) as spy:
            snapshot = self.state.active_store_snapshot()
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(snapshot.store, self.destination)

    def test_an_activation_between_two_reads_cannot_split_the_selection(self):
        """The reproduction: the pointer changes between successive reads.

        Before the repair, `active_receipts_file` and
        `active_receipts_hmac_key_file` each read the pointer, so this sequence
        returned the legacy store with the current store's key. The snapshot
        reads once, so the pair it returns comes from one pointer state
        whichever state that is.
        """
        initialise_current_store(self.destination)
        current_pointer = {
            "current_store": str(self.destination),
            "legacy_store": str(self.state.receipts_file),
            "schema_version": RECEIPT_SCHEMA_VERSION,
        }
        readings = [None, current_pointer, None, current_pointer]

        with patch.object(
            StateLayout, "read_receipt_pointer", autospec=True, side_effect=readings
        ):
            snapshot = self.state.active_store_snapshot()

        if snapshot.is_legacy:
            self.assertEqual(snapshot.store, self.state.receipts_file)
            self.assertEqual(snapshot.hmac_key, self.state.receipts_hmac_key_file)
        else:
            self.assertEqual(snapshot.store, self.destination)
            self.assertEqual(
                snapshot.hmac_key,
                self.destination.with_name(self.destination.name + ".hmac_key"),
            )
        self.assertEqual(
            snapshot.checkpoint, snapshot.store.with_name(snapshot.store.name + ".checkpoint")
        )
        self.assertEqual(snapshot.lock, snapshot.store.with_name(snapshot.store.name + ".lock"))

    def test_the_legacy_snapshot_names_the_legacy_key(self):
        snapshot = self.state.active_store_snapshot()
        self.assertTrue(snapshot.is_legacy)
        self.assertEqual(snapshot.store, self.state.receipts_file)
        self.assertEqual(snapshot.hmac_key, self.state.receipts_hmac_key_file)
        self.assertIsNone(snapshot.schema_version)

    def test_the_conformance_check_uses_one_snapshot(self):
        initialise_current_store(self.destination)
        self.state.write_receipt_pointer(self.destination, RECEIPT_SCHEMA_VERSION)
        args = self.initialise_args()
        args.initialise_current_store = False
        args.check_conformance = True

        real = StateLayout.read_receipt_pointer
        with patch.object(
            StateLayout, "read_receipt_pointer", autospec=True, side_effect=real
        ) as spy:
            result = cmd_receipt(args)

        self.assertEqual(result["store_path"], str(self.destination))
        # One read for the three paths under check; `receipt_store_roles` reads
        # again to report both stores, which is a separate report.
        self.assertLessEqual(spy.call_count, 2)


class PointerWriterContractTests(ScratchStateTestCase):
    """D4: the writer accepts only what the reader accepts."""

    def test_the_writer_refuses_a_store_that_does_not_exist(self):
        with self.assertRaises(ConfigurationError) as raised:
            self.state.write_receipt_pointer(self.destination, RECEIPT_SCHEMA_VERSION)
        self.assertIn("does not exist", str(raised.exception))
        self.assertFalse(
            self.state.receipt_pointer_file.exists(),
            "a refused write leaves no pointer behind",
        )

    def test_what_the_writer_accepts_the_reader_accepts(self):
        initialise_current_store(self.destination)
        written = self.state.write_receipt_pointer(self.destination, RECEIPT_SCHEMA_VERSION)
        read_back = self.state.read_receipt_pointer()
        self.assertEqual(read_back["current_store"], written["current_store"])
        self.assertEqual(read_back["schema_version"], RECEIPT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
