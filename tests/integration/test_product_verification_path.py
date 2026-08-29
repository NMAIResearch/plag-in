"""The product verification command must enforce what the reference verifier enforces.

`plag-in receipt --check-chain`, the chat shutdown check and append admission all
run `ReceiptStore.verify_chain()`. That method used to strip and skip blank lines
and never validated the checkpoint's field types, so the product reported a store
valid that the reference verifier rejected
(CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F1).

These tests run the command as a subprocess and compare its verdict against the
reference verifier on the same bytes. The two implementations stay separate on
purpose: a shared implementation would share its bugs, and the reference verifier
exists to be a second opinion, not a second call site. What is required is that
they agree, which is what the agreement test below asserts.
"""
import hashlib
import hmac
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ReceiptChainError
from plag_in.identity import RUNTIME_SOURCE_SCOPE
from plag_in.receipts import Receipt, ReceiptStore

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
        schema_version="5",
        runtime_source_digest="d" * 64,
        runtime_source_scope=RUNTIME_SOURCE_SCOPE,
    )
    base.update(overrides)
    return Receipt(**base)


class ProductVerificationPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.state_dir / "receipts.jsonl",
            hmac_key_path=self.state_dir / "receipts.jsonl.hmac_key",
        )
        self.checkpoint = self.state_dir / "receipts.jsonl.checkpoint"
        self.store.append(make_receipt("req-0"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_command(self, *flags):
        """Run `plag-in receipt` as the user runs it."""
        command = [
            sys.executable, "-m", "plag_in", "receipt",
            *(flags or ("--check-chain",)), "--state-dir", str(self.state_dir),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return result

    def run_reference(self):
        command = [
            sys.executable, str(TOOL), str(self.store.path),
            "--hmac-key", str(self.store.hmac_key_path),
            "--checkpoint", str(self.checkpoint),
        ]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def command_says_valid(self):
        """The product storage-integrity verdict.

        A1 renamed this field: `verify_chain()` establishes chain, checkpoint
        and canonical storage agreement, so the command reports
        `integrity_valid` and names its scope rather than printing `valid`,
        which a caller would read as conformance.
        """
        result = self.run_command()
        if result.returncode != 0:
            return False
        payload = json.loads(result.stdout)
        self.assertEqual(payload["verification_scope"], "chain_checkpoint_storage")
        return payload["integrity_valid"]

    def command_says_conforming(self):
        """The product conformance verdict, the check the reference verifier mirrors."""
        result = self.run_command("--check-conformance")
        if result.returncode != 0:
            return False
        return json.loads(result.stdout)["conformance_valid"]

    def forge_boolean_anchor(self):
        """Authenticate `"count": true` under the store's own key."""
        key = self.store.hmac_key_path.read_bytes()
        last = json.loads(self.store.path.read_bytes().splitlines()[-1])["hmac"]
        message = f"checkpoint:{True}:{last}".encode("utf-8")
        self.checkpoint.write_text(
            json.dumps(
                {
                    "count": True,
                    "last_hmac": last,
                    "checkpoint_hmac": hmac.new(key, message, hashlib.sha256).hexdigest(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_a_clean_store_passes_both(self):
        self.assertTrue(self.command_says_valid())
        self.assertEqual(self.run_reference().returncode, 0)

    def test_blank_line_is_rejected_by_the_command(self):
        self.store.path.write_bytes(self.store.path.read_bytes() + b"\n")
        self.assertFalse(self.command_says_valid())
        self.assertEqual(self.run_reference().returncode, 1)

    def test_boolean_anchor_is_rejected_by_the_command(self):
        self.forge_boolean_anchor()
        self.assertFalse(self.command_says_valid())
        self.assertEqual(self.run_reference().returncode, 1)

    def test_non_canonical_bytes_are_rejected_by_the_command(self):
        record = json.loads(self.store.path.read_bytes())
        self.store.path.write_bytes(json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")
        self.assertFalse(self.command_says_valid())
        self.assertEqual(self.run_reference().returncode, 1)

    def test_unterminated_final_line_is_rejected_by_the_command(self):
        self.store.path.write_bytes(self.store.path.read_bytes().rstrip(b"\n"))
        self.assertFalse(self.command_says_valid())
        self.assertEqual(self.run_reference().returncode, 1)

    def test_the_command_does_not_alter_the_store(self):
        self.store.path.write_bytes(self.store.path.read_bytes() + b"\n")
        before_store = self.store.path.read_bytes()
        before_checkpoint = self.checkpoint.read_bytes()
        self.run_command()
        self.assertEqual(self.store.path.read_bytes(), before_store)
        self.assertEqual(self.checkpoint.read_bytes(), before_checkpoint)

    def test_append_admission_refuses_a_store_the_command_rejects(self):
        self.store.path.write_bytes(self.store.path.read_bytes() + b"\n")
        before = self.store.path.read_bytes()
        with self.assertRaises(ReceiptChainError):
            self.store.append(make_receipt("req-1"))
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_conformance_and_integrity_are_reported_as_distinct_results(self):
        """A1: neither output can be mistaken for the other."""
        integrity = json.loads(self.run_command("--check-chain").stdout)
        conformance = json.loads(self.run_command("--check-conformance").stdout)

        self.assertNotIn("valid", integrity)
        self.assertNotIn("conformance_valid", integrity)
        self.assertEqual(integrity["verification_scope"], "chain_checkpoint_storage")

        self.assertNotIn("valid", conformance)
        self.assertNotIn("integrity_valid", conformance)
        self.assertIn("conformance_level", conformance)
        self.assertIn("unassessed_checks", conformance)
        self.assertTrue(conformance["conformance_valid"])

    def test_the_two_implementations_agree_across_every_mutation(self):
        clean_store = self.store.path.read_bytes()
        clean_checkpoint = self.checkpoint.read_bytes()
        record = json.loads(clean_store)

        mutations = {
            "clean": lambda: None,
            "trailing_blank_line": lambda: self.store.path.write_bytes(clean_store + b"\n"),
            "leading_blank_line": lambda: self.store.path.write_bytes(b"\n" + clean_store),
            "whitespace_line": lambda: self.store.path.write_bytes(clean_store + b"  \n"),
            "no_terminator": lambda: self.store.path.write_bytes(clean_store.rstrip(b"\n")),
            "non_canonical": lambda: self.store.path.write_bytes(
                json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
            ),
            "boolean_anchor": self.forge_boolean_anchor,
            "truncated": lambda: self.store.path.write_bytes(b""),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                self.store.path.write_bytes(clean_store)
                self.checkpoint.write_bytes(clean_checkpoint)
                mutate()
                command_valid = self.command_says_valid()
                conformance_valid = self.command_says_conforming()
                reference_valid = self.run_reference().returncode == 0
                self.assertEqual(
                    command_valid,
                    reference_valid,
                    f"{name}: product_integrity={command_valid} reference={reference_valid}",
                )
                self.assertEqual(
                    conformance_valid,
                    reference_valid,
                    f"{name}: product_conformance={conformance_valid} reference={reference_valid}",
                )


if __name__ == "__main__":
    unittest.main()
