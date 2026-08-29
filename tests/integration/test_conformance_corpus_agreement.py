"""The product conformance path and the reference verifier must agree.

Every case in `tests/receipt_corpus.py` is verified twice: once through
`plag_in.conformance.check_conformance`, which the `plag-in receipt
--check-conformance` command calls, and once through
`tools/verify_inference_receipts.py` run as a subprocess. The two are separate
implementations of one specification, so agreement across the corpus is the
evidence that the contract is the same on both sides; a shared implementation
would agree with itself while proving nothing.

Agreement is required on the validity decision, the conformance level and the
exit status. The wording of a diagnostic is not required to match: each
implementation names the rule it refused in its own words, and requiring the
same string would be evidence that one was copied from the other.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plag_in.conformance import check_conformance
from tests.receipt_corpus import build_corpus

TOOL = Path(__file__).resolve().parents[2] / "tools" / "verify_inference_receipts.py"


def run_reference(paths: dict) -> subprocess.CompletedProcess:
    command = [sys.executable, str(TOOL), str(paths["store"])]
    if paths["key"] is not None:
        command += ["--hmac-key", str(paths["key"])]
    if paths["checkpoint"] is not None:
        command += ["--checkpoint", str(paths["checkpoint"])]
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def reference_level(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("conformance_assessed="):
            return line.split("=", 1)[1]
    return "<absent>"


class CorpusAgreementTests(unittest.TestCase):
    """Both implementations, every case, one subtest each."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_corpus_covers_every_required_area(self):
        """A2 names the areas the corpus must reach; none may quietly drop out."""
        names = {case.name for case in self.corpus}
        required = {
            "invalid_calendar_timestamp",
            "wrong_spec_version",
            "missing_spec_version",
            "duplicate_request_id",
            "nested_prohibited_prompt",
            "malformed_runtime_profile_requested",
            "empty_runtime_profile_object",
            "float_in_a_typed_field",
            "non_finite_float_in_an_implementation_defined_block",
            "non_canonical_stored_bytes",
            "missing_checkpoint_file",
            "malformed_checkpoint_json",
            "checkpoint_count_mismatch",
            "tail_deletion_with_an_anchor",
            "tail_deletion_without_an_anchor",
            "escaped_lone_surrogate_in_a_key",
            "escaped_lone_surrogate_in_a_value",
            "valid_core",
            "valid_core_anchor",
            "valid_extended",
            "valid_extended_anchor",
            # The independent review of work package A found these three
            # accepted by both implementations. Each stays in the required set
            # so neither can quietly stop covering them again.
            "measured_status_without_an_effective_value",
            "vacant_extended_identity_is_not_extended",
            "requested_without_a_meaningful_value_is_not_extended",
        }
        self.assertEqual(required - names, set())
        self.assertEqual(len(names), len(self.corpus), "corpus case names must be unique")

    def test_every_case_decides_and_levels_identically_in_both_implementations(self):
        for case in self.corpus:
            with self.subTest(case=case.name, rule=case.rule):
                paths = case.materialise(self.root)

                product = check_conformance(paths["store"], paths["key"], paths["checkpoint"])
                reference = run_reference(paths)

                self.assertEqual(
                    reference.returncode,
                    case.expect_exit,
                    f"{case.name}: reference exit {reference.returncode}\n{reference.stdout}\n"
                    f"{reference.stderr}",
                )
                self.assertEqual(
                    product["conformance_valid"],
                    case.expect_valid,
                    f"{case.name}: product errors {product['errors']}",
                )
                self.assertEqual(
                    product["conformance_valid"],
                    reference.returncode == 0,
                    f"{case.name}: product={product['conformance_valid']} "
                    f"reference_exit={reference.returncode}\n{reference.stdout}",
                )
                self.assertEqual(
                    product["conformance_level"],
                    case.expect_level,
                    f"{case.name}: product level {product['conformance_level']}",
                )
                self.assertEqual(
                    reference_level(reference.stdout),
                    case.expect_level,
                    f"{case.name}: reference level line\n{reference.stdout}",
                )

    def test_every_case_names_its_expected_diagnostic_in_both_implementations(self):
        for case in self.corpus:
            if case.reference_diagnostic is None and case.product_diagnostic is None:
                continue
            with self.subTest(case=case.name, rule=case.rule):
                paths = case.materialise(self.root)
                product = check_conformance(paths["store"], paths["key"], paths["checkpoint"])
                reference = run_reference(paths)

                messages = product["errors"] + product["unassessed_checks"]
                self.assertTrue(
                    any(case.product_diagnostic in message for message in messages),
                    f"{case.name}: {case.product_diagnostic!r} not in {messages}",
                )
                self.assertIn(case.reference_diagnostic, reference.stdout, case.name)

    def test_neither_implementation_raises_on_any_case(self):
        """A5 in particular: an unencodable record is a diagnostic, not a traceback."""
        for case in self.corpus:
            with self.subTest(case=case.name):
                paths = case.materialise(self.root)
                reference = run_reference(paths)
                self.assertEqual(reference.stderr, "", f"{case.name}: {reference.stderr}")
                self.assertNotIn("Traceback", reference.stdout, case.name)
                self.assertIn(reference.returncode, (0, 1, 2), case.name)

    def test_effective_runtime_is_claimed_measured_only_when_a_field_says_so(self):
        """A4: a requested profile never supports a measured-operation claim."""
        for case in self.corpus:
            with self.subTest(case=case.name):
                paths = case.materialise(self.root)
                product = check_conformance(paths["store"], paths["key"], paths["checkpoint"])
                reference = run_reference(paths)
                self.assertEqual(
                    product["effective_runtime_measured"],
                    case.expect_measured_runtime,
                    case.name,
                )
                expected_line = (
                    f"effective_runtime_measured={str(case.expect_measured_runtime).lower()}"
                )
                self.assertIn(expected_line, reference.stdout, case.name)

    def test_a_store_verified_without_an_anchor_names_tail_deletion_unassessed(self):
        case = next(c for c in self.corpus if c.name == "tail_deletion_without_an_anchor")
        paths = case.materialise(self.root)
        product = check_conformance(paths["store"], paths["key"], paths["checkpoint"])
        reference = run_reference(paths)

        self.assertTrue(product["conformance_valid"])
        self.assertEqual(reference.returncode, 0)
        self.assertTrue(
            any("tail deletion" in item for item in product["unassessed_checks"]),
            product["unassessed_checks"],
        )
        self.assertIn("tail deletion not assessed", reference.stdout)


class ProductCommandRunsTheCorpusTests(unittest.TestCase):
    """The command an operator runs must produce the module's own verdict.

    A check that is not reached by the user-facing command does not protect the
    user, which is the defect R3-F1 recorded against the previous revision.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_command(self, state_dir: Path) -> dict:
        command = [
            sys.executable, "-m", "plag_in", "receipt",
            "--check-conformance", "--state-dir", str(state_dir),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_the_command_reports_what_the_module_reports(self):
        for case in build_corpus():
            if not case.supply_key:
                continue  # the command always has the store's key beside it
            with self.subTest(case=case.name):
                state_dir = self.root / case.name
                state_dir.mkdir(parents=True)
                (state_dir / "receipts.jsonl").write_bytes(case.store)
                (state_dir / "receipts.jsonl.hmac_key").write_bytes(case.key)
                if case.checkpoint is not None:
                    (state_dir / "receipts.jsonl.checkpoint").write_bytes(case.checkpoint)

                payload = self.run_command(state_dir)
                expected = check_conformance(
                    state_dir / "receipts.jsonl",
                    state_dir / "receipts.jsonl.hmac_key",
                    state_dir / "receipts.jsonl.checkpoint",
                )
                self.assertEqual(payload["conformance_valid"], expected["conformance_valid"])
                self.assertEqual(payload["conformance_level"], expected["conformance_level"])
                self.assertEqual(payload["errors"], expected["errors"])


if __name__ == "__main__":
    unittest.main()
