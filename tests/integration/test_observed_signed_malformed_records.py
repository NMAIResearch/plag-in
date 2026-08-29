"""Every signed malformed record observed in review must now fail both paths.

These are the records and stores that authenticated correctly under the store
key and were nevertheless accepted, or accepted by one implementation and
refused by the other, across the three independent reviews. Each is rebuilt
here from its review case name and run through both the product conformance
path and the reference verifier, which must both refuse it and must both name
the rule they refused it under.

The instruction behind this file names five such records. The review record
does not resolve to exactly five without a judgement about which observations
count as records rather than as store or anchor states, so the set below is the
superset: every signed case from `CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md`
and `CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md` that a verifier is
supposed to catch. Covering the superset satisfies the requirement under any
reading of which five were meant.

Not included, because each is spec-valid as a record and the defect was in the
emitter rather than in what a verifier can see: `declared_string_fields_as_floats`
and the string rendering in `nonfinite_float`. Both are covered emitter-side in
`test_receipt_spec_conformance.py`.
"""
import hashlib
import hmac
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plag_in.conformance import check_conformance
from tests.receipt_corpus import (
    CORPUS_KEY,
    anchor_for,
    canonical,
    core_record,
    sign_chain,
    store_bytes,
)

TOOL = Path(__file__).resolve().parents[2] / "tools" / "verify_inference_receipts.py"

# Each entry: review case name -> (store bytes, anchor bytes or None, the
# substring each implementation must produce).
WORKER_DIALECT_PROFILE = {
    "engine": "llama_server_direct",
    "arguments": {"context_tokens": 4096, "gpu_layers": 20},
    "measurement_status": "not_applicable_to_worker_adapter",
}


def one_record(**overrides) -> tuple[bytes, bytes]:
    records = sign_chain([core_record("req-observed", **overrides)])
    return store_bytes(records), anchor_for(records)


def observed_cases() -> list[dict]:
    cases: list[dict] = []

    store, anchor = one_record(timestamp="2026-13-45T99:99:99+00:00")
    cases.append({
        "name": "invalid_rfc3339_ranges",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md",
        "store": store,
        "anchor": anchor,
        "reference": "timestamp is not RFC 3339",
        "product": "timestamp is not a real RFC 3339 instant",
    })

    store, anchor = one_record(timestamp="2026-01-01T12:34:60Z")
    cases.append({
        "name": "non_event_leap_second",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F4",
        "store": store,
        "anchor": anchor,
        "reference": "timestamp is not RFC 3339",
        "product": "timestamp is not a real RFC 3339 instant",
    })

    store, anchor = one_record(
        source_identity={"digest": "short", "scope": ""},
        engine_identity={"version": 7},
        runtime_profile={"requested": "text", "effective": [], "measurement_status": 1},
        locality={"level": "", "evidence_class": "", "listen_address": ""},
    )
    cases.append({
        "name": "invalid_optional_block_shapes",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F2",
        "store": store,
        "anchor": anchor,
        "reference": "source_identity.digest must be 64 lowercase hex",
        "product": "source_identity.digest must be 64 lowercase hex",
    })

    store, anchor = one_record(runtime_profile=WORKER_DIALECT_PROFILE)
    cases.append({
        "name": "C1_worker_receipt",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F2",
        "store": store,
        "anchor": anchor,
        "reference": "runtime_profile.requested must be an object",
        "product": "runtime_profile.requested must be an object",
    })

    duplicates = sign_chain([core_record("req-same"), core_record("req-same")])
    cases.append({
        "name": "emitter_duplicate_request_id",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F6",
        "store": store_bytes(duplicates),
        "anchor": anchor_for(duplicates),
        "reference": "duplicate request_id",
        "product": "duplicate request_id",
    })

    store, anchor = one_record(plag_in_adapter_record={"sampling": {"temperature": float("nan")}})
    cases.append({
        "name": "nonfinite_float",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F4",
        "store": store,
        "anchor": anchor,
        "reference": "prohibited float at",
        "product": "prohibited float at",
    })

    clean = sign_chain([core_record("req-blank")])
    cases.append({
        "name": "extra_blank_line",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F3",
        "store": store_bytes(clean) + b"\n",
        "anchor": anchor_for(clean),
        "reference": "blank or whitespace-only line",
        "product": "blank or whitespace-only line",
    })

    boolean_message = f"checkpoint:{True}:{clean[-1]['hmac']}".encode("utf-8")
    boolean_anchor = json.dumps(
        {
            "count": True,
            "last_hmac": clean[-1]["hmac"],
            "checkpoint_hmac": hmac.new(CORPUS_KEY, boolean_message, hashlib.sha256).hexdigest(),
        },
        sort_keys=True,
    ).encode("utf-8")
    cases.append({
        "name": "boolean_checkpoint_count",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F1",
        "store": store_bytes(clean),
        "anchor": boolean_anchor,
        "reference": "checkpoint count must be an integer",
        "product": "checkpoint count must be an integer",
    })

    cases.append({
        "name": "missing_engine_identity_fields",
        "review": "CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F3",
        "store": store_bytes(sign_chain([core_record(
            "req-nullable",
            engine_identity={"engine": "libllama_embedded", "version": "0.1.0a2"},
        )])),
        "anchor": None,
        "reference": None,
        "product": None,
        "expect_valid": True,
        "note": (
            "resolved as valid rather than as a defect: section 5 now states that a "
            "nullable key may be absent and that absence means null. Kept here so the "
            "decision is visible beside the cases that do fail."
        ),
    })

    return cases


class ObservedSignedMalformedRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def materialise(self, case: dict) -> dict:
        base = self.root / case["name"]
        base.mkdir(parents=True, exist_ok=True)
        store = base / "receipts.jsonl"
        store.write_bytes(case["store"])
        key = base / "receipts.jsonl.hmac_key"
        key.write_bytes(CORPUS_KEY)
        checkpoint = base / "receipts.jsonl.checkpoint"
        if case["anchor"] is not None:
            checkpoint.write_bytes(case["anchor"])
        return {
            "store": store,
            "key": key,
            "checkpoint": checkpoint if case["anchor"] is not None else None,
        }

    def run_reference(self, paths: dict) -> subprocess.CompletedProcess:
        command = [sys.executable, str(TOOL), str(paths["store"]), "--hmac-key", str(paths["key"])]
        if paths["checkpoint"] is not None:
            command += ["--checkpoint", str(paths["checkpoint"])]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def test_each_observed_case_reaches_the_recorded_verdict_in_both_implementations(self):
        for case in observed_cases():
            with self.subTest(case=case["name"], review=case["review"]):
                paths = self.materialise(case)
                product = check_conformance(paths["store"], paths["key"], paths["checkpoint"])
                reference = self.run_reference(paths)
                expect_valid = case.get("expect_valid", False)

                self.assertEqual(product["conformance_valid"], expect_valid, product["errors"])
                self.assertEqual(reference.returncode, 0 if expect_valid else 1, reference.stdout)

                if expect_valid:
                    continue
                self.assertTrue(
                    any(case["product"] in message for message in product["errors"]),
                    f"{case['name']}: {case['product']!r} not in {product['errors']}",
                )
                self.assertIn(case["reference"], reference.stdout, case["name"])

    def test_the_chain_authenticates_every_malformed_record(self):
        """Each case is malformed, not corrupt: the HMAC over it verifies.

        Without this the cases would prove only that a broken chain is caught,
        which the chain check already established. The point of each case is a
        record the chain accepts and the specification does not.
        """
        for case in observed_cases():
            if case["name"] in ("extra_blank_line", "boolean_checkpoint_count"):
                continue  # store-level and anchor-level, not record bodies
            with self.subTest(case=case["name"]):
                previous = "0" * 64
                for raw in case["store"].splitlines():
                    record = json.loads(raw.decode("utf-8"))
                    self.assertEqual(record["prev_hmac"], previous)
                    body = {k: v for k, v in record.items() if k != "hmac"}
                    digest = hashlib.sha256(canonical(body)).hexdigest()
                    message = f"{previous}:{digest}".encode("utf-8")
                    self.assertEqual(
                        hmac.new(CORPUS_KEY, message, hashlib.sha256).hexdigest(),
                        record["hmac"],
                    )
                    previous = record["hmac"]


if __name__ == "__main__":
    unittest.main()
