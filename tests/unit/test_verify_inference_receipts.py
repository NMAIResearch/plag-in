"""Regression suite for the reference verifier, tools/verify_inference_receipts.py.

The verifier is a separate program from the gateway and had no test of its own
(CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D8), so the 314-test application
suite covered none of its exit contract, chain handling or byte handling.

Fixtures here are built from the specification alone, without importing
`plag_in`, so a defect in the implementation cannot mask a defect in the
verifier. Every test asserts the exit status as well as stdout: the exit status
is the part of the contract another program consumes.
"""
import hashlib
import hmac
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "verify_inference_receipts.py"

_KEY = b"k" * 32
_GENESIS = "0" * 64

EXIT_VALID = 0
EXIT_INVALID = 1
EXIT_INCOMPLETE = 2


def canonical(record: dict) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def record_hmac(key: bytes, prev_hmac: str, body: dict) -> str:
    message = f"{prev_hmac}:{hashlib.sha256(canonical(body)).hexdigest()}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def make_record(request_id: str, prev_hmac: str, key: bytes = _KEY, **overrides) -> dict:
    body = {
        "spec_version": "0.1",
        "request_id": request_id,
        "timestamp": "2026-08-26T00:00:00+00:00",
        "weight_digest": "a" * 64,
        "content_retained": False,
        "status": "completed",
        "input_tokens": 7,
        "output_tokens": 5,
        "integrity_mode": "local_hmac",
        "prev_hmac": prev_hmac,
    }
    body.update(overrides)
    return {**body, "hmac": record_hmac(key, prev_hmac, body)}


def make_chain(count: int, key: bytes = _KEY) -> list[dict]:
    records = []
    prev = _GENESIS
    for index in range(count):
        record = make_record(f"req-{index}", prev, key=key)
        records.append(record)
        prev = record["hmac"]
    return records


class VerifierTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = self.tmp_path / "receipts.jsonl"
        self.key_path = self.tmp_path / "key"
        self.key_path.write_bytes(_KEY)

    def tearDown(self):
        self.tmp.cleanup()

    def write_store(self, records, terminator: bytes = b"\n") -> None:
        self.store.write_bytes(b"".join(canonical(r) + terminator for r in records))

    def write_checkpoint(self, count: int, last_hmac: str, key: bytes = _KEY) -> Path:
        message = f"checkpoint:{count}:{last_hmac}".encode("utf-8")
        path = self.store.with_name(self.store.name + ".checkpoint")
        path.write_text(
            json.dumps(
                {
                    "count": count,
                    "last_hmac": last_hmac,
                    "checkpoint_hmac": hmac.new(key, message, hashlib.sha256).hexdigest(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def run_verifier(self, *args, store=None):
        command = [sys.executable, str(TOOL), str(store or self.store), *args]
        return subprocess.run(command, capture_output=True, text=True, timeout=60)


class ExitContractTests(VerifierTestCase):
    def test_conforming_store_with_key_is_valid(self):
        self.write_store(make_chain(2))
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertIn("result=valid", result.stdout)
        self.assertIn("records_read=2", result.stdout)

    def test_missing_key_is_incomplete_not_valid(self):
        # D1: chain linkage is mandatory, so a run that could not perform it
        # must never report a conforming result or exit 0.
        self.write_store(make_chain(2))
        result = self.run_verifier()
        self.assertEqual(result.returncode, EXIT_INCOMPLETE, result.stdout)
        self.assertIn("result=incomplete", result.stdout)
        self.assertIn("SKIPPED HMAC chain linkage not verified", result.stdout)
        self.assertNotIn("result=valid", result.stdout)

    def test_unreadable_key_is_invalid(self):
        self.write_store(make_chain(1))
        result = self.run_verifier("--hmac-key", str(self.tmp_path / "absent-key"))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("cannot read HMAC key", result.stdout)

    def test_missing_store_reports_a_structured_diagnostic(self):
        # D7: the early return path used to yield a bare Findings object, so
        # main() raised TypeError instead of printing the verifier's own
        # diagnostic. It failed closed, but told the caller nothing.
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), store=self.tmp_path / "absent.jsonl"
        )
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("cannot read store", result.stdout)
        self.assertIn("records_read=0", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_empty_store_is_invalid(self):
        self.store.write_bytes(b"")
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("store contains no records", result.stdout)


class ChainTests(VerifierTestCase):
    def test_mutated_record_is_rejected(self):
        records = make_chain(2)
        records[0]["status"] = "failed"
        self.write_store(records)
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("HMAC does not authenticate this record", result.stdout)

    def test_tail_deletion_without_anchor_is_named_as_unassessed(self):
        # D2: an HMAC chain has no record-count anchor, so a truncated chain
        # verifies. The verifier may not call that valid without saying what it
        # did not check.
        records = make_chain(2)
        self.write_store(records[:1])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertIn("SKIPPED tail deletion not assessed", result.stdout)
        self.assertIn("conformance_assessed=core", result.stdout)
        self.assertNotIn("conformance_assessed=core+anchor", result.stdout)

    def test_tail_deletion_with_anchor_is_detected(self):
        records = make_chain(2)
        checkpoint = self.write_checkpoint(2, records[-1]["hmac"])
        self.write_store(records[:1])
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("record count does not match the authenticated checkpoint", result.stdout)

    def test_intact_chain_with_anchor_is_valid(self):
        records = make_chain(2)
        checkpoint = self.write_checkpoint(2, records[-1]["hmac"])
        self.write_store(records)
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertIn("conformance_assessed=core+anchor", result.stdout)

    def test_forged_anchor_is_rejected(self):
        records = make_chain(2)
        checkpoint = self.write_checkpoint(1, records[0]["hmac"], key=b"z" * 32)
        self.write_store(records[:1])
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("checkpoint HMAC does not authenticate the checkpoint", result.stdout)

    def test_anchor_without_key_is_incomplete(self):
        records = make_chain(1)
        checkpoint = self.write_checkpoint(1, records[0]["hmac"])
        self.write_store(records)
        result = self.run_verifier("--checkpoint", str(checkpoint))
        self.assertEqual(result.returncode, EXIT_INCOMPLETE, result.stdout)
        self.assertIn("anchor not verified", result.stdout)


class RecordTests(VerifierTestCase):
    def test_duplicate_request_id_is_rejected(self):
        # D3: section 2 requires request_id to be unique within the store. A
        # chain over two records sharing one id still verifies.
        prev = _GENESIS
        first = make_record("req-duplicate", prev)
        second = make_record("req-duplicate", first["hmac"])
        self.write_store([first, second])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("duplicate request_id", result.stdout)

    def test_nested_prohibited_field_is_rejected(self):
        # D4: a prompt nested inside an implementation block is the same
        # disclosure as a prompt at the top level.
        record = make_record(
            "req-0", _GENESIS, implementation_block={"inner": {"prompt": "leaked text"}}
        )
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("prohibited content field present", result.stdout)
        self.assertIn("implementation_block.inner", result.stdout)

    def test_top_level_prohibited_field_is_still_rejected(self):
        record = make_record("req-0", _GENESIS, prompt="leaked text")
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("prohibited content field present", result.stdout)

    def test_additional_top_level_field_is_accepted(self):
        # D4: the core is a minimum, not an exact schema. An implementation
        # field that is not prohibited content must not be rejected.
        record = make_record("req-0", _GENESIS, gateway_version="0.1.0a2")
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)

    def test_missing_core_field_is_rejected(self):
        body = make_record("req-0", _GENESIS)
        del body["status"]
        self.write_store([body])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("mandatory core field absent: status", result.stdout)

    def test_nested_float_is_rejected(self):
        record = make_record("req-0", _GENESIS, profile={"sampling": {"temperature": 0.7}})
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("prohibited float at profile.sampling.temperature", result.stdout)

    def test_float_inside_a_list_is_rejected(self):
        record = make_record("req-0", _GENESIS, samples=[1, 2.5])
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("prohibited float at samples[1]", result.stdout)

    def test_weight_digest_with_trailing_newline_is_rejected(self):
        record = make_record("req-0", _GENESIS, weight_digest="a" * 64 + "\n")
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("weight_digest must be 64 lowercase hex characters", result.stdout)


class StoredByteTests(VerifierTestCase):
    def test_non_canonical_stored_bytes_are_rejected(self):
        # D5: parsing and reserialising before comparison would accept any
        # rendering a JSON library produced, which defeats canonicalisation.
        record = make_record("req-0", _GENESIS)
        self.store.write_bytes(
            json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        )
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("stored bytes are not canonical", result.stdout)

    def test_crlf_terminated_line_is_rejected(self):
        self.write_store(make_chain(1), terminator=b"\r\n")
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("newline translation altered the bytes", result.stdout)

    def test_unterminated_final_line_is_rejected(self):
        self.store.write_bytes(canonical(make_chain(1)[0]))
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("final record is not LF terminated", result.stdout)

    def test_raw_non_ascii_is_accepted_and_escaped_form_is_not(self):
        record = make_record("req-0", _GENESIS, model_alias="modèle")
        self.write_store([record])
        accepted = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(accepted.returncode, EXIT_VALID, accepted.stdout)

        self.store.write_bytes(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        rejected = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(rejected.returncode, EXIT_INVALID, rejected.stdout)
        self.assertIn("stored bytes are not canonical", rejected.stdout)


COMPLETE_OPTIONAL_BLOCKS = {
    "source_identity": {"digest": "b" * 64, "scope": "runtime_python_source:src/**/*.py"},
    "engine_identity": {
        "engine": "llama_server",
        "version": "b4321",
        "native_identity_digest": "c" * 64,
        "component_digests": {"libllama": "d" * 64},
        "upstream_identity": "ggml-org/llama.cpp@b4321",
    },
    "runtime_profile": {
        "requested": {"context_size": 4096},
        "effective": {"context_size": 4096},
        "measurement_status": {"context_size": "measured"},
    },
    "locality": {
        "level": "L1",
        "evidence_class": "loopback_bind_and_backend_observed",
        "listen_address": "127.0.0.1:8080",
    },
}


class AnchorShapeTests(VerifierTestCase):
    """F1: an anchor must be validated before it is authenticated."""

    def write_raw_checkpoint(self, count, last_hmac, key: bytes = _KEY) -> Path:
        message = f"checkpoint:{count}:{last_hmac}".encode("utf-8")
        path = self.store.with_name(self.store.name + ".checkpoint")
        path.write_text(
            json.dumps(
                {
                    "count": count,
                    "last_hmac": last_hmac,
                    "checkpoint_hmac": hmac.new(key, message, hashlib.sha256).hexdigest(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def assert_rejected(self, checkpoint, fragment):
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn(fragment, result.stdout)
        self.assertNotIn("result=valid", result.stdout)

    def test_boolean_count_is_rejected(self):
        # True == 1 in Python, so a boolean count authenticates and then
        # compares equal to a one-record store.
        records = make_chain(1)
        self.write_store(records)
        checkpoint = self.write_raw_checkpoint(True, records[0]["hmac"])
        self.assert_rejected(checkpoint, "checkpoint count must be an integer, not bool")

    def test_float_count_is_rejected(self):
        records = make_chain(1)
        self.write_store(records)
        checkpoint = self.write_raw_checkpoint(1.0, records[0]["hmac"])
        self.assert_rejected(checkpoint, "checkpoint count must be an integer, not float")

    def test_negative_count_is_rejected(self):
        records = make_chain(1)
        self.write_store(records)
        checkpoint = self.write_raw_checkpoint(-1, records[0]["hmac"])
        self.assert_rejected(checkpoint, "checkpoint count must be zero or greater")

    def test_malformed_last_hmac_is_rejected(self):
        records = make_chain(1)
        self.write_store(records)
        checkpoint = self.write_raw_checkpoint(1, "not-a-digest")
        self.assert_rejected(checkpoint, "checkpoint last_hmac must be 64 lowercase hex characters")

    def test_malformed_checkpoint_hmac_is_rejected(self):
        records = make_chain(1)
        self.write_store(records)
        path = self.store.with_name(self.store.name + ".checkpoint")
        path.write_text(
            json.dumps(
                {"count": 1, "last_hmac": records[0]["hmac"], "checkpoint_hmac": "short"},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.assert_rejected(path, "checkpoint_hmac must be 64 lowercase hex characters")

    def test_uppercase_last_hmac_is_rejected(self):
        records = make_chain(1)
        self.write_store(records)
        checkpoint = self.write_raw_checkpoint(1, records[0]["hmac"].upper())
        self.assert_rejected(checkpoint, "checkpoint last_hmac must be 64 lowercase hex characters")


class OptionalBlockTests(VerifierTestCase):
    """F2: every field a present optional block declares must be validated."""

    def verify_with(self, **blocks):
        record = make_record("req-0", _GENESIS, **blocks)
        self.write_store([record])
        return self.run_verifier("--hmac-key", str(self.key_path))

    def assert_block_rejected(self, fragment, **blocks):
        result = self.verify_with(**blocks)
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn(fragment, result.stdout)

    def test_complete_optional_blocks_are_accepted(self):
        result = self.verify_with(**COMPLETE_OPTIONAL_BLOCKS)
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)

    def test_string_component_digests_is_rejected(self):
        engine = dict(COMPLETE_OPTIONAL_BLOCKS["engine_identity"], component_digests="libllama")
        self.assert_block_rejected(
            "engine_identity.component_digests must be an object or null", engine_identity=engine
        )

    def test_malformed_component_digest_value_is_rejected(self):
        engine = dict(
            COMPLETE_OPTIONAL_BLOCKS["engine_identity"], component_digests={"libllama": "nope"}
        )
        self.assert_block_rejected(
            "engine_identity.component_digests.libllama must be 64 lowercase hex",
            engine_identity=engine,
        )

    def test_integer_upstream_identity_is_rejected(self):
        engine = dict(COMPLETE_OPTIONAL_BLOCKS["engine_identity"], upstream_identity=4321)
        self.assert_block_rejected(
            "engine_identity.upstream_identity must be a string or null", engine_identity=engine
        )

    def test_null_component_digests_and_upstream_identity_are_accepted(self):
        engine = dict(
            COMPLETE_OPTIONAL_BLOCKS["engine_identity"],
            component_digests=None,
            upstream_identity=None,
        )
        result = self.verify_with(engine_identity=engine)
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)

    def test_string_runtime_profile_requested_is_rejected(self):
        profile = dict(COMPLETE_OPTIONAL_BLOCKS["runtime_profile"], requested="context_size=4096")
        self.assert_block_rejected(
            "runtime_profile.requested must be an object", runtime_profile=profile
        )

    def test_missing_runtime_profile_requested_is_rejected(self):
        profile = dict(COMPLETE_OPTIONAL_BLOCKS["runtime_profile"])
        del profile["requested"]
        self.assert_block_rejected(
            "runtime_profile.requested must be an object", runtime_profile=profile
        )

    def test_empty_locality_object_is_rejected(self):
        result = self.verify_with(locality={})
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        for field in ("level", "evidence_class", "listen_address"):
            self.assertIn(f"locality.{field} must be a non-empty string", result.stdout)

    def test_each_locality_field_is_validated(self):
        for field, bad_value in (
            ("level", 1),
            ("evidence_class", ""),
            ("listen_address", None),
        ):
            with self.subTest(field=field):
                locality = dict(COMPLETE_OPTIONAL_BLOCKS["locality"])
                locality[field] = bad_value
                self.assert_block_rejected(
                    f"locality.{field} must be a non-empty string", locality=locality
                )


class NullableKeyTests(VerifierTestCase):
    """R3-F3: absence and explicit null are the same thing, and are stated to be."""

    def verify_with_engine_identity(self, engine_identity):
        record = make_record("req-0", _GENESIS, engine_identity=engine_identity)
        self.write_store([record])
        return self.run_verifier("--hmac-key", str(self.key_path))

    def test_absent_nullable_keys_are_accepted(self):
        result = self.verify_with_engine_identity({"engine": "llama_server", "version": "b1"})
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)

    def test_absent_and_explicitly_null_forms_agree(self):
        absent = self.verify_with_engine_identity({"engine": "llama_server", "version": "b1"})
        explicit = self.verify_with_engine_identity({
            "engine": "llama_server",
            "version": "b1",
            "native_identity_digest": None,
            "component_digests": None,
            "upstream_identity": None,
        })
        self.assertEqual(absent.returncode, explicit.returncode)
        self.assertEqual(absent.returncode, EXIT_VALID, absent.stdout)

    def test_absent_mandatory_engine_key_is_rejected(self):
        result = self.verify_with_engine_identity({"version": "b1"})
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("engine_identity.engine must be a string", result.stdout)

    def test_absent_mandatory_version_key_is_rejected(self):
        result = self.verify_with_engine_identity({"engine": "llama_server"})
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("engine_identity.version must be a string", result.stdout)


class ConformanceLevelTests(VerifierTestCase):
    """R3-F6: the verifier reports the level it assessed, all four values."""

    def level_of(self, result):
        for line in result.stdout.splitlines():
            if line.startswith("conformance_assessed="):
                return line.split("=", 1)[1]
        raise AssertionError(f"no level reported: {result.stdout}")

    def test_core_when_blocks_are_absent(self):
        self.write_store(make_chain(1))
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertEqual(self.level_of(result), "core")

    def test_core_plus_anchor_when_only_the_anchor_is_supplied(self):
        records = make_chain(1)
        checkpoint = self.write_checkpoint(1, records[0]["hmac"])
        self.write_store(records)
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(self.level_of(result), "core+anchor")

    def test_extended_when_every_record_carries_the_blocks(self):
        record = make_record("req-0", _GENESIS, **COMPLETE_OPTIONAL_BLOCKS)
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertEqual(self.level_of(result), "extended")

    def test_extended_plus_anchor(self):
        record = make_record("req-0", _GENESIS, **COMPLETE_OPTIONAL_BLOCKS)
        checkpoint = self.write_checkpoint(1, record["hmac"])
        self.write_store([record])
        result = self.run_verifier(
            "--hmac-key", str(self.key_path), "--checkpoint", str(checkpoint)
        )
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertEqual(self.level_of(result), "extended+anchor")

    def test_one_record_short_of_the_blocks_caps_the_store_at_core(self):
        first = make_record("req-0", _GENESIS, **COMPLETE_OPTIONAL_BLOCKS)
        second = make_record("req-1", first["hmac"])
        self.write_store([first, second])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)
        self.assertEqual(self.level_of(result), "core")

    def test_runtime_profile_alone_is_not_enough_for_extended(self):
        record = make_record(
            "req-0", _GENESIS, runtime_profile=COMPLETE_OPTIONAL_BLOCKS["runtime_profile"]
        )
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(self.level_of(result), "core")


class BlankLineTests(VerifierTestCase):
    """F3: the store admits one canonical record per line and nothing else."""

    def assert_blank_line_rejected(self, raw: bytes, expected_line: int):
        self.store.write_bytes(raw)
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn(f"line {expected_line}: blank or whitespace-only line", result.stdout)

    def test_leading_blank_line_is_rejected(self):
        record = canonical(make_chain(1)[0])
        self.assert_blank_line_rejected(b"\n" + record + b"\n", 1)

    def test_interior_blank_line_is_rejected(self):
        records = make_chain(2)
        raw = canonical(records[0]) + b"\n" + b"\n" + canonical(records[1]) + b"\n"
        self.assert_blank_line_rejected(raw, 2)

    def test_surplus_trailing_blank_line_is_rejected(self):
        record = canonical(make_chain(1)[0])
        self.assert_blank_line_rejected(record + b"\n" + b"\n", 2)

    def test_whitespace_only_line_is_rejected(self):
        record = canonical(make_chain(1)[0])
        self.assert_blank_line_rejected(record + b"\n" + b"   \t" + b"\n", 2)


class TimestampTests(VerifierTestCase):
    """F5: an anchored pattern is not a calendar."""

    def assert_timestamp_rejected(self, timestamp):
        record = make_record("req-0", _GENESIS, timestamp=timestamp)
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_INVALID, result.stdout)
        self.assertIn("timestamp is not RFC 3339 with offset", result.stdout)

    def test_impossible_month_is_rejected(self):
        self.assert_timestamp_rejected("2026-13-01T00:00:00+00:00")

    def test_impossible_day_is_rejected(self):
        self.assert_timestamp_rejected("2026-01-32T00:00:00+00:00")

    def test_impossible_hour_is_rejected(self):
        self.assert_timestamp_rejected("2026-01-01T24:00:00+00:00")

    def test_impossible_offset_is_rejected(self):
        self.assert_timestamp_rejected("2026-01-01T00:00:00+99:99")

    def test_the_reviewed_probe_value_is_rejected(self):
        self.assert_timestamp_rejected("2026-99-99T99:99:99+99:99")

    def test_leap_day_in_a_common_year_is_rejected(self):
        self.assert_timestamp_rejected("2026-02-29T00:00:00+00:00")

    def test_leap_day_in_a_leap_year_is_accepted(self):
        record = make_record("req-0", _GENESIS, timestamp="2024-02-29T00:00:00+00:00")
        self.write_store([record])
        result = self.run_verifier("--hmac-key", str(self.key_path))
        self.assertEqual(result.returncode, EXIT_VALID, result.stdout)

    def test_end_of_day_hour_is_rejected(self):
        # ISO 8601 permits 24:00:00; RFC 3339 section 5.6 bounds the hour at 23.
        self.assert_timestamp_rejected("2026-01-01T24:00:00+00:00")

    def test_leap_second_is_rejected_in_version_0_1(self):
        # R3-F4: substituting 59 range-checked everything except the one thing
        # that makes second 60 legal, so version 0.1 excludes it outright.
        self.assert_timestamp_rejected("2016-12-31T23:59:60+00:00")

    def test_non_event_leap_second_is_rejected(self):
        self.assert_timestamp_rejected("2026-01-01T12:34:60Z")

    def test_zulu_and_fractional_seconds_are_accepted(self):
        for timestamp in ("2026-08-26T00:00:00Z", "2026-08-26T00:00:00.123456+02:00"):
            with self.subTest(timestamp=timestamp):
                record = make_record("req-0", _GENESIS, timestamp=timestamp)
                self.write_store([record])
                result = self.run_verifier("--hmac-key", str(self.key_path))
                self.assertEqual(result.returncode, EXIT_VALID, result.stdout)


if __name__ == "__main__":
    unittest.main()
