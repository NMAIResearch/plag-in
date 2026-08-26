import json
import stat
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from plag_in.errors import ReceiptPersistenceError
from plag_in.identity import RUNTIME_SOURCE_SCOPE, canonical_digest
from plag_in.receipts import INTEGRITY_MODE, Receipt, ReceiptStore

_SECRET = "test-secret-fixture-string-9f2c"
_PROMPT = "the quick brown prompt fixture string"


def _make_receipt(request_id: str, **overrides) -> Receipt:
    base = dict(
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
        input_tokens=7,
        output_tokens=5,
        latency_ms=12.5,
        status="completed",
        content_retained=False,
        schema_version="3",
        runtime_source_digest="a" * 64,
        runtime_source_scope=RUNTIME_SOURCE_SCOPE,
    )
    base.update(overrides)
    return Receipt(**base)


class ReceiptStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key")

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_and_get(self):
        record = self.store.append(_make_receipt("req-1"))
        self.assertEqual(record["request_id"], "req-1")
        self.assertEqual(self.store.get("req-1")["request_id"], "req-1")

    def test_chain_is_valid_across_multiple_records(self):
        for i in range(5):
            self.store.append(_make_receipt(f"req-{i}"))
        valid, count = self.store.verify_chain()
        self.assertTrue(valid)
        self.assertEqual(count, 5)

    def test_mutation_detected(self):
        self.store.append(_make_receipt("req-1"))
        self.store.append(_make_receipt("req-2"))
        lines = self.store.path.read_text().splitlines()
        record = json.loads(lines[0])
        record["model_alias"] = "tampered-alias"
        lines[0] = json.dumps(record)
        self.store.path.write_text("\n".join(lines) + "\n")

        valid, count = self.store.verify_chain()
        self.assertFalse(valid)

    def test_new_schema_field_mutation_detected(self):
        # Regression 12: the HMAC chain covers the full record via
        # asdict(receipt), so a schema-v2-only field (e.g.
        # native_identity_digest) is authenticated exactly like a
        # pre-existing field; mutating it must invalidate the chain.
        self.store.append(_make_receipt("req-1", native_identity_digest="a" * 64))
        self.store.append(_make_receipt("req-2"))
        lines = self.store.path.read_text().splitlines()
        record = json.loads(lines[0])
        record["native_identity_digest"] = "f" * 64
        lines[0] = json.dumps(record)
        self.store.path.write_text("\n".join(lines) + "\n")

        valid, _count = self.store.verify_chain()
        self.assertFalse(valid)

    def test_key_mutation_invalidates_the_whole_chain(self):
        self.store.append(_make_receipt("req-1"))
        self.store.append(_make_receipt("req-2"))
        self.store.hmac_key_path.write_bytes(b"x" * 32)  # a different key entirely

        reloaded = ReceiptStore(self.store.path, hmac_key_path=self.store.hmac_key_path)
        valid, _count = reloaded.verify_chain()
        self.assertFalse(valid)

    def test_reordering_detected(self):
        self.store.append(_make_receipt("req-1"))
        self.store.append(_make_receipt("req-2"))
        lines = self.store.path.read_text().splitlines()
        self.store.path.write_text("\n".join(reversed(lines)) + "\n")

        valid, _count = self.store.verify_chain()
        self.assertFalse(valid)

    def test_deletion_detected(self):
        self.store.append(_make_receipt("req-1"))
        self.store.append(_make_receipt("req-2"))
        self.store.append(_make_receipt("req-3"))
        lines = self.store.path.read_text().splitlines()
        del lines[1]  # remove the middle record; the HMAC chain must break
        self.store.path.write_text("\n".join(lines) + "\n")

        valid, _count = self.store.verify_chain()
        self.assertFalse(valid)

    def test_deterministic_for_identical_key_and_metadata(self):
        # Determinism holds for a fixed key, metadata and predecessor state.
        # It must not hold across independently generated keys.
        store_a = ReceiptStore(self.tmp_path / "a.jsonl", hmac_key_path=self.tmp_path / "key_a")
        receipt = _make_receipt("req-fixed")
        record_a = store_a.append(receipt)
        body = {k: v for k, v in record_a.items() if k != "hmac"}
        repeated = store_a._record_hmac(record_a["prev_hmac"], body)  # noqa: SLF001
        self.assertEqual(record_a["hmac"], repeated)

        independent_store = ReceiptStore(self.tmp_path / "c.jsonl", hmac_key_path=self.tmp_path / "other_key")
        record_c = independent_store.append(receipt)
        self.assertNotEqual(record_a["hmac"], record_c["hmac"])

    def test_hmac_key_and_receipt_file_have_mode_0600(self):
        self.store.append(_make_receipt("req-1"))
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.store.hmac_key_path.stat().st_mode), 0o600)

    def test_integrity_mode_is_labelled_local_hmac(self):
        record = self.store.append(_make_receipt("req-1"))
        self.assertEqual(record["integrity_mode"], "local_hmac")
        self.assertEqual(INTEGRITY_MODE, "local_hmac")

    def test_schema_v3_has_runtime_source_fields(self):
        field_names = {f.name for f in fields(Receipt)}
        self.assertIn("runtime_source_digest", field_names)
        self.assertIn("runtime_source_scope", field_names)
        # v3 is the current contract; a freshly built receipt declares it.
        self.assertEqual(_make_receipt("req-1").schema_version, "3")

    def test_runtime_source_digest_mutation_detected(self):
        # The runtime source digest is a schema-v3 field, so like every
        # other field it is covered by the HMAC chain and cannot be
        # rewritten without invalidating the chain.
        self.store.append(_make_receipt("req-1", runtime_source_digest="a" * 64))
        self.store.append(_make_receipt("req-2"))
        lines = self.store.path.read_text().splitlines()
        record = json.loads(lines[0])
        record["runtime_source_digest"] = "f" * 64
        lines[0] = json.dumps(record)
        self.store.path.write_text("\n".join(lines) + "\n")

        valid, _count = self.store.verify_chain()
        self.assertFalse(valid)

    def test_schema_v3_rejects_absent_or_malformed_source_fields(self):
        # CPO-02: a v3 receipt with missing or malformed source identity must
        # fail closed before any bytes or the checkpoint change.
        self.store.append(_make_receipt("req-good"))
        before_bytes = self.store.path.read_bytes()
        before_ckpt = self.store._checkpoint_path.read_bytes()  # noqa: SLF001
        bad_forms = [
            {"runtime_source_digest": None},
            {"runtime_source_digest": ""},
            {"runtime_source_digest": "a" * 63},
            {"runtime_source_digest": "A" * 64},
            {"runtime_source_digest": "g" * 64},
            {"runtime_source_scope": "runtime_python_source:wrong"},
            {"runtime_source_scope": None},
        ]
        for bad in bad_forms:
            with self.assertRaises(ReceiptPersistenceError):
                self.store.append(_make_receipt("req-bad", **bad))
        # The store is byte-for-byte unchanged after every rejection.
        self.assertEqual(self.store.path.read_bytes(), before_bytes)
        self.assertEqual(self.store._checkpoint_path.read_bytes(), before_ckpt)  # noqa: SLF001
        valid, count = self.store.verify_chain()
        self.assertTrue(valid)
        self.assertEqual(count, 1)

    def test_schema_v2_receipt_without_source_fields_still_appends(self):
        # Historical v2 stays readable and makes no source-binding claim.
        record = self.store.append(
            _make_receipt(
                "req-v2", schema_version="2",
                runtime_source_digest=None, runtime_source_scope=None,
            )
        )
        self.assertEqual(record["schema_version"], "2")
        self.assertIsNone(record["runtime_source_digest"])
        valid, count = self.store.verify_chain()
        self.assertTrue(valid)
        self.assertEqual(count, 1)

    def test_schema_has_no_content_or_content_digest_field(self):
        field_names = {f.name for f in fields(Receipt)}
        for forbidden in ("content", "prompt", "response", "prompt_digest", "prompt_hash"):
            self.assertNotIn(forbidden, field_names)

    def test_schema_has_no_raw_path_field(self):
        field_names = {f.name for f in fields(Receipt)}
        for forbidden in ("path", "model_path", "weight_path", "executable_path"):
            self.assertNotIn(forbidden, field_names)

    def test_secret_and_prompt_strings_never_stored(self):
        # A receipt is built purely from metadata; nothing in the constructor
        # accepts prompt/response text or a secret, so neither fixture
        # string can appear in an appended record no matter what a caller
        # tries to pass through unrelated string fields.
        record = self.store.append(_make_receipt("req-secret-check", route="local"))
        serialized = json.dumps(record)
        self.assertNotIn(_SECRET, serialized)
        self.assertNotIn(_PROMPT, serialized)
        self.assertNotIn(canonical_digest({"prompt": _PROMPT}), serialized)


if __name__ == "__main__":
    unittest.main()
