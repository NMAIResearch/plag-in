"""End-to-end conformance: a store PLAG IN wrote must pass the reference verifier.

The unit suite for the verifier builds its fixtures from the specification
alone, so it cannot catch the two implementations drifting apart. This runs the
verifier as a subprocess against bytes `ReceiptStore` actually wrote, including
its authenticated checkpoint, and asserts the `core+anchor` level.
"""
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from dataclasses import fields
from pathlib import Path

from plag_in.conformance import check_conformance
from plag_in.errors import ReceiptPersistenceError
from plag_in.identity import RUNTIME_SOURCE_SCOPE, canonical_line
from plag_in.receipts import Receipt, ReceiptStore
from tests.support import build_gateway_stack

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


class SpecConformanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_verifier(self, *args):
        command = [sys.executable, str(TOOL), str(self.store.path), *args]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def test_emitted_store_passes_the_reference_verifier_at_core_plus_anchor(self):
        for index in range(3):
            self.store.append(make_receipt(f"req-{index}"))

        result = self.run_verifier(
            "--hmac-key", str(self.store.hmac_key_path),
            "--checkpoint", str(self.store.path) + ".checkpoint",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("result=valid", result.stdout)
        self.assertIn("records_read=3", result.stdout)
        self.assertIn("conformance_assessed=core+anchor", result.stdout)

    def test_emitted_lines_are_exactly_canonical_bytes(self):
        record = self.store.append(make_receipt("req-canonical"))
        self.assertEqual(self.store.path.read_bytes(), canonical_line(record))

    def test_non_ascii_alias_is_stored_raw_not_escaped(self):
        record = self.store.append(make_receipt("req-unicode", model_alias="modèle"))
        stored = self.store.path.read_bytes()
        self.assertIn("modèle".encode("utf-8"), stored)
        self.assertNotIn(b"\\u00e8", stored)  # spec section 4.4: raw, not escaped
        self.assertEqual(stored, canonical_line(record))

    def test_float_latency_is_rejected_rather_than_converted(self):
        with self.assertRaises(ReceiptPersistenceError) as raised:
            self.store.append(make_receipt("req-float", latency_us=12.5))
        self.assertIn("latency_us must be an integer or null", str(raised.exception))
        self.assertEqual(self.store.path.read_bytes(), b"")

    def test_float_in_an_implementation_block_is_converted_to_a_decimal_string(self):
        record = self.store.append(
            make_receipt("req-profile", requested_runtime_profile={"temperature": 0.0})
        )
        self.assertEqual(record["requested_runtime_profile"]["temperature"], "0.0")

        result = self.run_verifier("--hmac-key", str(self.store.hmac_key_path))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_tail_deletion_of_an_emitted_store_is_caught_by_its_own_checkpoint(self):
        self.store.append(make_receipt("req-0"))
        self.store.append(make_receipt("req-1"))
        lines = self.store.path.read_bytes().splitlines(keepends=True)
        self.store.path.write_bytes(lines[0])

        result = self.run_verifier(
            "--hmac-key", str(self.store.hmac_key_path),
            "--checkpoint", str(self.store.path) + ".checkpoint",
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("record count does not match the authenticated checkpoint", result.stdout)


class WorkerAdapterConformanceTests(unittest.TestCase):
    """R3-F2: a receipt the real worker path writes must pass the reference verifier.

    The store here is written by the gateway serving an actual request through
    `build_gateway_stack`, not by a fixture receipt assembled in this file. That
    is the distinction that let the defect through: every earlier conformance
    test constructed its own `Receipt`, and none of them carried the adapter
    record the worker setup path puts in `runtime_profile`.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stack = build_gateway_stack(Path(self.tmp.name), alias="fixture-alias")

    def tearDown(self):
        self.stack.stop()
        self.tmp.cleanup()

    def test_a_served_request_writes_a_conforming_receipt(self):
        body = json.dumps(
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.stack.server.base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            self.assertEqual(response.status, 200)

        store_path = self.stack.receipts_path
        command = [
            sys.executable, str(TOOL), str(store_path),
            "--hmac-key", str(store_path) + ".hmac_key",
            "--checkpoint", str(store_path) + ".checkpoint",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("result=valid", result.stdout)
        self.assertIn("records_read=1", result.stdout)

        record = json.loads(store_path.read_bytes().splitlines()[0])
        self.assertEqual(
            sorted(record["runtime_profile"]), ["effective", "measurement_status", "requested"]
        )
        self.assertEqual(record["plag_in_adapter_record"]["engine"], "llama_server_direct")

    def test_the_repaired_path_retains_no_prompt_or_response_content(self):
        """Coverage item 10, against a receipt the real gateway wrote.

        The prompt and the answer are distinctive strings, so their absence
        from the stored bytes is checked directly rather than inferred from the
        field-name scan. The name scan is checked too, because it is the rule
        the specification states; neither establishes that some other field is
        free of content-derived data, which stays a reviewer's judgement about
        this implementation.
        """
        prompt = "sphinx-of-black-quartz-judge-my-vow"
        body = json.dumps(
            {"model": "fixture-alias", "messages": [{"role": "user", "content": prompt}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.stack.server.base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            answer = json.loads(response.read().decode("utf-8"))
        answer_text = answer["choices"][0]["message"]["content"]

        stored = self.stack.receipts_path.read_bytes()
        self.assertNotIn(prompt.encode("utf-8"), stored)
        if answer_text:
            self.assertNotIn(answer_text.encode("utf-8"), stored)

        record = json.loads(stored.splitlines()[0])
        self.assertIs(record["content_retained"], False)

        result = check_conformance(
            self.stack.receipts_path,
            Path(str(self.stack.receipts_path) + ".hmac_key"),
            Path(str(self.stack.receipts_path) + ".checkpoint"),
        )
        self.assertTrue(result["conformance_valid"], result["errors"])
        self.assertTrue(
            any("content derivation" in item for item in result["unassessed_checks"]),
            result["unassessed_checks"],
        )


class LoneSurrogateTests(unittest.TestCase):
    """A5: an unencodable string is refused by name, not by a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def assert_refused_without_traceback(self, fragment, **overrides):
        before = self.store.path.read_bytes()
        try:
            self.store.append(make_receipt("req-surrogate", **overrides))
        except ReceiptPersistenceError as exc:
            self.assertIn(fragment, str(exc))
        except UnicodeEncodeError as exc:  # pragma: no cover - the defect this pins
            self.fail(f"encoder error escaped as a traceback: {exc}")
        else:
            self.fail("an unencodable receipt was accepted")
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_a_surrogate_in_a_declared_string_field_is_refused(self):
        self.assert_refused_without_traceback(
            "lone Unicode surrogate U+D800", model_alias="mod\ud800el"
        )

    def test_a_surrogate_in_an_implementation_block_value_is_refused(self):
        self.assert_refused_without_traceback(
            "lone Unicode surrogate U+DFFF",
            requested_runtime_profile={"template": "a\udfffb"},
        )

    def test_a_surrogate_in_an_object_key_is_refused(self):
        self.assert_refused_without_traceback(
            "in the object key",
            requested_runtime_profile={"ke\ud800y": "value"},
        )

    def test_a_surrogate_nested_in_a_list_is_refused(self):
        self.assert_refused_without_traceback(
            "lone Unicode surrogate U+D800",
            gpu_offload={"labels": ["fine", "bad\ud800"]},
        )

    def test_a_valid_astral_character_is_still_accepted(self):
        """A surrogate pair is a character; only an unpaired half is refused."""
        record = self.store.append(make_receipt("req-astral", model_alias="modèle-𝄞"))
        self.assertEqual(record["model_alias"], "modèle-𝄞")
        self.assertEqual(self.store.path.read_bytes(), canonical_line(record))


class DocumentedFloatRepresentationTests(unittest.TestCase):
    """A6: the emitter names the decimal representation it produces."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_finite_float_is_stored_as_the_documented_python_repr(self):
        for value in (0.7, 0.1 + 0.2, 1e-7, 2.0, -3.25):
            with self.subTest(value=value):
                record = self.store.append(
                    make_receipt(
                        f"req-float-{value!r}",
                        requested_runtime_profile={"temperature": value},
                    )
                )
                self.assertEqual(
                    record["requested_runtime_profile"]["temperature"], repr(value)
                )

    def test_a_typed_float_is_rejected_rather_than_represented(self):
        with self.assertRaises(ReceiptPersistenceError):
            self.store.append(make_receipt("req-typed-float", latency_us=1.5))

    def test_a_non_finite_float_is_rejected_in_an_implementation_block(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ReceiptPersistenceError) as raised:
                    self.store.append(
                        make_receipt(
                            "req-non-finite", requested_runtime_profile={"temperature": value}
                        )
                    )
                self.assertIn("non-finite float", str(raised.exception))


class EmittedLevelTests(unittest.TestCase):
    """A4: PLAG IN's own receipts are Core, and are not reported as Extended."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_emitted_store_is_assessed_at_core_not_extended(self):
        self.store.append(make_receipt("req-0"))
        result = check_conformance(
            self.store.path,
            self.store.hmac_key_path,
            self.store.path.with_name(self.store.path.name + ".checkpoint"),
        )
        self.assertTrue(result["conformance_valid"], result["errors"])
        # PLAG IN emits its identity fields at the top level rather than in the
        # section 5 `source_identity` and `engine_identity` blocks, so its
        # records are Core. The level says so instead of implying more.
        self.assertEqual(result["conformance_level"], "core+anchor")

    def test_the_receipt_declares_no_section_5_identity_blocks(self):
        """Why the level above is Core, stated as a fact about the dataclass.

        `Receipt` has no `source_identity` or `engine_identity` field, so no
        emitted record can reach Extended. The Extended rule itself, including
        the empty-requested case, is exercised against signed records in
        `tests/receipt_corpus.py`, which can build records this emitter cannot.
        """
        field_names = {field.name for field in fields(Receipt)}
        self.assertNotIn("source_identity", field_names)
        self.assertNotIn("engine_identity", field_names)

    def test_no_emitted_receipt_claims_measured_effective_operation(self):
        self.store.append(make_receipt("req-0"))
        result = check_conformance(
            self.store.path,
            self.store.hmac_key_path,
            self.store.path.with_name(self.store.path.name + ".checkpoint"),
        )
        self.assertFalse(result["effective_runtime_measured"])


class DeclaredFieldTypeTests(unittest.TestCase):
    """F4: a field the dataclass declares is validated, never converted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def assert_rejected(self, fragment, **overrides):
        before = self.store.path.read_bytes()
        request_id = overrides.pop("request_id", "req-typed")
        with self.assertRaises(ReceiptPersistenceError) as raised:
            self.store.append(make_receipt(request_id, **overrides))
        self.assertIn(fragment, str(raised.exception))
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_float_request_id_is_rejected(self):
        # Previously stored as the string '3.5' and accepted by the verifier.
        self.assert_rejected("request_id must be str", request_id=3.5)

    def test_float_gateway_version_is_rejected(self):
        self.assert_rejected("gateway_version must be str", gateway_version=2.5)

    def test_float_content_retained_is_rejected(self):
        self.assert_rejected("content_retained must be bool", content_retained=1.5)

    def test_boolean_in_an_integer_field_is_rejected(self):
        self.assert_rejected("input_tokens must be an integer or null", input_tokens=True)

    def test_string_in_a_declared_dict_field_is_rejected(self):
        self.assert_rejected(
            "gpu_offload must be dict", gpu_offload="offloaded"
        )

    def test_null_in_a_required_field_is_rejected(self):
        self.assert_rejected("model_alias must not be null", model_alias=None)

    def test_nan_in_an_implementation_block_is_rejected(self):
        self.assert_rejected(
            "non-finite float at requested_runtime_profile.temperature",
            requested_runtime_profile={"temperature": float("nan")},
        )

    def test_infinity_nested_in_a_list_is_rejected(self):
        self.assert_rejected(
            "non-finite float at gpu_offload.layers[1]",
            gpu_offload={"layers": [1.0, float("inf")]},
        )

    def test_non_finite_in_a_tuple_is_rejected(self):
        self.assert_rejected(
            "non-finite float at gpu_offload.layers[0]",
            gpu_offload={"layers": (float("-inf"),)},
        )

    def test_finite_floats_in_a_list_and_tuple_are_converted(self):
        record = self.store.append(
            make_receipt(
                "req-blocks",
                requested_runtime_profile={"list": [1.5, 2.0], "tuple": (0.25,)},
            )
        )
        self.assertEqual(record["requested_runtime_profile"]["list"], ["1.5", "2.0"])
        self.assertEqual(record["requested_runtime_profile"]["tuple"], ["0.25"])


class EmitterUniquenessTests(unittest.TestCase):
    """F6: the emitter must not create the duplicate the verifier rejects."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = ReceiptStore(
            self.tmp_path / "receipts.jsonl", hmac_key_path=self.tmp_path / "key"
        )
        self.checkpoint = self.store.path.with_name(self.store.path.name + ".checkpoint")

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_request_id_is_rejected_without_touching_the_store(self):
        self.store.append(make_receipt("req-dup"))
        store_before = self.store.path.read_bytes()
        checkpoint_before = self.checkpoint.read_bytes()

        with self.assertRaises(ReceiptPersistenceError) as raised:
            self.store.append(make_receipt("req-dup"))
        self.assertIn("request_id is already present in this store", str(raised.exception))

        self.assertEqual(self.store.path.read_bytes(), store_before)
        self.assertEqual(self.checkpoint.read_bytes(), checkpoint_before)
        valid, count = self.store.verify_chain()
        self.assertTrue(valid)
        self.assertEqual(count, 1)

    def test_a_distinct_identifier_still_appends_after_a_rejection(self):
        self.store.append(make_receipt("req-0"))
        with self.assertRaises(ReceiptPersistenceError):
            self.store.append(make_receipt("req-0"))
        self.store.append(make_receipt("req-1"))
        valid, count = self.store.verify_chain()
        self.assertTrue(valid)
        self.assertEqual(count, 2)

    def test_the_reference_verifier_agrees_the_store_is_clean(self):
        self.store.append(make_receipt("req-dup"))
        with self.assertRaises(ReceiptPersistenceError):
            self.store.append(make_receipt("req-dup"))

        command = [
            sys.executable,
            str(TOOL),
            str(self.store.path),
            "--hmac-key",
            str(self.store.hmac_key_path),
            "--checkpoint",
            str(self.checkpoint),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("records_read=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
