"""Generated agreement corpus for the Inference Receipt Specification.

Every mandatory rule in the specification gets at least one positive or
negative case here, with the exact validity decision, conformance level and
named diagnostic each implementation must produce. The product conformance
path (`plag_in.conformance`) and the reference verifier
(`tools/verify_inference_receipts.py`) are separate implementations of the same
contract, so the corpus is what establishes that they agree; eight hand-written
mutations did not reach most of these rules.

Diagnostics are recorded per implementation. The two are required to reach the
same decision and report the same level, not to phrase a refusal in the same
words: identical wording would be evidence that one was copied from the other,
which is the coupling the two-implementation design exists to avoid.

Records are signed here rather than emitted by `ReceiptStore`, because most of
these stores are ones a conforming emitter refuses to write. Signing them keeps
the chain authentic so that the rule under test is the only thing failing.
"""
from __future__ import annotations

import hashlib
import hmac as hmac_module
import json
from dataclasses import dataclass
from pathlib import Path

CORPUS_KEY = b"\x11" * 32
GENESIS = "0" * 64
VALID_TIMESTAMP = "2026-08-26T00:00:00+00:00"


def canonical(record: dict) -> bytes:
    """The canonical serialisation of section 4, written out here in full.

    Deliberately not imported from either implementation under test: a fixture
    that borrows the encoder it is meant to pin would agree with a defect in it.
    """
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def core_record(request_id: str = "req-0", **overrides) -> dict:
    """A minimal record carrying exactly the mandatory core and chain fields."""
    record = {
        "spec_version": "0.1",
        "request_id": request_id,
        "timestamp": VALID_TIMESTAMP,
        "weight_digest": "a" * 64,
        "content_retained": False,
        "status": "completed",
        "input_tokens": 7,
        "output_tokens": 5,
        "integrity_mode": "local_hmac",
    }
    record.update(overrides)
    return record


def extended_blocks(requested: dict | None = None, effective: dict | None = None,
                    measurement: dict | None = None) -> dict:
    """The three blocks section 8 requires at the Extended level."""
    return {
        "source_identity": {"digest": "b" * 64, "scope": "runtime_python_source:src/**/*.py"},
        "engine_identity": {
            "engine": "libllama_embedded",
            "version": "0.1.0a2",
            "native_identity_digest": None,
            "component_digests": None,
            "upstream_identity": None,
        },
        "runtime_profile": {
            "requested": {"context_tokens": 4096} if requested is None else requested,
            "effective": {} if effective is None else effective,
            "measurement_status": {} if measurement is None else measurement,
        },
    }


def sign_chain(records: list[dict], key: bytes = CORPUS_KEY) -> list[dict]:
    """Link and authenticate each record against its predecessor."""
    signed: list[dict] = []
    previous = GENESIS
    for record in records:
        body = {k: v for k, v in record.items() if k != "hmac"}
        body["prev_hmac"] = previous
        digest = hashlib.sha256(canonical(body)).hexdigest()
        message = f"{previous}:{digest}".encode("utf-8")
        body["hmac"] = hmac_module.new(key, message, hashlib.sha256).hexdigest()
        signed.append(body)
        previous = body["hmac"]
    return signed


def store_bytes(records: list[dict]) -> bytes:
    return b"".join(canonical(record) + b"\n" for record in records)


def checkpoint_bytes(count: int, last_hmac: str, key: bytes = CORPUS_KEY) -> bytes:
    """An anchor that authenticates the count and final HMAC it declares."""
    message = f"checkpoint:{count}:{last_hmac}".encode("utf-8")
    anchor = {
        "count": count,
        "last_hmac": last_hmac,
        "checkpoint_hmac": hmac_module.new(key, message, hashlib.sha256).hexdigest(),
    }
    return json.dumps(anchor, sort_keys=True).encode("utf-8")


def anchor_for(records: list[dict], key: bytes = CORPUS_KEY) -> bytes:
    last = records[-1]["hmac"] if records else GENESIS
    return checkpoint_bytes(len(records), last, key)


@dataclass
class CorpusCase:
    """One store, the run it is verified under, and the required outcome."""

    name: str
    rule: str
    store: bytes
    expect_valid: bool
    expect_level: str
    expect_exit: int
    checkpoint: bytes | None = None
    supply_checkpoint_path: bool = True
    supply_key: bool = True
    key: bytes = CORPUS_KEY
    reference_diagnostic: str | None = None
    product_diagnostic: str | None = None
    notes: str = ""
    expect_measured_runtime: bool = False

    def materialise(self, directory: Path) -> dict:
        """Write the case into `directory` and return the paths to verify."""
        base = Path(directory) / self.name
        base.mkdir(parents=True, exist_ok=True)
        store_path = base / "receipts.jsonl"
        store_path.write_bytes(self.store)
        key_path = base / "receipts.jsonl.hmac_key"
        key_path.write_bytes(self.key)
        checkpoint_path = base / "receipts.jsonl.checkpoint"
        if self.checkpoint is not None:
            checkpoint_path.write_bytes(self.checkpoint)
        return {
            "store": store_path,
            "key": key_path if self.supply_key else None,
            "checkpoint": checkpoint_path if self.supply_checkpoint_path else None,
        }


def _anchored(records: list[dict], **kwargs) -> dict:
    return {"store": store_bytes(records), "checkpoint": anchor_for(records), **kwargs}


def _unanchored(records: list[dict], **kwargs) -> dict:
    return {
        "store": store_bytes(records),
        "checkpoint": None,
        "supply_checkpoint_path": False,
        **kwargs,
    }


def _bad_record(rule: str, name: str, reference: str, product: str, **overrides) -> CorpusCase:
    """One signed record carrying exactly one defect, verified without an anchor."""
    records = sign_chain([core_record(**overrides)])
    return CorpusCase(
        name=name,
        rule=rule,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic=reference,
        product_diagnostic=product,
        **_unanchored(records),
    )


def build_corpus() -> list[CorpusCase]:  # noqa: PLR0915 - one flat case table
    """Every case, one per mandatory rule or per positive level."""
    cases: list[CorpusCase] = []

    # -- positive cases, one per reportable conformance level ---------------

    core = sign_chain([core_record("req-core")])
    cases.append(CorpusCase(
        name="valid_core",
        rule="section 8: Core",
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        notes="mandatory core plus chain linkage, verified without an anchor",
        **_unanchored(core),
    ))
    cases.append(CorpusCase(
        name="valid_core_anchor",
        rule="section 8: Core+Anchor",
        expect_valid=True,
        expect_level="core+anchor",
        expect_exit=0,
        **_anchored(core),
    ))

    extended = sign_chain([core_record("req-extended", **extended_blocks())])
    cases.append(CorpusCase(
        name="valid_extended",
        rule="section 8: Extended",
        expect_valid=True,
        expect_level="extended",
        expect_exit=0,
        **_unanchored(extended),
    ))
    cases.append(CorpusCase(
        name="valid_extended_anchor",
        rule="section 8: Extended+Anchor",
        expect_valid=True,
        expect_level="extended+anchor",
        expect_exit=0,
        **_anchored(extended),
    ))

    measured = sign_chain([core_record(
        "req-measured",
        **extended_blocks(
            effective={"context_tokens": 4096},
            measurement={"context_tokens": "measured"},
        ),
    )])
    cases.append(CorpusCase(
        name="valid_extended_with_measured_effective_field",
        rule="A4: effective runtime operation may be claimed only when measured",
        expect_valid=True,
        expect_level="extended",
        expect_exit=0,
        expect_measured_runtime=True,
        **_unanchored(measured),
    ))

    unassessed_only = sign_chain([core_record(
        "req-unassessed",
        **extended_blocks(
            effective={"gpu_layers": 20},
            measurement={"gpu_layers": "unassessed"},
        ),
    )])
    cases.append(CorpusCase(
        name="extended_without_any_measured_field",
        rule="A4: an unassessed effective field is not measured operation",
        expect_valid=True,
        expect_level="extended",
        expect_exit=0,
        expect_measured_runtime=False,
        **_unanchored(unassessed_only),
    ))

    empty_requested = sign_chain([core_record(
        "req-empty-requested", **extended_blocks(requested={})
    )])
    cases.append(CorpusCase(
        name="empty_requested_profile_is_not_extended",
        rule="A4: an empty requested profile cannot qualify as Extended",
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        notes="valid, but capped at Core: emptying the block is not runtime binding",
        **_unanchored(empty_requested),
    ))

    orphan_measured = sign_chain([core_record(
        "req-orphan-measured",
        **extended_blocks(effective={}, measurement={"gpu_layers": "measured"}),
    )])
    cases.append(CorpusCase(
        name="measured_status_without_an_effective_value",
        rule="section 5: a measured status names a key the effective block carries",
        expect_valid=False,
        expect_level="extended",
        expect_exit=1,
        expect_measured_runtime=False,
        reference_diagnostic="claims measured with no",
        product_diagnostic="is measured but",
        notes=(
            "the record both implementations reported as establishing measured "
            "effective operation while carrying no effective value at all "
            "(independent review of work package A, D3)"
        ),
        **_unanchored(orphan_measured),
    ))

    vacant_identity = sign_chain([core_record(
        "req-vacant-identity",
        source_identity={"digest": "0" * 64, "scope": "x"},
        engine_identity={"engine": "", "version": ""},
        runtime_profile={"requested": {"": None}, "effective": {}, "measurement_status": {}},
    )])
    cases.append(CorpusCase(
        name="vacant_extended_identity_is_not_extended",
        rule="section 8: Extended requires a named engine, a version and a real setting",
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        notes=(
            "every Extended block is present and technically non-empty, and the "
            "record binds no engine name, no version and no requested setting. "
            "Valid, and capped at Core (D4)"
        ),
        **_unanchored(vacant_identity),
    ))

    vacant_requested = sign_chain([core_record(
        "req-vacant-requested",
        **extended_blocks(requested={"context_tokens": None, "gpu_layers": ""}),
    )])
    cases.append(CorpusCase(
        name="requested_without_a_meaningful_value_is_not_extended",
        rule="section 8: a named requested key carrying no value is not runtime binding",
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        notes="the identity blocks are sound; nothing was requested (D4)",
        **_unanchored(vacant_requested),
    ))

    partial_extended = sign_chain([
        core_record("req-extended-1", **extended_blocks()),
        core_record("req-plain-2"),
    ])
    cases.append(CorpusCase(
        name="one_record_short_of_extended_caps_the_store",
        rule="section 8: the level is bounded by what every record carries",
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        **_unanchored(partial_extended),
    ))

    # -- timestamps ---------------------------------------------------------

    timestamp_reference = "timestamp is not RFC 3339"
    timestamp_product = "timestamp is not a real RFC 3339 instant"
    for name, value in (
        ("invalid_calendar_timestamp", "2026-02-30T00:00:00+00:00"),
        ("impossible_timestamp_fields", "2026-99-99T99:99:99+99:99"),
        ("end_of_day_hour_24", "2026-08-26T24:00:00+00:00"),
        ("leap_second_that_is_not_an_event", "2026-01-01T12:34:60Z"),
        ("leap_second_at_a_real_leap_month", "2016-12-31T23:59:60Z"),
        ("timestamp_without_offset", "2026-08-26T00:00:00"),
        ("timestamp_is_not_a_string", 20260826),
    ):
        cases.append(_bad_record(
            "section 2: timestamp must be a real RFC 3339 instant",
            name, timestamp_reference, timestamp_product, timestamp=value,
        ))

    # -- spec version -------------------------------------------------------

    cases.append(_bad_record(
        "section 2: spec_version names the contract the record claims",
        "wrong_spec_version",
        "spec_version is '0.2', expected '0.1'",
        "spec_version is '0.2', expected '0.1'",
        spec_version="0.2",
    ))
    missing_version = core_record("req-no-spec-version")
    del missing_version["spec_version"]
    cases.append(CorpusCase(
        name="missing_spec_version",
        rule="section 2: spec_version names the contract the record claims",
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="spec_version is None, expected '0.1'",
        product_diagnostic="spec_version is None, expected '0.1'",
        **_unanchored(sign_chain([missing_version])),
    ))

    # -- mandatory core fields ---------------------------------------------

    for field_name in ("request_id", "weight_digest", "content_retained", "status",
                       "input_tokens", "output_tokens"):
        absent = core_record(f"req-absent-{field_name}")
        del absent[field_name]
        cases.append(CorpusCase(
            name=f"absent_core_field_{field_name}",
            rule="section 2: a record missing a mandatory core field is not a receipt",
            expect_valid=False,
            expect_level="core",
            expect_exit=1,
            reference_diagnostic=f"mandatory core field absent: {field_name}",
            product_diagnostic=f"mandatory core field absent: {field_name}",
            **_unanchored(sign_chain([absent])),
        ))

    cases.append(_bad_record(
        "section 2: weight_digest is 64 lowercase hex characters",
        "uppercase_weight_digest",
        "weight_digest must be 64 lowercase hex",
        "weight_digest must be 64 lowercase hex",
        weight_digest="A" * 64,
    ))
    cases.append(_bad_record(
        "section 2: weight_digest is 64 lowercase hex characters",
        "short_weight_digest",
        "weight_digest must be 64 lowercase hex",
        "weight_digest must be 64 lowercase hex",
        weight_digest="a" * 63,
    ))
    cases.append(_bad_record(
        "section 2: status is one of three terminal values",
        "unknown_status",
        "is not one of",
        "is not one of",
        status="in_progress",
    ))
    cases.append(_bad_record(
        "section 2 and 6: content_retained is a boolean",
        "non_boolean_content_retained",
        "content_retained must be a boolean",
        "content_retained must be a boolean",
        content_retained="false",
    ))
    cases.append(_bad_record(
        "section 2: a declared integer field is validated, not converted",
        "boolean_token_count",
        "input_tokens must not be a boolean",
        "input_tokens must not be a boolean",
        input_tokens=True,
    ))
    cases.append(_bad_record(
        "section 4.2: a declared integer field must reject a float",
        "float_in_a_typed_field",
        "input_tokens must be an integer or null",
        "input_tokens must be an integer or null",
        input_tokens=7.5,
    ))

    # -- floats -------------------------------------------------------------

    cases.append(_bad_record(
        "section 4: floats are prohibited anywhere in a record",
        "float_in_an_implementation_defined_block",
        "prohibited float at",
        "prohibited float at",
        plag_in_adapter_record={"sampling": {"temperature": 0.7}},
    ))
    cases.append(_bad_record(
        "section 4.2: non-finite values are prohibited, not converted",
        "non_finite_float_in_an_implementation_defined_block",
        "prohibited float at",
        "prohibited float at",
        plag_in_adapter_record={"sampling": {"temperature": float("nan")}},
    ))
    cases.append(_bad_record(
        "section 4.2: non-finite values are prohibited at every depth",
        "infinity_nested_in_an_array",
        "prohibited float at",
        "prohibited float at",
        plag_in_adapter_record={"layers": [1, float("inf")]},
    ))

    # -- prohibited content -------------------------------------------------

    cases.append(_bad_record(
        "section 6: the prohibition applies at every depth",
        "nested_prohibited_prompt",
        "prohibited content field present",
        "prohibited content field present",
        plag_in_adapter_record={"trace": {"inner": {"prompt": "hello"}}},
    ))
    cases.append(_bad_record(
        "section 6: the prohibition applies at every depth",
        "nested_prohibited_response_digest",
        "prohibited content field present",
        "prohibited content field present",
        plag_in_adapter_record={"trace": {"response_digest": "c" * 64}},
    ))
    cases.append(_bad_record(
        "section 6: the prohibition applies inside arrays",
        "prohibited_content_inside_an_array",
        "prohibited content field present",
        "prohibited content field present",
        plag_in_adapter_record={"turns": [{"content": "hello"}]},
    ))
    cases.append(_bad_record(
        "section 6: the prohibition applies at the top level",
        "top_level_prohibited_messages",
        "prohibited content field present",
        "prohibited content field present",
        messages=[{"role": "user"}],
    ))

    # -- optional blocks ----------------------------------------------------

    cases.append(_bad_record(
        "section 5: source_identity is validated per field",
        "malformed_source_identity_digest",
        "source_identity.digest must be 64 lowercase hex",
        "source_identity.digest must be 64 lowercase hex",
        source_identity={"digest": "short", "scope": "runtime_python_source:src/**/*.py"},
    ))
    cases.append(_bad_record(
        "section 5: source_identity.scope is mandatory alongside the digest",
        "empty_source_identity_scope",
        "source_identity.scope must be a non-empty string",
        "source_identity.scope must be a non-empty string",
        source_identity={"digest": "b" * 64, "scope": ""},
    ))
    cases.append(_bad_record(
        "section 5: engine and version are mandatory when engine_identity is present",
        "engine_identity_without_engine",
        "engine_identity.engine must be a string",
        "engine_identity.engine must be a string",
        engine_identity={"version": "0.1.0a2"},
    ))
    cases.append(_bad_record(
        "section 5: component_digests values are 64 lowercase hex",
        "malformed_component_digest",
        "component_digests.ggml must be 64 lowercase hex",
        "component_digests.ggml must be 64 lowercase hex",
        engine_identity={
            "engine": "libllama_embedded",
            "version": "0.1.0a2",
            "component_digests": {"ggml": "not-a-digest"},
        },
    ))
    cases.append(_bad_record(
        "section 5: upstream_identity is a string or null",
        "non_string_upstream_identity",
        "engine_identity.upstream_identity must be a string or null",
        "engine_identity.upstream_identity must be a string or null",
        engine_identity={"engine": "e", "version": "v", "upstream_identity": 7},
    ))
    cases.append(_bad_record(
        "section 5: runtime_profile carries three objects",
        "malformed_runtime_profile_requested",
        "runtime_profile.requested must be an object",
        "runtime_profile.requested must be an object",
        runtime_profile={"requested": "context=4096", "effective": {}, "measurement_status": {}},
    ))
    cases.append(_bad_record(
        "section 5: runtime_profile carries three objects",
        "empty_runtime_profile_object",
        "runtime_profile.requested must be an object",
        "runtime_profile.requested must be an object",
        runtime_profile={},
    ))
    cases.append(_bad_record(
        "section 5: runtime_profile must be an object",
        "runtime_profile_is_not_an_object",
        "runtime_profile must be an object",
        "runtime_profile must be an object",
        runtime_profile=[],
    ))
    cases.append(_bad_record(
        "section 5: every effective key needs a measurement_status",
        "effective_key_without_measurement_status",
        "runtime_profile.effective.gpu_layers has no measurement_status",
        "runtime_profile.effective.gpu_layers has no measurement_status",
        runtime_profile={
            "requested": {"gpu_layers": 20},
            "effective": {"gpu_layers": 20},
            "measurement_status": {},
        },
    ))
    cases.append(_bad_record(
        "section 5: measurement_status values are measured or unassessed",
        "invalid_measurement_status_value",
        "must be one of",
        "must be one of",
        runtime_profile={
            "requested": {"gpu_layers": 20},
            "effective": {"gpu_layers": 20},
            "measurement_status": {"gpu_layers": "probably"},
        },
    ))
    cases.append(_bad_record(
        "section 5: locality carries three non-empty strings",
        "empty_locality_field",
        "locality.evidence_class must be a non-empty string",
        "locality.evidence_class must be a non-empty string",
        locality={"level": "L1", "evidence_class": "", "listen_address": "127.0.0.1:8080"},
    ))

    # -- chain shape --------------------------------------------------------

    cases.append(_bad_record(
        "section 3: integrity_mode is local_hmac in this version",
        "unsupported_integrity_mode",
        "unsupported integrity_mode",
        "unsupported integrity_mode",
        integrity_mode="external_signature",
    ))

    broken_link = sign_chain([core_record("req-a"), core_record("req-b")])
    broken_link[1]["prev_hmac"] = "f" * 64
    cases.append(CorpusCase(
        name="broken_chain_linkage",
        rule="section 3: each record links to its predecessor",
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="chain linkage broken",
        product_diagnostic="chain linkage broken",
        **_unanchored(broken_link),
    ))

    forged_body = sign_chain([core_record("req-forged")])
    forged_body[0]["status"] = "refused"
    cases.append(CorpusCase(
        name="record_edited_after_signing",
        rule="section 3: the HMAC authenticates the record body",
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="HMAC does not authenticate this record",
        product_diagnostic="HMAC does not authenticate this record",
        **_unanchored(forged_body),
    ))

    # -- duplicate identifiers ---------------------------------------------

    duplicates = sign_chain([core_record("req-same"), core_record("req-same")])
    cases.append(CorpusCase(
        name="duplicate_request_id",
        rule="section 2: request_id is unique within the emitting store",
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="duplicate request_id",
        product_diagnostic="duplicate request_id",
        notes="the chain over two records sharing one identifier verifies perfectly well",
        **_unanchored(duplicates),
    ))

    # -- store bytes --------------------------------------------------------

    clean = store_bytes(core)
    linked_pair = sign_chain([core_record("req-pair-1"), core_record("req-pair-2")])
    for name, mutated in (
        ("leading_blank_line", b"\n" + clean),
        # A correctly linked pair with one blank line between them, so the
        # blank line is the only rule the store breaks.
        ("interior_blank_line",
         store_bytes(linked_pair[:1]) + b"\n" + store_bytes(linked_pair[1:])),
        ("trailing_blank_line", clean + b"\n"),
        ("whitespace_only_line", clean + b"   \n"),
    ):
        cases.append(CorpusCase(
            name=name,
            rule="section 4: a store holds one canonical record per line",
            store=mutated,
            checkpoint=None,
            supply_checkpoint_path=False,
            expect_valid=False,
            expect_level="core",
            expect_exit=1,
            reference_diagnostic="blank or whitespace-only line",
            product_diagnostic="blank or whitespace-only line",
        ))

    cases.append(CorpusCase(
        name="final_record_not_lf_terminated",
        rule="section 4: the store is one record per line, LF terminated",
        store=clean.rstrip(b"\n"),
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="final record is not LF terminated",
        product_diagnostic="final record is not LF terminated",
    ))

    cases.append(CorpusCase(
        name="non_canonical_stored_bytes",
        rule="section 4: the stored bytes are normative, not only the hashed bytes",
        store=json.dumps(core[0], sort_keys=True).encode("utf-8") + b"\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="stored bytes are not canonical",
        product_diagnostic="stored bytes are not canonical",
        notes="every HMAC in this store verifies; only the rendering differs",
    ))

    cases.append(CorpusCase(
        name="unsorted_keys_in_stored_bytes",
        rule="section 4: keys are sorted by Unicode code point",
        # Reverse-sorted rather than merely unsorted, so the case cannot pass by
        # accident on a record whose insertion order happens to be sorted.
        store=json.dumps(
            {key: core[0][key] for key in sorted(core[0], reverse=True)},
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8") + b"\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="stored bytes are not canonical",
        product_diagnostic="stored bytes are not canonical",
    ))

    cases.append(CorpusCase(
        name="escaped_non_ascii_in_stored_bytes",
        rule="section 4: non-ASCII is emitted raw, not \\uXXXX escaped",
        store=json.dumps(
            sign_chain([core_record("req-modèle")])[0],
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8") + b"\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="stored bytes are not canonical",
        product_diagnostic="stored bytes are not canonical",
    ))

    cases.append(CorpusCase(
        name="crlf_terminated_line",
        rule="section 4: newline translation must not alter the stored bytes",
        store=clean.replace(b"\n", b"\r\n"),
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="stored line contains CR",
        product_diagnostic="stored line contains CR",
    ))

    cases.append(CorpusCase(
        name="empty_store",
        rule="section 7: an empty store is not reported as valid",
        store=b"",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="store contains no records",
        product_diagnostic="store contains no records",
    ))

    cases.append(CorpusCase(
        name="line_is_not_a_json_object",
        rule="section 4: every line is exactly one canonical record",
        store=b"[1,2,3]\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="record is not a JSON object",
        product_diagnostic="record is not a JSON object",
    ))

    # -- lone surrogates ----------------------------------------------------
    #
    # A record carrying an escaped lone surrogate parses, and then cannot be
    # encoded as UTF-8. It is signed with a placeholder HMAC because no
    # conforming emitter can compute a real one over bytes that do not exist.

    surrogate_value = core_record("req-surrogate-value", note="lead \ud800 surrogate")
    surrogate_value["prev_hmac"] = GENESIS
    surrogate_value["hmac"] = "e" * 64
    cases.append(CorpusCase(
        name="escaped_lone_surrogate_in_a_value",
        rule="A5: a value that cannot be encoded as UTF-8 is refused by name",
        store=json.dumps(surrogate_value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="lone Unicode surrogate U+D800",
        product_diagnostic="lone Unicode surrogate U+D800",
        notes="must fail without a traceback in both implementations",
    ))

    surrogate_key = core_record("req-surrogate-key")
    surrogate_key["bad\udfffkey"] = "value"
    surrogate_key["prev_hmac"] = GENESIS
    surrogate_key["hmac"] = "e" * 64
    cases.append(CorpusCase(
        name="escaped_lone_surrogate_in_a_key",
        rule="A5: an object key that cannot be encoded as UTF-8 is refused by name",
        store=json.dumps(surrogate_key, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n",
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=1,
        reference_diagnostic="lone Unicode surrogate U+DFFF in the object key",
        product_diagnostic="lone Unicode surrogate U+DFFF in the object key",
    ))

    # -- anchors ------------------------------------------------------------

    cases.append(CorpusCase(
        name="missing_checkpoint_file",
        rule="section 3.2: a supplied anchor must be readable",
        store=clean,
        checkpoint=None,
        supply_checkpoint_path=True,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="cannot read checkpoint",
        product_diagnostic="checkpoint cannot be read",
    ))

    cases.append(CorpusCase(
        name="malformed_checkpoint_json",
        rule="section 3.2: a supplied anchor must be a JSON object",
        store=clean,
        checkpoint=b"{not json",
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="checkpoint is not valid JSON",
        product_diagnostic="checkpoint is not valid JSON",
    ))

    boolean_count_message = f"checkpoint:{True}:{core[-1]['hmac']}".encode("utf-8")
    boolean_anchor = json.dumps(
        {
            "count": True,
            "last_hmac": core[-1]["hmac"],
            "checkpoint_hmac": hmac_module.new(
                CORPUS_KEY, boolean_count_message, hashlib.sha256
            ).hexdigest(),
        },
        sort_keys=True,
    ).encode("utf-8")
    cases.append(CorpusCase(
        name="boolean_checkpoint_count",
        rule="section 3.2: validate the anchor's field shapes before authenticating it",
        store=clean,
        checkpoint=boolean_anchor,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="checkpoint count must be an integer",
        product_diagnostic="checkpoint count must be an integer",
        notes="authenticates under the store key and compares equal to a one-record store",
    ))

    negative_anchor = checkpoint_bytes(-1, core[-1]["hmac"])
    cases.append(CorpusCase(
        name="negative_checkpoint_count",
        rule="section 3.2: count is an integer of at least zero",
        store=clean,
        checkpoint=negative_anchor,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="checkpoint count must be zero or greater",
        product_diagnostic="checkpoint count must be zero or greater",
    ))

    short_hmac_anchor = json.dumps(
        {"count": 1, "last_hmac": "abc", "checkpoint_hmac": "d" * 64}, sort_keys=True
    ).encode("utf-8")
    cases.append(CorpusCase(
        name="malformed_checkpoint_last_hmac",
        rule="section 3.2: last_hmac is 64 lowercase hex characters",
        store=clean,
        checkpoint=short_hmac_anchor,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="checkpoint last_hmac must be 64 lowercase hex",
        product_diagnostic="checkpoint last_hmac must be 64 lowercase hex",
    ))

    unauthenticated_anchor = json.dumps(
        {"count": 1, "last_hmac": core[-1]["hmac"], "checkpoint_hmac": "0" * 64},
        sort_keys=True,
    ).encode("utf-8")
    cases.append(CorpusCase(
        name="checkpoint_hmac_does_not_authenticate",
        rule="section 3.2: the anchor is authenticated under the chain key",
        store=clean,
        checkpoint=unauthenticated_anchor,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="does not authenticate the checkpoint",
        product_diagnostic="does not authenticate the checkpoint",
    ))

    cases.append(CorpusCase(
        name="checkpoint_count_mismatch",
        rule="section 3.2: the anchor binds the expected record count",
        store=clean,
        checkpoint=checkpoint_bytes(5, core[-1]["hmac"]),
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="record count does not match the authenticated checkpoint",
        product_diagnostic="record count does not match the authenticated checkpoint",
    ))

    two_records = sign_chain([core_record("req-1"), core_record("req-2")])
    two_record_anchor = anchor_for(two_records)
    truncated = store_bytes(two_records[:1])
    cases.append(CorpusCase(
        name="tail_deletion_with_an_anchor",
        rule="section 3.2: the anchor is what detects tail deletion",
        store=truncated,
        checkpoint=two_record_anchor,
        expect_valid=False,
        expect_level="core+anchor",
        expect_exit=1,
        reference_diagnostic="record count does not match the authenticated checkpoint",
        product_diagnostic="record count does not match the authenticated checkpoint",
    ))
    cases.append(CorpusCase(
        name="tail_deletion_without_an_anchor",
        rule="section 3.1: the chain alone cannot detect deletion of the final records",
        store=truncated,
        checkpoint=None,
        supply_checkpoint_path=False,
        expect_valid=True,
        expect_level="core",
        expect_exit=0,
        notes=(
            "the same truncated store as the case above. Both implementations "
            "report it valid and name tail deletion as unassessed, which is the "
            "limitation section 3.1 states rather than a defect in either"
        ),
    ))

    # -- run-level: a mandatory check that could not run --------------------

    cases.append(CorpusCase(
        name="no_hmac_key_supplied",
        rule="section 7: a mandatory check that did not run is incomplete, never valid",
        store=clean,
        checkpoint=None,
        supply_checkpoint_path=False,
        supply_key=False,
        expect_valid=False,
        expect_level="core",
        expect_exit=2,
        reference_diagnostic="HMAC chain linkage not verified",
        product_diagnostic="chain linkage: no HMAC key was supplied",
        notes="the store is clean; the run is incomplete because the key was withheld",
    ))

    return cases


def case_names() -> list[str]:
    return [case.name for case in build_corpus()]
