"""Product conformance checking for the Inference Receipt Specification.

This is a distinct operation from `ReceiptStore.verify_chain()`, which is a
storage-integrity check: it establishes that the HMAC chain and the
authenticated checkpoint agree, over records this implementation can read. It
does not establish that each record satisfies the portable specification, and
its result must never be reported as plain `valid` (work package A1).

This module runs every mandatory rule the specification places on a record and
on a store, and reports the conformance level it assessed. It is a second
implementation of that contract, independent of `tools/verify_inference_receipts.py`:
the reference verifier is a separate opinion on the same bytes, and sharing an
implementation between them would mean sharing its defects. Agreement between
the two is asserted by a generated corpus that covers every mandatory rule; it
is not obtained by making one call the other.

What a passing result does not establish: that the store's HMAC key is held by
anyone other than the operator, that a value under a permitted field name is
free of content-derived data, or that a second implementation in another
language produces the same bytes.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac as hmac_module
import json
import re
from pathlib import Path

from plag_in.identity import canonical_bytes
from plag_in.receipts import CHECKPOINT_MAX_BYTES, read_confined_bytes, store_lock

SPEC_VERSION = "0.1"
GENESIS_HMAC = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RFC3339_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(\.[0-9]+)?([Zz]|[+-][0-9]{2}:[0-9]{2})"
)

MANDATORY_CORE_FIELDS = (
    "request_id",
    "timestamp",
    "weight_digest",
    "content_retained",
    "status",
    "input_tokens",
    "output_tokens",
)
CHAIN_FIELDS = ("integrity_mode", "prev_hmac", "hmac")
TERMINAL_STATUSES = ("completed", "failed", "refused")
MEASUREMENT_VALUES = ("measured", "unassessed")
PROHIBITED_FIELD_NAMES = frozenset({
    "prompt", "response", "messages", "completion", "content",
    "prompt_digest", "response_digest", "prompt_hash", "response_hash",
    "embedding", "token_ids", "input_text", "output_text",
})

_THIRD_PARTY_UNASSESSED = (
    "third-party authenticity: integrity_mode local_hmac is tamper-evident to the "
    "holder of the local key and establishes nothing to anyone else"
)
_TAIL_UNASSESSED = (
    "tail deletion: no checkpoint was supplied, and the HMAC chain alone cannot "
    "detect removal of the final records"
)
_CONTENT_DERIVATION_UNASSESSED = (
    "content derivation under a permitted field name: the prohibited-content check "
    "is a name check and cannot establish that another field is free of "
    "content-derived data"
)


def _is_sha256(value) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_real_rfc3339(value) -> bool:
    """Shape, then a real calendar instant.

    An anchored pattern accepts `2026-99-99T99:99:99+99:99`, so the value is
    parsed as well. Hour 24 is excluded: RFC 3339 section 5.6 bounds the hour
    at 23, while ISO 8601 permits the end-of-day form. Second 60 is excluded
    outright in version 0.1, which carries no leap-second table and therefore
    cannot establish that a given second 60 was an event.
    """
    if not isinstance(value, str) or _RFC3339_RE.fullmatch(value) is None:
        return False
    if value[11:13] == "24":
        return False
    if value[17:19] == "60":
        return False
    normalised = value[:-1] + "+00:00" if value[-1] in "Zz" else value
    try:
        datetime.datetime.fromisoformat(normalised)
    except ValueError:
        return False
    return True


def _surrogate_in(text: str) -> str | None:
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            return f"U+{ord(char):04X}"
    return None


class _Report:
    """Errors and unassessed checks, with mandatory skips tracked separately.

    A mandatory check that could not run makes the result not valid, in the
    same way an error does: reporting conformance for a check that never ran
    would make a passing result mean nothing.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.unassessed: list[str] = []
        self.mandatory_unassessed: list[str] = []

    def error(self, line_no: int | None, message: str) -> None:
        where = f"line {line_no}: " if line_no is not None else ""
        self.errors.append(f"{where}{message}")

    def unassessed_check(self, message: str) -> None:
        if message not in self.unassessed:
            self.unassessed.append(message)

    def mandatory_unassessed_check(self, message: str) -> None:
        self.unassessed_check(message)
        if message not in self.mandatory_unassessed:
            self.mandatory_unassessed.append(message)


def _check_surrogates(value, line_no: int, report: _Report, path: str = "<root>") -> bool:
    """Reject a lone surrogate in any string value or object key.

    `json.loads` turns the escape `"\\ud800"` into a Python string holding an
    unpaired surrogate, which has no UTF-8 encoding. Canonical encoding of such
    a record raises `UnicodeEncodeError`, so the record is refused by name here
    before any encoding is attempted, and the caller sees a diagnostic rather
    than a traceback (work package A5). Returns False when one was found.
    """
    clean = True
    if isinstance(value, str):
        found = _surrogate_in(value)
        if found is not None:
            report.error(line_no, f"lone Unicode surrogate {found} at {path}: not encodable as UTF-8")
            return False
        return True
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found = _surrogate_in(key)
                if found is not None:
                    report.error(
                        line_no,
                        f"lone Unicode surrogate {found} in the object key at {path}: "
                        "not encodable as UTF-8",
                    )
                    clean = False
                    continue
            if not _check_surrogates(item, line_no, report, f"{path}.{key}"):
                clean = False
        return clean
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not _check_surrogates(item, line_no, report, f"{path}[{index}]"):
                clean = False
    return clean


def _check_core(record: dict, line_no: int, report: _Report) -> None:
    declared = record.get("spec_version")
    if declared != SPEC_VERSION:
        report.error(line_no, f"spec_version is {declared!r}, expected {SPEC_VERSION!r}")

    for field in MANDATORY_CORE_FIELDS:
        if field not in record:
            report.error(line_no, f"mandatory core field absent: {field}")

    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        report.error(line_no, "request_id must be a non-empty string")

    if not _is_real_rfc3339(record.get("timestamp")):
        report.error(line_no, f"timestamp is not a real RFC 3339 instant: {record.get('timestamp')!r}")

    if not _is_sha256(record.get("weight_digest")):
        report.error(line_no, "weight_digest must be 64 lowercase hex characters")

    if not isinstance(record.get("content_retained"), bool):
        report.error(line_no, "content_retained must be a boolean")

    status = record.get("status")
    if status not in TERMINAL_STATUSES:
        report.error(line_no, f"status {status!r} is not one of {sorted(TERMINAL_STATUSES)}")

    for field in ("input_tokens", "output_tokens"):
        value = record.get(field)
        if isinstance(value, bool):
            report.error(line_no, f"{field} must not be a boolean")
        elif value is not None and not isinstance(value, int):
            report.error(line_no, f"{field} must be an integer or null")


def _check_no_floats(value, line_no: int, report: _Report, path: str = "<root>") -> None:
    """Section 4: floats are prohibited anywhere in a record."""
    if isinstance(value, float):
        report.error(line_no, f"prohibited float at {path}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_no_floats(item, line_no, report, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_floats(item, line_no, report, f"{path}[{index}]")


def _check_prohibited_names(value, line_no: int, report: _Report, path: str = "<root>") -> None:
    """Section 6: the prohibition applies at every depth, not to the top level only."""
    if isinstance(value, dict):
        present = PROHIBITED_FIELD_NAMES.intersection(value)
        if present:
            report.error(line_no, f"prohibited content field present at {path}: {sorted(present)}")
        for key, item in value.items():
            _check_prohibited_names(item, line_no, report, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_prohibited_names(item, line_no, report, f"{path}[{index}]")


def _check_optional_blocks(record: dict, line_no: int, report: _Report) -> None:
    """Section 5: validate every field a present block declares, per field."""
    source = record.get("source_identity")
    if source is not None:
        if not isinstance(source, dict):
            report.error(line_no, "source_identity must be an object")
        else:
            if not _is_sha256(source.get("digest")):
                report.error(line_no, "source_identity.digest must be 64 lowercase hex")
            scope = source.get("scope")
            if not isinstance(scope, str) or not scope:
                report.error(line_no, "source_identity.scope must be a non-empty string")

    engine = record.get("engine_identity")
    if engine is not None:
        if not isinstance(engine, dict):
            report.error(line_no, "engine_identity must be an object")
        else:
            for field in ("engine", "version"):
                if not isinstance(engine.get(field), str):
                    report.error(line_no, f"engine_identity.{field} must be a string")
            native = engine.get("native_identity_digest")
            if native is not None and not _is_sha256(native):
                report.error(line_no, "engine_identity.native_identity_digest must be 64 hex or null")
            components = engine.get("component_digests")
            if components is not None:
                if not isinstance(components, dict):
                    report.error(line_no, "engine_identity.component_digests must be an object or null")
                else:
                    for name, digest in components.items():
                        if not _is_sha256(digest):
                            report.error(
                                line_no,
                                f"engine_identity.component_digests.{name} must be 64 lowercase hex",
                            )
            upstream = engine.get("upstream_identity")
            if upstream is not None and not isinstance(upstream, str):
                report.error(line_no, "engine_identity.upstream_identity must be a string or null")

    profile = record.get("runtime_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            report.error(line_no, "runtime_profile must be an object")
        else:
            for field in ("requested", "effective", "measurement_status"):
                if not isinstance(profile.get(field), dict):
                    report.error(line_no, f"runtime_profile.{field} must be an object")
            effective = profile.get("effective")
            measurement = profile.get("measurement_status")
            if isinstance(effective, dict) and isinstance(measurement, dict):
                for key in effective:
                    if key not in measurement:
                        report.error(line_no, f"runtime_profile.effective.{key} has no measurement_status")
                    elif measurement[key] not in MEASUREMENT_VALUES:
                        report.error(
                            line_no,
                            f"runtime_profile.measurement_status.{key} must be one of "
                            f"{sorted(MEASUREMENT_VALUES)}",
                        )
                # `measured` names a value that was read. A status naming a key
                # the record does not carry an effective value for asserts a
                # measurement of nothing, and both implementations reported such
                # a record as establishing measured operation (independent
                # review of work package A, D3). `unassessed` remains permitted
                # for a field the emitter can name but could not measure, which
                # is what the embedded adapter reports for GPU offload.
                for key, value in measurement.items():
                    if value == "measured" and key not in effective:
                        report.error(
                            line_no,
                            f"runtime_profile.measurement_status.{key} is measured but "
                            f"runtime_profile.effective carries no {key} value",
                        )

    locality = record.get("locality")
    if locality is not None:
        if not isinstance(locality, dict):
            report.error(line_no, "locality must be an object")
        else:
            for field in ("level", "evidence_class", "listen_address"):
                value = locality.get(field)
                if not isinstance(value, str) or not value:
                    report.error(line_no, f"locality.{field} must be a non-empty string")


def _check_chain_shape(record: dict, line_no: int, report: _Report) -> None:
    for field in CHAIN_FIELDS:
        if field not in record:
            report.error(line_no, f"chain field absent: {field}")
    if record.get("integrity_mode") != "local_hmac":
        report.error(line_no, f"unsupported integrity_mode: {record.get('integrity_mode')!r}")
    for field in ("prev_hmac", "hmac"):
        if not _is_sha256(record.get(field)):
            report.error(line_no, f"{field} must be 64 lowercase hex characters")


def _is_meaningful_value(value) -> bool:
    """True when a value carries something a reader could act on.

    Null, an empty or whitespace-only string and an empty container carry no
    setting. A number and a boolean do, including zero and false, which are
    ordinary runtime values.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return True


def _record_is_extended(record: dict) -> bool:
    """Section 8, narrowed by work package A4 and again after its review.

    Extended requires a valid source identity, a valid engine identity and a
    non-empty `runtime_profile.requested`. An empty requested profile does not
    qualify: emptying the block that would otherwise disagree is not evidence
    that the runtime was bound.

    Presence alone was not enough. A record naming an empty engine, an empty
    version and a single empty requested key reached Extended while binding
    nothing a reader could use, so the identity strings must be non-empty and
    at least one requested entry must be a named key carrying a real value
    (independent review of work package A, D4).
    """
    source = record.get("source_identity")
    if not isinstance(source, dict):
        return False
    if not _is_sha256(source.get("digest")):
        return False
    if not isinstance(source.get("scope"), str) or not source.get("scope").strip():
        return False

    engine = record.get("engine_identity")
    if not isinstance(engine, dict):
        return False
    for field in ("engine", "version"):
        value = engine.get(field)
        if not isinstance(value, str) or not value.strip():
            return False

    profile = record.get("runtime_profile")
    if not isinstance(profile, dict):
        return False
    requested = profile.get("requested")
    if not isinstance(requested, dict) or not requested:
        return False
    return any(
        isinstance(key, str) and key.strip() and _is_meaningful_value(value)
        for key, value in requested.items()
    )


def _record_claims_measured_runtime(record: dict) -> bool:
    """True only when an effective field the record carries is labelled `measured`.

    The receipt binds a declared or requested runtime configuration. Nothing in
    it establishes effective runtime operation unless an effective field says
    it was measured, so no output may describe the runtime as measured or
    effective on the strength of a requested profile alone (work package A4).

    The status must name a key `effective` actually carries. Reading the status
    block alone reported measured operation for a record whose effective block
    was empty, which is a measurement of nothing (D3).
    """
    profile = record.get("runtime_profile")
    if not isinstance(profile, dict):
        return False
    measurement = profile.get("measurement_status")
    effective = profile.get("effective")
    if not isinstance(measurement, dict) or not isinstance(effective, dict):
        return False
    return any(measurement.get(key) == "measured" for key in effective)


def _check_anchor(
    checkpoint_path: Path | None,
    key: bytes | None,
    count: int,
    last_hmac: str,
    report: _Report,
    checkpoint_bytes: bytes | None = None,
    checkpoint_read_error: str | None = None,
) -> None:
    """Assess the anchor from bytes already read under the store lock.

    The bytes are passed in rather than read here, so the log and the
    checkpoint come from one snapshot. Reading the checkpoint at this point
    let an append complete between the two reads, and the check then reported
    a mismatch against a store that verified as sound (R5-F1).
    """
    if checkpoint_path is None:
        report.unassessed_check(_TAIL_UNASSESSED)
        return
    if key is None:
        report.mandatory_unassessed_check(
            "anchor: a checkpoint was supplied but the HMAC key is unavailable"
        )
        return
    if checkpoint_read_error is not None:
        report.error(None, f"checkpoint cannot be read: {checkpoint_read_error}")
        return

    try:
        anchor = json.loads((checkpoint_bytes or b"").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        report.error(None, f"checkpoint is not valid JSON: {exc}")
        return

    if not isinstance(anchor, dict):
        report.error(None, "checkpoint is not a JSON object")
        return

    declared_count = anchor.get("count")
    declared_last = anchor.get("last_hmac")
    declared_hmac = anchor.get("checkpoint_hmac")

    # Shape before authentication. The checkpoint HMAC covers an interpolated
    # string, and a boolean equals an integer in Python, so an anchor carrying
    # `"count": true` authenticates and then compares equal to a one-record
    # store. Authenticating first would certify the malformation.
    shape_errors: list[str] = []
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        shape_errors.append(
            f"checkpoint count must be an integer, not {type(declared_count).__name__}"
        )
    elif declared_count < 0:
        shape_errors.append(f"checkpoint count must be zero or greater, not {declared_count}")
    if not _is_sha256(declared_last):
        shape_errors.append("checkpoint last_hmac must be 64 lowercase hex characters")
    if not _is_sha256(declared_hmac):
        shape_errors.append("checkpoint_hmac must be 64 lowercase hex characters")
    if shape_errors:
        for message in shape_errors:
            report.error(None, message)
        return

    message = f"checkpoint:{declared_count}:{declared_last}".encode("utf-8")
    expected = hmac_module.new(key, message, hashlib.sha256).hexdigest()
    if not hmac_module.compare_digest(expected, str(declared_hmac)):
        report.error(None, "checkpoint HMAC does not authenticate the checkpoint")
        return

    if declared_count != count:
        report.error(
            None,
            f"record count does not match the authenticated checkpoint: "
            f"{count} in store, {declared_count} declared",
        )
    if str(declared_last) != last_hmac:
        report.error(None, "final record HMAC does not match the authenticated checkpoint")


def check_conformance(
    store_path: Path,
    hmac_key_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run every mandatory specification rule against a store.

    Returns `conformance_valid`, `conformance_level`, `record_count`, `errors`
    and `unassessed_checks`. `conformance_valid` is true only when no rule
    failed and no mandatory check was left unassessed.
    """
    report = _Report()
    report.unassessed_check(_THIRD_PARTY_UNASSESSED)
    report.unassessed_check(_CONTENT_DERIVATION_UNASSESSED)

    store_path = Path(store_path)
    # The store, key and checkpoint are read through confined descriptors and
    # under the store's own lock. Reading them by name accepted links at the
    # legacy paths as conforming and blocked on a FIFO at an active sidecar,
    # and reading them without the lock let an ordinary append land between the
    # log read and the checkpoint read, so a sound store was reported invalid
    # (independent review of the fifth pass, R5-F1). The lock is released
    # before the rules are evaluated: it covers taking a coherent snapshot, not
    # the assessment of it.
    key: bytes | None = None
    checkpoint_bytes: bytes | None = None
    checkpoint_read_error: str | None = None
    with store_lock(store_path):
        try:
            raw_bytes = read_confined_bytes(store_path, "receipt store")
        except OSError as exc:
            report.error(None, f"store cannot be read: {exc}")
            return _result(report, 0, False, False, checkpoint_path is not None)

        if hmac_key_path is None:
            report.mandatory_unassessed_check("chain linkage: no HMAC key was supplied")
        else:
            try:
                key = read_confined_bytes(Path(hmac_key_path), "receipt store HMAC key")
            except OSError as exc:
                report.error(None, f"HMAC key cannot be read: {exc}")
                report.mandatory_unassessed_check("chain linkage: the HMAC key is unreadable")

        if checkpoint_path is not None:
            try:
                checkpoint_bytes = read_confined_bytes(
                    Path(checkpoint_path),
                    "receipt store checkpoint",
                    limit=CHECKPOINT_MAX_BYTES,
                )
            except OSError as exc:
                checkpoint_read_error = str(exc)

    lines = raw_bytes.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif raw_bytes:
        report.error(len(lines), "final record is not LF terminated")

    count = 0
    previous = GENESIS_HMAC
    every_record_extended = True
    any_measured_runtime = False
    seen_request_ids: dict[str, int] = {}

    for line_no, raw in enumerate(lines, start=1):
        # A line that cannot be read as a record caps the reported level. The
        # level describes what every record in the store carries, and a line
        # this check could not parse carries nothing it has established.
        if not raw.strip():
            report.error(line_no, "blank or whitespace-only line: a store holds one record per line")
            every_record_extended = False
            continue
        count += 1
        if b"\r" in raw:
            report.error(line_no, "stored line contains CR: newline translation altered the bytes")
        try:
            record = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            report.error(line_no, f"record is not valid UTF-8: {exc}")
            every_record_extended = False
            continue
        except json.JSONDecodeError as exc:
            report.error(line_no, f"record is not valid JSON: {exc}")
            every_record_extended = False
            continue
        if not isinstance(record, dict):
            report.error(line_no, "record is not a JSON object")
            every_record_extended = False
            continue

        encodable = _check_surrogates(record, line_no, report)

        _check_core(record, line_no, report)
        _check_no_floats(record, line_no, report)
        _check_prohibited_names(record, line_no, report)
        _check_optional_blocks(record, line_no, report)
        _check_chain_shape(record, line_no, report)

        # Section 4 makes the stored bytes normative, so they are compared
        # against the canonical serialisation of the parsed record before any
        # reserialisation is trusted. A record carrying an unencodable string
        # has already failed by name above; encoding it here would raise.
        if encodable:
            expected = canonical_bytes(record)
            if raw != expected:
                report.error(
                    line_no,
                    f"stored bytes are not canonical: {len(raw)} stored, {len(expected)} canonical",
                )

        if not _record_is_extended(record):
            every_record_extended = False
        if _record_claims_measured_runtime(record):
            any_measured_runtime = True

        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            if request_id in seen_request_ids:
                report.error(
                    line_no,
                    f"duplicate request_id {request_id!r}: first seen at line "
                    f"{seen_request_ids[request_id]}",
                )
            else:
                seen_request_ids[request_id] = line_no

        if key is not None and encodable:
            if record.get("prev_hmac") != previous:
                report.error(line_no, "chain linkage broken: prev_hmac does not match the preceding record")
            body = {k: v for k, v in record.items() if k != "hmac"}
            digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
            message = f"{record.get('prev_hmac')}:{digest}".encode("utf-8")
            expected_hmac = hmac_module.new(key, message, hashlib.sha256).hexdigest()
            if not hmac_module.compare_digest(expected_hmac, str(record.get("hmac"))):
                report.error(line_no, "HMAC does not authenticate this record")
            previous = str(record.get("hmac"))

    if count == 0:
        report.error(None, "store contains no records; an empty store is not reported as conforming")

    _check_anchor(
        Path(checkpoint_path) if checkpoint_path is not None else None,
        key,
        count,
        previous,
        report,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_read_error=checkpoint_read_error,
    )

    return _result(
        report,
        count,
        every_record_extended and count > 0,
        any_measured_runtime,
        checkpoint_path is not None,
    )


def _result(
    report: _Report,
    count: int,
    extended: bool,
    measured_runtime: bool,
    anchor_in_scope: bool,
) -> dict:
    # The level is bounded twice: by what the run was given, so a run without a
    # checkpoint assesses no anchor however many the emitter maintains, and by
    # what every record carries, so one record short of the Extended blocks
    # caps the whole store at Core.
    level = "extended" if extended else "core"
    if anchor_in_scope:
        level += "+anchor"
    return {
        "conformance_valid": not report.errors and not report.mandatory_unassessed,
        "conformance_level": level,
        "record_count": count,
        "errors": list(report.errors),
        "unassessed_checks": list(report.unassessed),
        "mandatory_checks_unassessed": list(report.mandatory_unassessed),
        # Never claimed from a requested profile alone (work package A4).
        "effective_runtime_measured": measured_runtime,
    }
