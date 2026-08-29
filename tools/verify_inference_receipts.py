#!/usr/bin/env python3
"""Reference verifier for Inference Receipt Specification v0.1.

Standalone. Python 3 standard library only. No dependency on PLAG IN.

Exit status, as required by spec section 7:

    0  valid       every mandatory check ran and passed
    1  invalid     at least one check failed
    2  incomplete  a mandatory check could not be run

A mandatory check that did not run never yields exit 0. Chain linkage is
mandatory, so a run without `--hmac-key` is incomplete, not valid. Skipped
checks are named on exit whatever the status.

Suffix deletion is outside what the HMAC chain can detect. Pass `--checkpoint`
to verify the authenticated anchor that does detect it; without one, the
tail-deletion check is named as unassessed and the reported conformance level
is `core` rather than `core+anchor`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac as hmac_module
import json
import re
import sys

SPEC_VERSION = "0.1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ZERO_HMAC = "0" * 64
VALID_STATUS = {"completed", "failed", "refused"}
VALID_MEASUREMENT = {"measured", "unassessed"}

CORE_FIELDS = (
    "request_id",
    "timestamp",
    "weight_digest",
    "content_retained",
    "status",
    "input_tokens",
    "output_tokens",
)
CHAIN_FIELDS = ("integrity_mode", "prev_hmac", "hmac")

RFC3339_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(\.[0-9]+)?([Zz]|[+-][0-9]{2}:[0-9]{2})"
)


def is_sha256(value) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def is_rfc3339(value) -> bool:
    """Anchored shape, then a real calendar and offset check.

    The pattern alone accepts `2026-99-99T99:99:99+99:99`, which is not a
    timestamp (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F5). Parsing
    after the pattern rejects an impossible month, day, time or offset, and
    rejects 29 February in a common year, which no range check on individual
    fields would catch.
    """
    if not isinstance(value, str) or RFC3339_RE.fullmatch(value) is None:
        return False

    # RFC 3339 section 5.6 bounds the hour at 23. ISO 8601 permits 24:00:00 as
    # an end-of-day form and `fromisoformat` accepts it, so the parse alone
    # would let it through.
    if value[11:13] == "24":
        return False

    # RFC 3339 section 5.7 permits second 60 only at the end of a month in
    # which a leap second was announced, shifted by the offset. Establishing
    # that needs a leap-second table and a UTC-normalised instant, which
    # version 0.1 does not carry, and substituting 59 proved nothing: it
    # accepted 2026-01-01T12:34:60Z, which is not an event
    # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F4).
    # Version 0.1 therefore excludes second 60 outright.
    if value[17:19] == "60":
        return False
    candidate = value
    if candidate[-1] in "Zz":
        candidate = candidate[:-1] + "+00:00"
    try:
        datetime.datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def canonical_digest(record: dict) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def surrogate_in(text: str) -> str | None:
    """Return the first unpaired surrogate code point in `text`, or None."""
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            return f"U+{ord(char):04X}"
    return None


class Findings:
    """Errors, plus skipped checks separated by whether the spec makes them mandatory.

    A mandatory skip forces exit 2. An advisory skip is a property the
    specification does not require this run to establish, such as the anchor
    check when no anchor was supplied, or third-party authenticity, which
    `local_hmac` cannot provide at all. Both are printed; only the mandatory
    one changes the exit status.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.skipped: list[str] = []
        self.mandatory_skipped: list[str] = []

    def error(self, line_no: int | None, message: str) -> None:
        where = f"line {line_no}: " if line_no is not None else ""
        self.errors.append(f"{where}{message}")

    def skip(self, message: str) -> None:
        if message not in self.skipped:
            self.skipped.append(message)

    def skip_mandatory(self, message: str) -> None:
        self.skip(message)
        if message not in self.mandatory_skipped:
            self.mandatory_skipped.append(message)


def check_core(record: dict, line_no: int, found: Findings) -> None:
    declared = record.get("spec_version")
    if declared != SPEC_VERSION:
        found.error(line_no, f"spec_version is {declared!r}, expected {SPEC_VERSION!r}")

    for field in CORE_FIELDS:
        if field not in record:
            found.error(line_no, f"mandatory core field absent: {field}")

    if not isinstance(record.get("request_id"), str) or not record.get("request_id"):
        found.error(line_no, "request_id must be a non-empty string")

    timestamp = record.get("timestamp")
    if not is_rfc3339(timestamp):
        found.error(line_no, f"timestamp is not RFC 3339 with offset: {timestamp!r}")

    if not is_sha256(record.get("weight_digest")):
        found.error(line_no, "weight_digest must be 64 lowercase hex characters")

    if record.get("content_retained") is not False:
        if record.get("content_retained") is not True:
            found.error(line_no, "content_retained must be a boolean")

    status = record.get("status")
    if status not in VALID_STATUS:
        found.error(line_no, f"status {status!r} is not one of {sorted(VALID_STATUS)}")

    for field in ("input_tokens", "output_tokens"):
        value = record.get(field)
        if value is not None and not isinstance(value, int):
            found.error(line_no, f"{field} must be an integer or null")
        if isinstance(value, bool):
            found.error(line_no, f"{field} must not be a boolean")


def check_no_floats(value, line_no: int, found: Findings, path: str = "") -> None:
    """Section 4.5: floats are prohibited; their JSON rendering is not portable."""
    if isinstance(value, float):
        found.error(line_no, f"prohibited float at {path or '<root>'}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_no_floats(item, line_no, found, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_no_floats(item, line_no, found, f"{path}[{index}]")


BANNED_FIELD_NAMES = frozenset({
    "prompt", "response", "messages", "completion", "content",
    "prompt_digest", "response_digest", "prompt_hash", "response_hash",
    "embedding", "token_ids", "input_text", "output_text",
})


def check_prohibited(value, line_no: int, found: Findings, path: str = "") -> None:
    """Scan for prohibited content field names at every depth, not only the top level.

    The core is a minimum rather than an exact schema, so an implementation may
    nest arbitrary structure inside its own blocks. A prompt buried two levels
    down is the same disclosure as a prompt at the top level, so the scan
    recurses (CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D4).

    This is a name check. It cannot establish that an arbitrarily named field
    is free of content-derived data; that remains a reviewer's judgement about
    the emitting implementation.
    """
    if isinstance(value, dict):
        present = BANNED_FIELD_NAMES.intersection(value)
        if present:
            where = f" at {path}" if path else ""
            found.error(line_no, f"prohibited content field present{where}: {sorted(present)}")
        for key, item in value.items():
            check_prohibited(item, line_no, found, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_prohibited(item, line_no, found, f"{path}[{index}]")


def check_encodable(value, line_no: int, found: Findings, path: str = "<root>") -> bool:
    """Reject a lone surrogate in any string value or object key.

    `json.loads` turns the escape `"\\ud800"` into a Python string holding an
    unpaired surrogate, which has no UTF-8 encoding. Every canonical operation
    below would raise `UnicodeEncodeError` on such a record, so it is refused by
    name here and the caller sees `invalid`, not a traceback. Returns False when
    a surrogate was found anywhere in the value.
    """
    clean = True
    if isinstance(value, str):
        found_at = surrogate_in(value)
        if found_at is not None:
            found.error(line_no, f"lone Unicode surrogate {found_at} at {path}: not encodable as UTF-8")
            return False
        return True
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found_at = surrogate_in(key)
                if found_at is not None:
                    found.error(
                        line_no,
                        f"lone Unicode surrogate {found_at} in the object key at {path}: "
                        "not encodable as UTF-8",
                    )
                    clean = False
                    continue
            if not check_encodable(item, line_no, found, f"{path}.{key}"):
                clean = False
        return clean
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not check_encodable(item, line_no, found, f"{path}[{index}]"):
                clean = False
    return clean


def check_optional_blocks(record: dict, line_no: int, found: Findings) -> None:
    """Validate every field an optional block declares, not a subset of them.

    Section 7 obliges a verifier to validate any optional block that is
    present. Checking part of a declared shape and passing the rest through
    silently is the same failure as not checking at all, because the fields
    that go unchecked are exactly the ones an emitter gets wrong without
    noticing (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F2).
    """
    source = record.get("source_identity")
    if source is not None:
        if not isinstance(source, dict):
            found.error(line_no, "source_identity must be an object")
        else:
            if not is_sha256(source.get("digest")):
                found.error(line_no, "source_identity.digest must be 64 lowercase hex")
            if not isinstance(source.get("scope"), str) or not source.get("scope"):
                found.error(line_no, "source_identity.scope must be a non-empty string")

    engine = record.get("engine_identity")
    if engine is not None:
        if not isinstance(engine, dict):
            found.error(line_no, "engine_identity must be an object")
        else:
            # `engine` and `version` are mandatory when the block is present.
            # The three nullable keys may be absent, and absence is read as
            # null: section 5 says so explicitly rather than leaving a reader
            # to infer it from this verifier's use of a defaulting lookup
            # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F3).
            for field in ("engine", "version"):
                if not isinstance(engine.get(field), str):
                    found.error(line_no, f"engine_identity.{field} must be a string")
            nid = engine.get("native_identity_digest")
            if nid is not None and not is_sha256(nid):
                found.error(line_no, "engine_identity.native_identity_digest must be 64 hex or null")

            components = engine.get("component_digests")
            if components is not None:
                if not isinstance(components, dict):
                    found.error(line_no, "engine_identity.component_digests must be an object or null")
                else:
                    for name, digest in components.items():
                        if not is_sha256(digest):
                            found.error(
                                line_no,
                                f"engine_identity.component_digests.{name} must be 64 lowercase hex",
                            )

            upstream = engine.get("upstream_identity")
            if upstream is not None and not isinstance(upstream, str):
                found.error(line_no, "engine_identity.upstream_identity must be a string or null")

    profile = record.get("runtime_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            found.error(line_no, "runtime_profile must be an object")
        else:
            for field in ("requested", "effective", "measurement_status"):
                if not isinstance(profile.get(field), dict):
                    found.error(line_no, f"runtime_profile.{field} must be an object")
            effective = profile.get("effective")
            measurement = profile.get("measurement_status")
            if isinstance(effective, dict) and isinstance(measurement, dict):
                for key in effective:
                    if key not in measurement:
                        found.error(line_no, f"runtime_profile.effective.{key} has no measurement_status")
                    elif measurement[key] not in VALID_MEASUREMENT:
                        found.error(
                            line_no,
                            f"runtime_profile.measurement_status.{key} must be one of {sorted(VALID_MEASUREMENT)}",
                        )
                # Section 5: `measured` reports a value that was read back, so
                # the status must name a key the effective block carries. A
                # status over an absent key is a measurement of nothing, and it
                # was accepted as evidence of measured operation before this
                # check existed. `unassessed` over an absent key stays
                # permitted: naming a field the emitter could not measure is
                # exactly what that value is for.
                for key, value in measurement.items():
                    if value == "measured" and key not in effective:
                        found.error(
                            line_no,
                            f"measurement_status.{key} claims measured with no "
                            f"effective.{key} value to have measured",
                        )

    locality = record.get("locality")
    if locality is not None:
        if not isinstance(locality, dict):
            found.error(line_no, "locality must be an object")
        else:
            for field in ("level", "evidence_class", "listen_address"):
                value = locality.get(field)
                if not isinstance(value, str) or not value:
                    found.error(line_no, f"locality.{field} must be a non-empty string")


def check_chain_shape(record: dict, line_no: int, found: Findings) -> None:
    for field in CHAIN_FIELDS:
        if field not in record:
            found.error(line_no, f"chain field absent: {field}")
    if record.get("integrity_mode") != "local_hmac":
        found.error(line_no, f"unsupported integrity_mode: {record.get('integrity_mode')!r}")
    for field in ("prev_hmac", "hmac"):
        if not is_sha256(record.get(field)):
            found.error(line_no, f"{field} must be 64 lowercase hex characters")


def check_canonical_bytes(raw: bytes, record: dict, line_no: int, found: Findings) -> None:
    """Section 4: the stored bytes are normative, not merely the hashed bytes.

    Parsing and reserialising before comparison would accept any rendering a
    JSON library happens to produce, which is exactly the cross-implementation
    disagreement canonicalisation exists to remove
    (CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D5).
    """
    expected = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if raw != expected:
        found.error(
            line_no,
            f"stored bytes are not canonical: {len(raw)} bytes stored, {len(expected)} canonical",
        )


def check_anchor(
    checkpoint_path: str | None,
    key: bytes | None,
    count: int,
    last_hmac: str,
    found: Findings,
) -> None:
    """Verify the authenticated anchor that binds the expected record count.

    An HMAC chain authenticates each record against its predecessor but carries
    no external notion of how many records should exist, so removing the tail
    leaves a chain that still verifies. The anchor supplies the missing count
    (spec section 3.2).
    """
    if checkpoint_path is None:
        found.skip(
            "tail deletion not assessed: no --checkpoint supplied; the HMAC chain "
            "alone cannot detect removal of the final records"
        )
        return
    if key is None:
        found.skip_mandatory(
            "anchor not verified: --checkpoint supplied but the HMAC key is unavailable"
        )
        return

    try:
        with open(checkpoint_path, encoding="utf-8") as handle:
            anchor = json.loads(handle.read())
    except OSError as exc:
        found.error(None, f"cannot read checkpoint: {exc}")
        return
    except json.JSONDecodeError as exc:
        found.error(None, f"checkpoint is not valid JSON: {exc}")
        return

    if not isinstance(anchor, dict):
        found.error(None, "checkpoint is not a JSON object")
        return

    declared_count = anchor.get("count")
    declared_last = anchor.get("last_hmac")

    # Validate the anchor's shape before authenticating it. The HMAC is
    # computed over an interpolated string, and Python renders True as the
    # integer 1 in a comparison, so an anchor carrying `"count": true`
    # authenticates and then compares equal to a one-record store
    # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F1). Authenticating
    # a malformed anchor and only then comparing it certifies the malformation.
    shape_errors = []
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        shape_errors.append(f"checkpoint count must be an integer, not {type(declared_count).__name__}")
    elif declared_count < 0:
        shape_errors.append(f"checkpoint count must be zero or greater, not {declared_count}")
    if not is_sha256(declared_last):
        shape_errors.append("checkpoint last_hmac must be 64 lowercase hex characters")
    if not is_sha256(anchor.get("checkpoint_hmac")):
        shape_errors.append("checkpoint_hmac must be 64 lowercase hex characters")
    if shape_errors:
        for message in shape_errors:
            found.error(None, message)
        return

    message = f"checkpoint:{declared_count}:{declared_last}".encode("utf-8")
    expected = hmac_module.new(key, message, hashlib.sha256).hexdigest()
    if not hmac_module.compare_digest(expected, str(anchor.get("checkpoint_hmac"))):
        found.error(None, "checkpoint HMAC does not authenticate the checkpoint")
        return

    if declared_count != count:
        found.error(
            None,
            f"record count does not match the authenticated checkpoint: "
            f"{count} in store, {declared_count} declared",
        )
    if str(declared_last) != last_hmac:
        found.error(None, "final record HMAC does not match the authenticated checkpoint")


def carries_a_value(value) -> bool:
    """True when a value is something rather than a placeholder for nothing.

    Null, a blank string and an empty container name a field without recording
    a setting for it. Numbers and booleans always count: zero and false are
    real runtime values.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return True


def record_is_extended(record: dict) -> bool:
    """Section 8: Extended requires source, engine and requested runtime binding.

    Narrowed in revision 5: an empty `runtime_profile.requested` does not
    qualify. A block emptied of the fields that would otherwise disagree is not
    evidence that the runtime was bound, and the level exists to say that it
    was. The blocks must also be valid, not merely present as objects.

    Narrowed again in revision 6: the identity strings must carry a name and a
    version, and the requested block must carry at least one named setting with
    a value. A record declaring an empty engine, an empty version and one empty
    requested key satisfied every presence test while binding nothing.
    """
    source = record.get("source_identity")
    if not isinstance(source, dict) or not is_sha256(source.get("digest")):
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
        isinstance(name, str) and name.strip() and carries_a_value(setting)
        for name, setting in requested.items()
    )


def record_claims_measured_runtime(record: dict) -> bool:
    """True only when an effective field the record carries is labelled `measured`.

    The receipt binds a declared or requested runtime configuration. Effective
    runtime evidence is optional and labelled per field, so no output may
    describe the runtime as measured on the strength of a requested profile.

    The label must belong to a field the effective block carries. Reading the
    status block on its own reported measured operation for a record whose
    effective block was empty.
    """
    profile = record.get("runtime_profile")
    if not isinstance(profile, dict):
        return False
    measurement = profile.get("measurement_status")
    effective = profile.get("effective")
    if not isinstance(measurement, dict) or not isinstance(effective, dict):
        return False
    return any(measurement.get(key) == "measured" for key in effective)


def verify(
    path: str, key_path: str | None, checkpoint_path: str | None = None
) -> tuple[Findings, int, bool, bool]:
    """Verify a store.

    Returns (findings, record_count, every_record_is_extended,
    any_effective_runtime_measured) on every path, including failure. The third
    value drives the reported conformance level: section 8 defines `Extended`
    as valid source and engine identity plus a non-empty requested runtime
    profile, and a verifier must report the level it assessed
    (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F6). The fourth
    bounds what may be said about effective runtime operation.
    """
    found = Findings()

    try:
        with open(path, "rb") as handle:
            raw_bytes = handle.read()
    except OSError as exc:
        found.error(None, f"cannot read store: {exc}")
        return found, 0, False, False

    key = None
    if key_path is None:
        found.skip_mandatory("HMAC chain linkage not verified: no --hmac-key supplied")
    else:
        try:
            with open(key_path, "rb") as handle:
                key = handle.read()
        except OSError as exc:
            found.error(None, f"cannot read HMAC key: {exc}")
            found.skip_mandatory("HMAC chain linkage not verified: key unreadable")

    # Split on LF and keep the raw bytes of each line: the terminator and the
    # exact stored rendering are both part of what section 4 makes normative.
    raw_lines = raw_bytes.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    elif raw_bytes:
        found.error(len(raw_lines), "final record is not LF terminated")

    count = 0
    prev = ZERO_HMAC
    extended = True
    measured_runtime = False
    seen_ids: dict[str, int] = {}
    for index, raw in enumerate(raw_lines, start=1):
        # Section 4 admits exactly one canonical record per LF-terminated line.
        # Skipping a blank line quietly accepts store bytes the specification
        # does not describe, in any position: leading, interior or a surplus
        # trailing line (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F3).
        # A line that cannot be read as a record caps the reported level. The
        # level describes what every record in the store carries, and a line
        # this verifier could not parse carries nothing it has established.
        if not raw.strip():
            found.error(index, "blank or whitespace-only line: the store admits one record per line")
            extended = False
            continue
        count += 1
        if b"\r" in raw:
            found.error(index, "stored line contains CR: newline translation altered the bytes")
        try:
            record = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            found.error(index, f"record is not valid UTF-8: {exc}")
            extended = False
            continue
        except json.JSONDecodeError as exc:
            found.error(index, f"record is not valid JSON: {exc}")
            extended = False
            continue
        if not isinstance(record, dict):
            found.error(index, "record is not a JSON object")
            extended = False
            continue

        # A record carrying an unpaired surrogate cannot be canonically
        # encoded, so it is named invalid here and every canonical operation
        # below is skipped for it rather than allowed to raise.
        encodable = check_encodable(record, index, found)

        check_core(record, index, found)
        check_no_floats(record, index, found)
        check_prohibited(record, index, found)
        check_optional_blocks(record, index, found)
        check_chain_shape(record, index, found)
        if encodable:
            check_canonical_bytes(raw, record, index, found)

        if not record_is_extended(record):
            extended = False
        if record_claims_measured_runtime(record):
            measured_runtime = True

        # Section 2: request_id is unique within the emitting store. Two
        # records sharing one id make the store ambiguous about which
        # inference a receipt describes, and a chain over them still verifies.
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            if request_id in seen_ids:
                found.error(
                    index,
                    f"duplicate request_id {request_id!r}: first seen at line {seen_ids[request_id]}",
                )
            else:
                seen_ids[request_id] = index

        if key is not None and encodable:
            if record.get("prev_hmac") != prev:
                found.error(index, "chain linkage broken: prev_hmac does not match preceding record")
            body = {k: v for k, v in record.items() if k != "hmac"}
            message = f"{record.get('prev_hmac')}:{canonical_digest(body)}".encode("utf-8")
            expected = hmac_module.new(key, message, hashlib.sha256).hexdigest()
            if not hmac_module.compare_digest(expected, str(record.get("hmac"))):
                found.error(index, "HMAC does not authenticate this record")
            prev = str(record.get("hmac"))

    if count == 0:
        found.error(None, "store contains no records; refusing to report an empty store as valid")

    check_anchor(checkpoint_path, key, count, prev, found)

    found.skip(
        "third-party authenticity not established: integrity_mode local_hmac is "
        "tamper-evident to the key holder only"
    )
    if not measured_runtime:
        found.skip(
            "effective runtime operation not established: no effective field in any "
            "record carries measurement_status measured, so the runtime binding is "
            "declared or requested configuration only"
        )
    return found, count, extended and count > 0, measured_runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a receipt store against Inference Receipt Specification v0.1."
    )
    parser.add_argument("store", help="path to the JSONL receipt store")
    parser.add_argument("--hmac-key", default=None, help="path to the local HMAC key")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to the authenticated anchor; without it, tail deletion is not assessed",
    )
    args = parser.parse_args()

    found, count, extended, measured_runtime = verify(args.store, args.hmac_key, args.checkpoint)

    print(f"store={args.store}")
    print(f"records_read={count}")
    for message in found.errors:
        print(f"FAIL {message}")
    for message in found.skipped:
        print(f"SKIPPED {message}")
    print(f"errors={len(found.errors)}")
    print(f"mandatory_checks_skipped={len(found.mandatory_skipped)}")
    # The level is bounded by what the run was given and by what every record
    # carries: one record short of the Extended blocks caps the whole store at
    # Core, and no anchor caps it below +anchor whatever the emitter maintains.
    level = "extended" if extended else "core"
    if args.checkpoint:
        level += "+anchor"
    print(f"conformance_assessed={level}")
    print(f"effective_runtime_measured={str(measured_runtime).lower()}")

    if found.errors:
        print("result=invalid")
        return 1
    if found.mandatory_skipped:
        print("result=incomplete")
        return 2
    print("result=valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
