"""Reviewed compatibility records: the only authority for `tested`.

D-021 restricts `tested` to an exact reviewed profile and an exact model
identity. A status string in an operator's configuration file is an
assertion, not evidence, so it cannot carry that restriction: an arbitrary
model reached `tested` merely by being named in a profile whose engine
existed (independent review of general model admission, F1).

A record here binds three things that a configuration file cannot invent:
the complete SHA-256 of the exact model bytes, the identifier of the exact
registered native ABI profile that loaded them, and the reviewed decision
and completed trial that admitted the pair. Configuration may reference a
record by identifier. It may not supply one. Adding a record is a reviewed
source change with its own regression coverage, exactly as adding a native
ABI profile is.

`tested` therefore fails closed in three separate places: configuration
parsing refuses a profile whose record is absent or whose engine is not the
recorded native bundle, and the serve path refuses to carry the state
forward when the model bytes present at load do not hash to the recorded
digest.
"""
from __future__ import annotations

from dataclasses import dataclass

from plag_in.identity import canonical_digest
from plag_in.native_profiles import REGISTERED_NATIVE_ABI_PROFILES


@dataclass(frozen=True)
class CompatibilityRecord:
    """One completed compatibility trial, bound to exact bytes.

    `model_sha256` is the complete-file digest of the model weights, not a
    manifest-declared digest, so a manifest that misreports its own blob
    cannot satisfy it. `trial_report_sha256` binds the record to the exact
    evidence bytes that were reviewed, so editing the report after the fact
    invalidates the record rather than silently re-authorising it.
    """

    record_id: str
    model_sha256: str
    native_abi_profile_id: str
    decision_id: str
    reviewed_on: str
    trial_report: str
    trial_report_sha256: str

    def binding_record(self) -> dict:
        return {
            "record_id": self.record_id,
            "model_sha256": self.model_sha256,
            "native_abi_profile_id": self.native_abi_profile_id,
            "decision_id": self.decision_id,
            "reviewed_on": self.reviewed_on,
            "trial_report": self.trial_report,
            "trial_report_sha256": self.trial_report_sha256,
        }

    def binding_digest(self) -> str:
        """One value covering every field a `tested` claim rests on."""
        return canonical_digest(self.binding_record())


REGISTERED_COMPATIBILITY_RECORDS = (
    CompatibilityRecord(
        record_id="qwen25-3b-linux-x86_64-ollama-v0.32.13-llama-b10380",
        model_sha256=(
            "5ee4f07cdb9beadbbb293e85803c569b"
            "01bd37ed059d2715faa7bb405f31caa6"
        ),
        native_abi_profile_id="linux-x86_64-ollama-v0.32.13-llama-b10380",
        decision_id="D-014",
        reviewed_on="2026-08-26",
        trial_report="LIVE_L3_MODEL_TRIAL_ATTEMPT_5_RESULT_2026-08-26.md",
        trial_report_sha256=(
            "ddd4a4cdf418357c7922d59a335c0f52"
            "71a265e6bdd16ac29f38099fe9d3bf81"
        ),
    ),
)


def find_compatibility_record(
    record_id: str,
    records: tuple[CompatibilityRecord, ...] | None = None,
) -> CompatibilityRecord | None:
    """Return the reviewed record with this exact identifier, or None.

    The registry is read when the lookup runs, so the registered set has one
    live definition rather than a copy captured at import time.
    """
    if records is None:
        records = REGISTERED_COMPATIBILITY_RECORDS
    for record in records:
        if record.record_id == record_id:
            return record
    return None


def registered_record_ids(
    records: tuple[CompatibilityRecord, ...] | None = None,
) -> list[str]:
    if records is None:
        records = REGISTERED_COMPATIBILITY_RECORDS
    return [record.record_id for record in records]


def known_native_abi_profile_ids() -> frozenset[str]:
    return frozenset(profile.profile_id for profile in REGISTERED_NATIVE_ABI_PROFILES)
