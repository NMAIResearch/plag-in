"""Append-only, chained, HMAC-authenticated, metadata-only inference receipts.

A receipt never carries prompt or response content, nor a content-derived
digest (EVIDENCE_REVIEW.md, privacy correction). Each record is bound to
the previous record's HMAC and authenticated with a local HMAC-SHA-256
key stored under the state directory (mode 0600), so mutation, deletion
or reordering is detectable by anyone holding that key. This is a local
integrity mode, not external attestation or non-repudiation
(CODEX_REVIEW_MVP_2026-08-25.md F8). Appends are serialised with an
in-process lock plus an `fcntl.flock` on a dedicated lock file so
concurrent threads and concurrent Linux processes writing the same store
cannot interleave the read-last/construct/append transaction (F3).

A separate authenticated checkpoint (record count plus final HMAC, itself
HMAC-signed under the same key) is maintained alongside the log
(CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md R1). The HMAC chain alone
authenticates each remaining record and its predecessor but carries no
external notion of how many records should exist, so deleting the tail of
the log is indistinguishable from a log that always ended there; the
checkpoint supplies that missing expected-count anchor. `append()` and
`get()` both require the checkpoint and chain to agree before extending or
returning anything (R2): a store that fails this check raises
`ReceiptChainError` rather than silently serving stale or truncated state.
If a receipt is appended but the checkpoint cannot then be replaced, the
store is left in a state that a subsequent verification will refuse (the
checkpoint's stored count will no longer match the log), and the caller
receives `ReceiptCheckpointError` naming that incomplete transaction. The
checkpoint is never silently rebuilt from the log: doing so would accept
exactly the tail-deletion attack it exists to catch.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac as hmac_module
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from plag_in.errors import ReceiptCheckpointError, ReceiptChainError, ReceiptNotFoundError, ReceiptPersistenceError
from plag_in.identity import RUNTIME_SOURCE_SCOPE, canonical_digest

_GENESIS = "0" * 64
_HMAC_KEY_BYTES = 32
INTEGRITY_MODE = "local_hmac"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Bumped whenever a field is added, removed or given a new meaning, so a
# reader can tell which contract a stored record was written under. v3 adds
# runtime_source_digest and runtime_source_scope, binding the identity of
# the PLAG IN Python source that served the request into the receipt itself
# rather than relying on an external run record
# (LIVE_RETEST_2026-08-25_CURRENT_BYTES.md open items).
RECEIPT_SCHEMA_VERSION = "3"

# What each locality level's claim is actually grounded in (PRODUCT_SPEC.md
# section 3): distinct from the level string itself so a reader does not
# have to hold the level-to-evidence mapping in their head.
LOCALITY_EVIDENCE_CLASS = {
    "L0": "configuration_inspected_only",
    "L1": "loopback_bind_and_backend_observed",
    "L2": "os_network_policy_enforced_unmeasured_by_mvp",
    "L3": "os_network_policy_enforced_with_traffic_observation_unmeasured_by_mvp",
}


def load_or_create_hmac_key(path: Path) -> bytes:
    """Create a 32-byte key under `path` with mode 0600, or load an existing one.

    Local integrity mode only: this key never leaves the state directory and
    proves nothing to a party without independent access to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as handle:
            handle.write(os.urandom(_HMAC_KEY_BYTES))
            handle.flush()
            os.fsync(handle.fileno())
    os.chmod(path, 0o600)

    for _ in range(50):  # tolerate a concurrent creator still flushing
        data = path.read_bytes()
        if len(data) == _HMAC_KEY_BYTES:
            return data
        time.sleep(0.02)
    raise ReceiptPersistenceError(
        f"HMAC key file has unexpected length: expected {_HMAC_KEY_BYTES} bytes", path=str(path)
    )


@dataclass(frozen=True)
class Receipt:
    """A metadata-only inference receipt.

    `engine_executable_digest` and `argv_digest` are worker-adapter
    (`llama_server_direct`) identity fields; they are `None` and carry no
    embedded-library meaning on an embedded (`libllama_embedded`) receipt.
    `native_identity_digest`, `native_component_digests` and
    `upstream_identity` are the embedded-adapter equivalents and are `None`
    on a worker receipt (CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md follow-up,
    handoff 07 item C).
    """

    request_id: str
    timestamp: str
    gateway_version: str
    engine: str
    engine_version: str
    model_alias: str
    weight_digest: str
    template_digest: str
    config_digest: str
    locality_level: str
    route: str
    listen_address: str
    backend_address: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    status: str
    # Default is the legacy schema: a bare receipt makes no source-binding
    # claim and is validated as such. Version 3 is opt-in and is stamped by
    # the gateway path, which also supplies the required source fields. A
    # receipt that declares version 3 must therefore carry a valid
    # runtime_source_digest and scope, enforced in `append` (CPO-02).
    schema_version: str = "2"
    inference_mode: str = "worker"
    content_retained: bool = False
    locality_evidence_class: str = ""
    # Worker-adapter identity; absent (`None`) on an embedded receipt.
    engine_executable_digest: str | None = None
    argv_digest: str | None = None
    # Embedded-adapter native identity; absent (`None`) on a worker receipt.
    native_identity_digest: str | None = None
    native_component_digests: dict | None = None
    upstream_identity: str | None = None
    requested_runtime_profile: dict | None = None
    effective_runtime_profile: dict | None = None
    measurement_status: dict | None = None
    gpu_offload: dict | None = None
    runtime_profile: dict | None = None
    # Schema v3: identity of the runtime Python source that served this
    # request. The scope label is stored alongside the digest so the value
    # cannot be mistaken for a repository or release-archive digest. This is
    # a source-tree identity, not a content-derived digest of prompt or
    # response, so it does not weaken the metadata-only guarantee.
    runtime_source_digest: str | None = None
    runtime_source_scope: str | None = None


class ReceiptStore:
    def __init__(self, path: Path, hmac_key_path: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        receipt_preexisting = self.path.exists()
        if not self.path.exists():
            os.close(os.open(str(self.path), os.O_CREAT | os.O_WRONLY, 0o600))
        os.chmod(self.path, 0o600)

        self.hmac_key_path = (
            Path(hmac_key_path) if hmac_key_path is not None else self.path.with_name(self.path.name + ".hmac_key")
        )
        key_preexisting = self.hmac_key_path.exists()
        self._hmac_key = load_or_create_hmac_key(self.hmac_key_path)

        self._checkpoint_path = self.path.with_name(self.path.name + ".checkpoint")
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._thread_lock = threading.Lock()

        # A genuinely new store starts with an authenticated genesis
        # checkpoint. Once state has existed, a missing checkpoint is an
        # integrity failure even when the log is empty. This distinguishes a
        # new store from a store whose complete log and checkpoint were
        # removed while its key remained.
        if not receipt_preexisting and not key_preexisting:
            with self._locked():
                if not self._checkpoint_path.exists() and self.path.stat().st_size == 0:
                    self._write_checkpoint(0, _GENESIS)

    @contextlib.contextmanager
    def _locked(self):
        with self._thread_lock:
            lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)  # cross-process serialisation
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def _record_hmac(self, prev_hmac: str, body: dict) -> str:
        payload_digest = canonical_digest(body)
        message = f"{prev_hmac}:{payload_digest}".encode("utf-8")
        return hmac_module.new(self._hmac_key, message, hashlib.sha256).hexdigest()

    def _checkpoint_hmac(self, count: int, last_hmac: str) -> str:
        message = f"checkpoint:{count}:{last_hmac}".encode("utf-8")
        return hmac_module.new(self._hmac_key, message, hashlib.sha256).hexdigest()

    def _iter_records(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _read_checkpoint(self) -> dict | None:
        if not self._checkpoint_path.exists():
            return None
        try:
            return json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            # An unreadable or malformed checkpoint must never be treated as
            # absent (which would tolerate a deleted tail); force a mismatch.
            return {"count": -1, "last_hmac": "", "checkpoint_hmac": ""}

    def _write_checkpoint(self, count: int, last_hmac: str) -> None:
        payload = json.dumps(
            {"count": count, "last_hmac": last_hmac, "checkpoint_hmac": self._checkpoint_hmac(count, last_hmac)},
            sort_keys=True,
        )
        tmp_path = self._checkpoint_path.with_name(self._checkpoint_path.name + ".tmp")
        fd = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        os.replace(tmp_path, self._checkpoint_path)
        os.chmod(self._checkpoint_path, 0o600)

    def _verify_chain_detailed(self) -> tuple[bool, int, str]:
        """Recompute the HMAC chain and cross-check it against the checkpoint.

        Returns (valid, record_count, last_hmac_seen). Malformed JSON in
        either the log or the checkpoint is treated as invalid rather than
        raised, so a caller never sees an uncaught decoder exception.
        """
        prev_hmac = _GENESIS
        count = 0
        try:
            for record in self._iter_records():
                count += 1
                if not isinstance(record, dict):
                    return False, count, prev_hmac
                if record.get("prev_hmac") != prev_hmac:
                    return False, count, prev_hmac
                body = {k: v for k, v in record.items() if k != "hmac"}
                expected = self._record_hmac(prev_hmac, body)
                if not hmac_module.compare_digest(str(record.get("hmac", "")), expected):
                    return False, count, prev_hmac
                prev_hmac = record["hmac"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, count, prev_hmac

        checkpoint = self._read_checkpoint()
        if checkpoint is None:
            return False, count, prev_hmac
        if not isinstance(checkpoint, dict):
            return False, count, prev_hmac
        if checkpoint.get("count") != count or checkpoint.get("last_hmac") != prev_hmac:
            return False, count, prev_hmac
        expected_checkpoint_hmac = self._checkpoint_hmac(checkpoint.get("count"), checkpoint.get("last_hmac"))
        if not hmac_module.compare_digest(str(checkpoint.get("checkpoint_hmac", "")), expected_checkpoint_hmac):
            return False, count, prev_hmac
        return True, count, prev_hmac

    def _validate_schema_fields(self, receipt: Receipt) -> None:
        """Fail closed on a schema-v3 receipt with absent or malformed source identity.

        A version 3 receipt makes a source-binding claim, so it must carry a
        64-character lowercase SHA-256 `runtime_source_digest` and exactly the
        declared `runtime_source_scope`. This runs before any bytes or the
        checkpoint change, so a rejected receipt leaves the store untouched
        (CPO-02). Older-schema receipts are left readable and make no
        source-binding claim, so they are not validated here.
        """
        if receipt.schema_version != RECEIPT_SCHEMA_VERSION:
            return
        digest = receipt.runtime_source_digest
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise ReceiptPersistenceError(
                "schema v3 receipt requires a 64-character lowercase runtime_source_digest",
                request_id=receipt.request_id,
            )
        if receipt.runtime_source_scope != RUNTIME_SOURCE_SCOPE:
            raise ReceiptPersistenceError(
                "schema v3 receipt requires the declared runtime_source_scope",
                request_id=receipt.request_id,
            )

    def append(self, receipt: Receipt) -> dict:
        self._validate_schema_fields(receipt)
        body = asdict(receipt)
        body["integrity_mode"] = INTEGRITY_MODE

        with self._locked():
            valid, count, prev_hmac = self._verify_chain_detailed()
            if not valid:
                raise ReceiptChainError(
                    "receipt store failed integrity verification; refusing to extend it", record_count=count
                )

            record = dict(body)
            record["prev_hmac"] = prev_hmac
            record["hmac"] = self._record_hmac(prev_hmac, record)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ReceiptPersistenceError(f"failed to persist receipt: {exc}") from exc

            try:
                self._write_checkpoint(count + 1, record["hmac"])
            except OSError as exc:
                raise ReceiptCheckpointError(
                    "receipt was appended but its checkpoint could not be updated; "
                    "the store now fails closed on this record count until an operator reconciles it",
                    request_id=receipt.request_id,
                    record_count=count + 1,
                ) from exc
        return record

    def get(self, request_id: str) -> dict:
        with self._locked():
            valid, count, _ = self._verify_chain_detailed()
            if not valid:
                raise ReceiptChainError(
                    "receipt store failed integrity verification; refusing to return a record from it",
                    record_count=count,
                )
            for record in self._iter_records():
                if record["request_id"] == request_id:
                    return record
        raise ReceiptNotFoundError(f"no receipt for request_id={request_id!r}", request_id=request_id)

    def verify_chain(self) -> tuple[bool, int]:
        """Recompute the HMAC chain and checkpoint agreement; returns (valid, record_count)."""
        with self._locked():
            valid, count, _ = self._verify_chain_detailed()
            return valid, count

    def require_valid_chain(self) -> int:
        valid, count = self.verify_chain()
        if not valid:
            raise ReceiptChainError("receipt chain failed verification", record_count=count)
        return count
