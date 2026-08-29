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
import errno
import fcntl
import functools
import hashlib
import hmac as hmac_module
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
import time
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from plag_in.errors import (
    ReceiptCheckpointError,
    ReceiptChainError,
    ReceiptMigrationRequiredError,
    ReceiptNotFoundError,
    ReceiptPersistenceError,
)
from plag_in.identity import RUNTIME_SOURCE_SCOPE, canonical_bytes, canonical_digest, canonical_line

_GENESIS = "0" * 64
_HMAC_KEY_BYTES = 32
# A checkpoint is three short fields. The bound exists so a large file planted
# at that name cannot be loaded into memory before it is rejected. It is a
# maximum size, not a maximum read: a file larger than this is refused, never
# truncated to its first bytes.
CHECKPOINT_MAX_BYTES = 4096
_CHECKPOINT_READ_LIMIT = CHECKPOINT_MAX_BYTES
INTEGRITY_MODE = "local_hmac"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Bumped whenever a field is added, removed or given a new meaning, so a
# reader can tell which contract a stored record was written under. v3 adds
# runtime_source_digest and runtime_source_scope, binding the identity of
# the PLAG IN Python source that served the request into the receipt itself
# rather than relying on an external run record
# (LIVE_RETEST_2026-08-25_CURRENT_BYTES.md open items).
# Version 4 gives `runtime_profile` a single meaning across both adapters: the
# Inference Receipt Specification section 5 shape, `{requested, effective,
# measurement_status}`, all objects. Until version 3 the worker adapter put its
# own record there and the embedded adapter put the specification shape, so one
# field name carried two dialects and a worker receipt failed the reference
# verifier (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F2). The
# adapter's own record now lives under `plag_in_adapter_record`, namespaced so a
# later specification version can add fields without colliding with it.
# Version 5 is the version boundary for the revision 5 specification: string
# values and object keys are validated for lone surrogates before canonical
# encoding, and a store holding any pre-version-5 record refuses a version 5
# append rather than carrying two field contracts under one chain. Versions 2,
# 3 and 4 remain readable under their documented historical integrity rules and
# are never rewritten.
RECEIPT_SCHEMA_VERSION = "5"

# Every schema generation this implementation can read. Only
# `RECEIPT_SCHEMA_VERSION` may be written.
READABLE_SCHEMA_VERSIONS = ("2", "3", "4", "5")

# Default filename for a current-schema store created beside a legacy one. The
# version is in the name so the two are distinguishable in a directory listing
# without opening either.
CURRENT_STORE_FILENAME = f"receipts.v{RECEIPT_SCHEMA_VERSION}.jsonl"

# Inference Receipt Specification version this record conforms to. Versioned
# independently of RECEIPT_SCHEMA_VERSION: the schema version tracks this
# implementation's field set, the spec version tracks the portable contract
# an external verifier checks.
RECEIPT_SPEC_VERSION = "0.1"

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

    The exclusive create refuses a link, but that only covers a key this call
    creates. An existing link at the key name took the `FileExistsError` branch
    and was then followed by `chmod` and by the read, which changed the mode of
    a file outside the state directory and accepted its bytes as the key
    (independent review of the second pass, F2). Every later operation on an
    existing key therefore goes through a descriptor opened with `O_NOFOLLOW`,
    and the mode is set on that descriptor rather than on the name.
    """
    return _load_or_create_hmac_key_with_origin(path)[0]


def _load_or_create_hmac_key_with_origin(path: Path) -> tuple[bytes, bool, tuple[int, int] | None]:
    """`load_or_create_hmac_key`, also reporting whether this call created it.

    Returns the key, whether this call created the file, and the identity of
    the file it created. The caller that needs the second and third values is
    `ReceiptStore`, which decides whether a store is genuinely new and which
    files its own failure must remove. Both decisions rest on the create itself
    rather than on an earlier look at the name (T3-F3), and the identity comes
    from the descriptor the exclusive create returned, so a key this call did
    not make can never be removed on its behalf (fourth pass, F4-R4).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    # Everything this call creates, registered at the moment of creation. The
    # obligation stays live through the read-back below, which previously sat
    # outside the cleanup block and left a new key behind when it failed
    # (independent review of the sixth pass, R6-F2, windows 3 and 4).
    own_created: list[tuple[Path, tuple[int, int]]] = []
    try:
        try:
            fd = _open_confined(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                "receipt store HMAC key",
                created_registry=own_created,
            )
        except FileExistsError:
            pass
        else:
            created = True
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(os.urandom(_HMAC_KEY_BYTES))
                handle.flush()
                os.fsync(handle.fileno())

        for _ in range(50):  # tolerate a concurrent creator still flushing
            key_fd = _open_confined(path, os.O_RDONLY, 0o600, "receipt store HMAC key")
            try:
                os.fchmod(key_fd, 0o600)
                # Read to the end, one byte past the key length, so the result
                # distinguishes a key of exactly the right size from a longer
                # file and from a shorter one (R7-F1).
                data = _read_to_end(key_fd, _HMAC_KEY_BYTES)
            finally:
                os.close(key_fd)
            if len(data) > _HMAC_KEY_BYTES:
                # Too long is a settled answer, not something a retry changes.
                # The retries exist for the declared case of a concurrent
                # creator that has not finished writing, which is short.
                raise ReceiptPersistenceError(
                    f"HMAC key file has unexpected length: expected {_HMAC_KEY_BYTES} bytes",
                    path=str(path),
                )
            if len(data) == _HMAC_KEY_BYTES:
                return data, created, (own_created[0][1] if own_created else None)
            time.sleep(0.02)
        raise ReceiptPersistenceError(
            f"HMAC key file has unexpected length: expected {_HMAC_KEY_BYTES} bytes",
            path=str(path),
        )
    except BaseException:
        for created_path, identity in reversed(own_created):
            _unlink_if_ours(created_path, identity)
        raise


def _canonicalise_floats(value):
    """Render every float as a decimal string so the record stays portable.

    Inference Receipt Specification v0.1 section 4.5 prohibits floats: the same
    float has several valid JSON renderings, so a non-Python implementation can
    produce different bytes for the same record and fail HMAC verification.
    The specification requires a deterministic decimal string documented by the
    emitter rather than a cross-language shortest form. PLAG IN documents
    Python `repr` for finite floats: it is deterministic for a given value on a
    given Python version, and a reader comparing two records compares strings.
    Two emitters that document different representations produce different
    logical records, which the specification permits and this docstring makes
    explicit rather than implying interoperability that is not established.

    This conversion is permitted only inside implementation-defined optional
    blocks, where a client-supplied sampling parameter such as `temperature`
    arrives as a float and the emitter has no field schema for it. It changes
    the value's JSON type from number to string, so it must never stand in for
    validating a field the record declares as an integer: those are rejected
    by `_validate_schema_fields` before this function runs (spec section 4.2,
    "Converting a float in an implementation-defined block",
    CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D6).

    Tuples are converted alongside lists because `json.dumps` serialises both
    as JSON arrays, so a tuple carrying a float would otherwise reach the store
    unconverted.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return {k: _canonicalise_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise_floats(v) for v in value]
    return value


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
    latency_us: int | None
    status: str
    # New receipt construction defaults to the current schema. Historical
    # version 2 records remain readable, but cannot be appended by current
    # code without the version 3 source-binding fields.
    schema_version: str = RECEIPT_SCHEMA_VERSION
    spec_version: str = RECEIPT_SPEC_VERSION
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
    # Two real shapes, not one. The embedded adapter reports a per-key mapping
    # (`{"gpu_layers": "unassessed", ...}`, `engines/libllama.py`); the worker
    # adapter has no effective-value query at all and reports the single reason
    # string `not_applicable_to_worker_adapter` (`config.py`). The annotation
    # records what the field carries; narrowing it to `dict` would describe a
    # contract neither adapter honours.
    measurement_status: dict | str | None = None
    gpu_offload: dict | None = None
    runtime_profile: dict | None = None
    # Schema v3: identity of the runtime Python source that served this
    # request. The scope label is stored alongside the digest so the value
    # cannot be mistaken for a repository or release-archive digest. This is
    # a source-tree identity, not a content-derived digest of prompt or
    # response, so it does not weaken the metadata-only guarantee.
    runtime_source_digest: str | None = None
    runtime_source_scope: str | None = None
    # Schema v4: the adapter's own record, whatever shape that adapter uses.
    # It is namespaced because it is this implementation's, not the
    # specification's, and `runtime_profile` above now carries only the
    # portable section 5 shape.
    plag_in_adapter_record: dict | None = None


# The blocks whose interior this implementation holds no schema for. A client
# may put anything inside them, including a float sampling parameter, so spec
# section 4.2 permits conversion there and only there. Every field outside this
# set is declared with a type and is validated instead
# (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F4).
IMPLEMENTATION_DEFINED_BLOCKS = (
    "requested_runtime_profile",
    "effective_runtime_profile",
    "measurement_status",
    "gpu_offload",
    "runtime_profile",
    "native_component_digests",
    # Schema v4. The adapter record is the most implementation-defined block in
    # the receipt: it carries whatever that adapter reports, including the
    # client's float sampling parameters.
    "plag_in_adapter_record",
)


@functools.cache
def _receipt_field_types() -> dict[str, tuple[tuple[type, ...], bool]]:
    """Map each `Receipt` field to its permitted types and whether null is allowed.

    Read from the dataclass annotations rather than a hand-written table, so a
    field added to `Receipt` is validated without a second edit here.
    """
    hints = typing.get_type_hints(Receipt)
    resolved: dict[str, tuple[tuple[type, ...], bool]] = {}
    for field in fields(Receipt):
        annotation = hints[field.name]
        args = typing.get_args(annotation)
        if args:
            allowed = tuple(arg for arg in args if arg is not type(None))
            optional = type(None) in args
        else:
            allowed = (annotation,)
            optional = False
        resolved[field.name] = (allowed, optional)
    return resolved


def _type_names(allowed: tuple[type, ...]) -> str:
    return " or ".join(sorted(item.__name__ for item in allowed))


def _reject_non_finite(value, request_id: str, path: str) -> None:
    """Reject NaN and the infinities anywhere inside an implementation-defined block.

    `repr(float('nan'))` is `'nan'`, which is not a decimal representation that
    round-trips to the same value, so converting it produces a string the
    specification does not permit and no reader can interpret
    (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F4).
    """
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ReceiptPersistenceError(
            f"non-finite float at {path}: a receipt carries no value that cannot be written as a decimal",
            request_id=request_id,
            field=path,
            observed_value=repr(value),
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, request_id, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, request_id, f"{path}[{index}]")


def _surrogate_in(text: str) -> str | None:
    """Return the first lone surrogate code point in `text`, or None."""
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            return f"U+{ord(char):04X}"
    return None


def _reject_surrogates(value, request_id: str, path: str = "<root>") -> None:
    """Reject surrogate code points in any string value or object key.

    A Python `str` may hold an unpaired surrogate, which has no UTF-8 encoding.
    `json.loads` produces one from the escape `"\\ud800"`, so a record can reach
    this emitter carrying a value that canonical encoding cannot represent.
    Left to the encoder it raises `UnicodeEncodeError` and the caller sees a
    traceback rather than a named refusal, so the whole record is walked here,
    keys included, before any encoding is attempted (work package A5).
    """
    if isinstance(value, str):
        found = _surrogate_in(value)
        if found is not None:
            raise ReceiptPersistenceError(
                f"lone Unicode surrogate {found} at {path}: a receipt carries no value "
                "that cannot be encoded as UTF-8",
                request_id=request_id,
                field=path,
                surrogate=found,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found = _surrogate_in(key)
                if found is not None:
                    raise ReceiptPersistenceError(
                        f"lone Unicode surrogate {found} in the object key at {path}: "
                        "a receipt carries no key that cannot be encoded as UTF-8",
                        request_id=request_id,
                        field=path,
                        surrogate=found,
                    )
            _reject_surrogates(item, request_id, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_surrogates(item, request_id, f"{path}[{index}]")


def _lstat_or_none(path: Path):
    """Whether this name is occupied, without following what occupies it.

    `Path.exists()` resolves a link and answers about the target, so it reports
    an absent name as present and a dangling link as absent. Both readings are
    wrong for a caller asking whether it is about to create something.
    """
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _identity_of_fd(fd: int) -> tuple[int, int]:
    """The device and inode of an open descriptor, for later ownership checks."""
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino)


def _read_to_end(fd: int, limit: int | None = None) -> bytes:
    """Read a descriptor until it ends, or until one byte past `limit`.

    The single rule for every length-sensitive read in this module. A `read(2)`
    that returns fewer bytes than it was asked for is permitted and says
    nothing about where the file ends, so no caller may treat one result as the
    whole file. Doing so accepted an oversized checkpoint on its prefix
    (independent review of the sixth pass, R6-F1) and then, in a place the
    sixth-pass repair did not reach, both refused a valid key read one byte at
    a time and accepted an over-long key whose first read returned the expected
    number of bytes (seventh pass, R7-F1).

    With `limit`, one byte beyond it is requested, so a caller can tell a file
    of exactly `limit` bytes from a longer one. Without, the read is unbounded.
    """
    raw = b""
    while limit is None or len(raw) <= limit:
        want = 1 << 20 if limit is None else limit + 1 - len(raw)
        chunk = os.read(fd, want)
        if not chunk:
            break
        raw += chunk
    return raw


def _unlink_created_without_identity(path: Path) -> None:
    """Remove a name this call just created exclusively, without an inode check.

    The one case where inode-bound cleanup is unavailable: the exclusive create
    succeeded, so the file exists and is this call's, but `fstat` on its
    descriptor then failed, so its identity was never obtained. Leaving it is
    the alternative, and that is the defect this exists to close (independent
    review of the sixth pass, R6-F2, window 1).

    The removal is by name, so it rests on the cooperating-process boundary
    `PRODUCT_SPEC.md` declares: a party that replaces the name between the
    exclusive create and this removal is outside the threat model. Every other
    cleanup in this module is inode-bound and does not rest on that.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return
    if stat.S_ISREG(info.st_mode):
        try:
            os.unlink(path)
        except OSError:
            pass


def _open_confined(
    path: Path,
    flags: int,
    mode: int,
    role: str,
    created_registry=None,
    *,
    error_type: type = ReceiptPersistenceError,
    set_noun: str = "the receipt store set",
) -> int:
    """Open `path` for the store, refusing a symbolic link at that name.

    `error_type` and `set_noun` exist so state files outside the receipt set
    can use this same rule and still refuse in their own vocabulary. The
    defaults reproduce the receipt wording and type exactly, so every caller
    inside this module is unaffected. Two implementations of one confinement
    rule is the shape that let a repaired read rule survive unrepaired in the
    key loader (independent review of the seventh pass, R7-F1), so the rule
    stays in one place and the message varies.

    Every name the receipt store creates or opens sits under the state
    directory by construction, but a name is not a file. A link planted at one
    of these names sends the open, and everything written through it, to
    whatever the link names, which may be outside the state directory
    entirely. `O_NOFOLLOW` refuses that in the same system call that would
    otherwise follow it, so there is no window between a check and the open,
    and with `O_CREAT` no file is created at the far end.

    Confinement of the pointer and of the store, key and checkpoint names is
    enforced separately in `StateLayout.validate_active_store_path`. That check
    is reached only through the pointer. This one covers the names a directly
    constructed store opens, which is how the transaction lock was reached
    without any confinement at all (independent review of work package A,
    second pass, D1).

    The regular-file check is not redundant with `O_NOFOLLOW`: a FIFO or a
    device node at one of these names is not a link and would otherwise be
    opened, blocking the store or writing to a device.

    `O_NONBLOCK` is what makes that check reachable. `fstat` can only classify
    a descriptor the open has already returned, and opening a FIFO waits for a
    peer: for writing until a reader arrives, for reading until a writer does.
    A FIFO planted at the store or key name therefore held the constructor
    inside `os.open` indefinitely, which is a denial of the refusal rather than
    a refusal (independent review of the third pass, T3-F1 and T3-F4). With
    `O_NONBLOCK` the open either returns, and `fstat` refuses it, or fails at
    once with an errno this maps to the same typed refusal. The flag is cleared
    once the descriptor is known to be a regular file, for which it means
    nothing on Linux either way.

    Errors are mapped rather than propagated because a caller of this function
    is asking one question, whether this name may be used as a receipt-store
    file, and a raw `IsADirectoryError` is that answer in a form no caller
    branches on (T3-F2).
    """
    try:
        fd = os.open(str(path), flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
    except OSError as exc:
        # Linux reports ELOOP for O_NOFOLLOW on a link; some other systems
        # report EMLINK for the same refusal.
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise error_type(
                f"the {role} is a symbolic link; {set_noun} is regular files only",
                path=str(path),
                role=role,
            ) from exc
        # EISDIR: a directory, opened for writing. ENXIO: a FIFO opened for
        # writing with no reader, which is the non-blocking refusal. EAGAIN:
        # a device that declines a non-blocking open. Each is the same answer
        # to the caller's question.
        if exc.errno in (errno.EISDIR, errno.ENXIO, errno.EAGAIN, errno.EWOULDBLOCK):
            raise error_type(
                f"the {role} is not a regular file",
                path=str(path),
                role=role,
                errno=errno.errorcode.get(exc.errno, exc.errno),
            ) from exc
        raise
    # A file this call created exclusively is this call's to remove if any
    # later step in this function fails. Ownership metadata that never returns
    # to the caller cannot be rolled back by it, which left a store behind when
    # a failure landed between the create and the return (independent review of
    # the fifth pass, R5-F3).
    # Identity is taken and registered as the first thing after the create, so
    # the obligation to remove the file exists from the moment the file does.
    # Registering it later left a window in which the file existed and nothing
    # was responsible for it, at the caller's mode setting, at its own second
    # `fstat` and at each later step (R6-F2).
    created_exclusively = bool(flags & os.O_CREAT) and bool(flags & os.O_EXCL)
    identity = None
    try:
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        if created_exclusively and created_registry is not None:
            created_registry.append((Path(path), identity))
        if not stat.S_ISREG(info.st_mode):
            raise error_type(
                f"the {role} is not a regular file",
                path=str(path),
                role=role,
            )
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        if created_exclusively:
            if identity is not None:
                _unlink_if_ours(path, identity)
            else:
                # `fstat` itself failed, so no inode is available.
                _unlink_created_without_identity(Path(path))
        raise
    return fd


def read_confined_bytes(
    path: Path,
    role: str,
    limit: int | None = None,
    *,
    error_type: type = ReceiptPersistenceError,
    set_noun: str = "the receipt store set",
) -> bytes:
    """Read a receipt-set file through a confined descriptor.

    The reading half of `_open_confined`, for every route that needs the bytes
    of a store, key or checkpoint. Reading these by name is what let links at
    the legacy paths be accepted as conforming and let a FIFO at an active
    sidecar hold a check open (independent review of the fifth pass, R5-F1).

    `limit` is a maximum size, not a maximum read. One byte beyond it is
    requested so an oversized file is detected rather than silently truncated
    to its first `limit` bytes: accepting a valid prefix and never seeing the
    trailer was the regression this bound introduced (R5-F2).
    """
    fd = _open_confined(path, os.O_RDONLY, 0o600, role, error_type=error_type, set_noun=set_noun)
    try:
        raw = _read_to_end(fd, limit)
        if limit is not None and len(raw) > limit:
            raise error_type(
                f"the {role} is larger than the {limit} byte maximum",
                path=str(path),
                role=role,
                maximum_bytes=limit,
            )
        return raw
    finally:
        os.close(fd)


@contextlib.contextmanager
def store_lock(store_path: Path):
    """Hold a store's own lock, for a reader outside `ReceiptStore`.

    A conformance check reads the log and then the checkpoint. Without this,
    an ordinary append could complete between those two reads, and the check
    reported a store invalid that verified as sound immediately afterwards
    (R5-F1). The lock is the same one `ReceiptStore` takes, so the two
    serialise against each other.
    """
    lock_path = Path(store_path).with_name(Path(store_path).name + ".lock")
    lock_fd = _open_confined(lock_path, os.O_CREAT | os.O_RDWR, 0o600, "receipt store lock file")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield lock_path
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


class ReceiptStore:
    def __init__(self, path: Path, hmac_key_path: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Creation ownership comes from the create, never from an earlier
        # observation of the name. An `lstat` that reports the name absent is a
        # statement about the past by the time the open runs: a second process
        # creating the store in that window left this call holding someone
        # else's inode and calling it its own, which the rollback below then
        # removed (independent review of the third pass, T3-F3). The exclusive
        # create answers the question atomically. Winning it is the only
        # evidence that this call created the file; losing it with EEXIST means
        # the name was already occupied, and the second open then applies the
        # same confinement to whatever occupies it.
        #
        # The mode is set through the descriptor, so it cannot reach a file
        # other than the one opened. That is a property of this operation, not
        # a lasting confinement: later reads and appends address the store by
        # pathname, under the boundary PRODUCT_SPEC.md declares.
        self.hmac_key_path = (
            Path(hmac_key_path) if hmac_key_path is not None else self.path.with_name(self.path.name + ".hmac_key")
        )
        self._checkpoint_path = self.path.with_name(self.path.name + ".checkpoint")
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._thread_lock = threading.Lock()

        # The whole of first construction runs under the store's own lock, and
        # the lock is taken before anything is created.
        #
        # Taking it first is what makes the rollback complete. A refusal at the
        # lock name previously happened after the store and key existed, and
        # the rollback covered key loading only, so a FIFO at the lock left a
        # half-built store set behind (independent review of the fourth pass,
        # F4-R4). Nothing has been created when this open runs, so there is
        # nothing to undo.
        #
        # Holding it across store, key and checkpoint is what makes the answer
        # coherent. Each file's creation was already atomic on its own, but two
        # atomic answers are not one atomic answer: two first constructors
        # could split the pair, each seeing one file as pre-existing, so
        # neither wrote the genesis checkpoint and both then failed
        # verification (F4-R1). Serialised, the first caller creates the whole
        # set and the second sees a complete one.
        with self._locked():
            self._construct_store_set()

    def _construct_store_set(self) -> None:
        """Create or adopt the store, key and genesis checkpoint as one set.

        The caller holds the store lock. Every file this call creates is
        removed if any later step fails, and no file that was already present
        is touched: ownership is the inode of a descriptor obtained by winning
        an exclusive create, never an observation of the name.
        """
        created: list[tuple[Path, tuple[int, int]]] = []
        try:
            # Creation ownership comes from the create, never from an earlier
            # observation of the name. An `lstat` that reports the name absent
            # is a statement about the past by the time the open runs: a second
            # process creating the store in that window left this call holding
            # someone else's inode and calling it its own, which the rollback
            # then removed (third pass, T3-F3). Winning the exclusive create is
            # the only evidence that this call created the file; losing it with
            # EEXIST means the name was occupied, and the second open applies
            # the same confinement to whatever occupies it.
            #
            # The mode is set through the descriptor, so it cannot reach a file
            # other than the one opened. That is a property of this operation,
            # not a lasting confinement: later reads and appends address the
            # store by pathname, under the boundary PRODUCT_SPEC.md declares.
            try:
                store_fd = _open_confined(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                    "receipt store",
                    # Registered by the create itself, so a failure at the mode
                    # setting below is already covered (R6-F2, window 2).
                    created_registry=created,
                )
            except FileExistsError:
                receipt_preexisting = True
                store_fd = _open_confined(self.path, os.O_WRONLY, 0o600, "receipt store")
                try:
                    os.fchmod(store_fd, 0o600)
                finally:
                    os.close(store_fd)
            else:
                receipt_preexisting = False
                try:
                    os.fchmod(store_fd, 0o600)
                finally:
                    os.close(store_fd)

            # Same rule for the key: the create reports whether this call made
            # it, so the genesis decision cannot rest on a stale reading.
            self._hmac_key, key_created, key_identity = _load_or_create_hmac_key_with_origin(
                self.hmac_key_path
            )
            if key_created and key_identity is not None:
                created.append((self.hmac_key_path, key_identity))
            key_preexisting = not key_created

            # A genuinely new store starts with an authenticated genesis
            # checkpoint. Once state has existed, a missing checkpoint is an
            # integrity failure even when the log is empty. This distinguishes
            # a new store from a store whose complete log and checkpoint were
            # removed while its key remained.
            if not receipt_preexisting and not key_preexisting:
                if _lstat_or_none(self._checkpoint_path) is not None:
                    # A checkpoint beside a store and key that did not exist
                    # until this call is not state this call can adopt: it was
                    # authenticated under some other key, so the set it would
                    # complete cannot verify. Returning it as a successful
                    # construction handed back a larger invalid set than the
                    # one found (independent review of the fifth pass, R5-F4).
                    # Refusing rolls back the store and key created above.
                    raise ReceiptPersistenceError(
                        "a checkpoint already exists beside a store and key that did not; "
                        "the receipt set cannot be completed from it",
                        path=str(self._checkpoint_path),
                        role="receipt store checkpoint",
                    )
                if self.path.stat().st_size == 0:
                    self._write_checkpoint(0, _GENESIS, created_registry=created)
        except BaseException:
            # Reverse order, so the store set never exists in a state this call
            # would not itself have produced.
            for path, identity in reversed(created):
                _unlink_if_ours(path, identity)
            raise

    @contextlib.contextmanager
    def _locked(self):
        with self._thread_lock:
            lock_fd = _open_confined(
                self._lock_path, os.O_CREAT | os.O_RDWR, 0o600, "receipt store lock file"
            )
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

    def _read_lines(self) -> tuple[list[bytes], bool]:
        """Return (raw lines, whether the last one carried its LF terminator).

        The product verification path reads the same bytes the reference
        verifier reads. Stripping and skipping meant `--check-chain` and chat
        shutdown reported a store valid that the reference verifier rejected
        (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REVISION_3_2026-08-26.md R3-F1).
        """
        try:
            raw = read_confined_bytes(self.path, "receipt store")
        except FileNotFoundError:
            return [], True
        if not raw:
            return [], True
        lines = raw.split(b"\n")
        if lines[-1] == b"":
            lines.pop()
            return lines, True
        return lines, False

    def _iter_records(self):
        lines, _terminated = self._read_lines()
        for raw in lines:
            if raw.strip():
                yield json.loads(raw.decode("utf-8"))

    def _read_checkpoint(self) -> dict | None:
        """The stored checkpoint, or None when the name is unoccupied.

        Read through a confined descriptor. Reading by name accepted a link
        here: the checkpoint is the anchor that says how many records should
        exist, so outside bytes planted at this name made an empty store verify
        (independent review of the fourth pass, F4-R2), and a FIFO at it held
        verification open indefinitely (F4-R3). The active-pointer path refuses
        a linked checkpoint during pointer validation, but the direct legacy
        path never passes through that validation, which is how both were
        reachable.

        A refusal is raised rather than folded into the mismatch value below. A
        checkpoint that cannot be trusted to be the product's own file is a
        different fact from a checkpoint whose contents disagree, and reporting
        the second in place of the first would tell an operator the chain is
        broken when the truth is that the name has been taken over.
        """
        try:
            raw = read_confined_bytes(
                self._checkpoint_path, "receipt store checkpoint", limit=_CHECKPOINT_READ_LIMIT
            )
        except FileNotFoundError:
            return None
        except OSError:
            return {"count": -1, "last_hmac": "", "checkpoint_hmac": ""}
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # An unreadable or malformed checkpoint must never be treated as
            # absent (which would tolerate a deleted tail); force a mismatch.
            return {"count": -1, "last_hmac": "", "checkpoint_hmac": ""}

    def _write_checkpoint(self, count: int, last_hmac: str, created_registry=None) -> None:
        payload = json.dumps(
            {"count": count, "last_hmac": last_hmac, "checkpoint_hmac": self._checkpoint_hmac(count, last_hmac)},
            sort_keys=True,
        )
        # A fresh name, created exclusively. The predictable `.checkpoint.tmp`
        # name was opened with O_TRUNC and no O_EXCL, so a regular file already
        # sitting there was truncated, filled with this call's bytes and then
        # renamed over the checkpoint. That is a pre-existing file destroyed by
        # a successful construction, which the specification says never happens
        # (independent review of the sixth pass, R6-F3). Exclusivity makes the
        # create the proof of ownership, as it already is for the store and the
        # key, and a name nothing else predicts cannot be occupied in advance.
        own_created: list[tuple[Path, tuple[int, int]]] = []
        tmp_path = self._checkpoint_path.with_name(
            f"{self._checkpoint_path.name}.tmp.{secrets.token_hex(8)}"
        )
        fd = _open_confined(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
            "receipt store checkpoint temporary file",
            created_registry=own_created,
        )
        # The temporary file is this call's responsibility until `os.replace`
        # gives it its final name, and the mode is set through the descriptor
        # before that, so there is no step after the rename that can fail and
        # leave a file behind (independent review of the fifth pass, R5-F3,
        # which found the temporary file surviving a replace failure and the
        # checkpoint surviving a chmod failure).
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # The rename does not change the inode, so the identity taken at
            # the exclusive create identifies the checkpoint afterwards. Taking
            # it again by `lstat` after the rename was another window in which
            # the file existed and nothing owned it (R6-F2, window 6).
            identity = own_created[0][1] if own_created else None
            os.replace(tmp_path, self._checkpoint_path)
            if created_registry is not None and identity is not None:
                created_registry.append((self._checkpoint_path, identity))
            own_created.clear()
        except BaseException:
            # The rename moves the inode between two names, and the obligation
            # to remove it cannot be handed from this cleanup to the caller's
            # registry in one step. So this cleanup covers both names for the
            # whole interval: whichever one holds the inode it created is the
            # one removed. Ordering the registration before the clear is not
            # enough on its own, because a failure can land between the rename
            # and the registration, which is where the checkpoint was being
            # stranded (independent review of the seventh pass, R7-F2).
            #
            # The final name is covered only for a caller that is building the
            # store set in one transaction, which is the caller that passes a
            # registry. An append has already written and synced the record
            # this checkpoint authenticates, and the replacement has already
            # displaced the checkpoint that authenticated the shorter log, so
            # removing the installed file there leaves a store with no anchor
            # and nothing to restore (independent review of the eighth pass,
            # R8-F1). After a committed append replacement the installed
            # checkpoint is the coherent state and stays.
            #
            # Both removals are inode-bound, so a failed rename, or a final
            # name now holding a different file, removes nothing.
            for created_path, created_id in reversed(own_created):
                if _unlink_if_ours(created_path, created_id):
                    continue
                if created_registry is not None:
                    _unlink_if_ours(self._checkpoint_path, created_id)
            raise

    def _verify_chain_detailed(self) -> tuple[bool, int, str]:
        """Recompute the HMAC chain and cross-check it against the checkpoint.

        Returns (valid, record_count, last_hmac_seen). Malformed JSON in
        either the log or the checkpoint is treated as invalid rather than
        raised, so a caller never sees an uncaught decoder exception.
        """
        prev_hmac = _GENESIS
        count = 0
        lines, lf_terminated = self._read_lines()
        if not lf_terminated:
            return False, count, prev_hmac
        try:
            for raw in lines:
                # A blank line is not a record. Skipping it here while the
                # reference verifier rejects it is what let the product path
                # report a store the reference verifier refused (R3-F1).
                if not raw.strip():
                    return False, count, prev_hmac
                count += 1
                record = json.loads(raw.decode("utf-8"))
                if not isinstance(record, dict):
                    return False, count, prev_hmac
                if not self._record_schema_is_valid(record):
                    return False, count, prev_hmac
                if not self._stored_bytes_are_canonical(raw, record):
                    return False, count, prev_hmac
                if record.get("prev_hmac") != prev_hmac:
                    return False, count, prev_hmac
                body = {k: v for k, v in record.items() if k != "hmac"}
                expected = self._record_hmac(prev_hmac, body)
                if not hmac_module.compare_digest(str(record.get("hmac", "")), expected):
                    return False, count, prev_hmac
                prev_hmac = record["hmac"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            return False, count, prev_hmac

        checkpoint = self._read_checkpoint()
        if checkpoint is None:
            return False, count, prev_hmac
        if not isinstance(checkpoint, dict):
            return False, count, prev_hmac
        if not self._checkpoint_shape_is_valid(checkpoint):
            return False, count, prev_hmac
        if checkpoint.get("count") != count or checkpoint.get("last_hmac") != prev_hmac:
            return False, count, prev_hmac
        expected_checkpoint_hmac = self._checkpoint_hmac(checkpoint.get("count"), checkpoint.get("last_hmac"))
        if not hmac_module.compare_digest(str(checkpoint.get("checkpoint_hmac", "")), expected_checkpoint_hmac):
            return False, count, prev_hmac
        return True, count, prev_hmac

    @staticmethod
    def _stored_bytes_are_canonical(raw: bytes, record: dict) -> bool:
        """Compare the bytes on disk against the canonical form of the parsed record.

        Records written before canonical-byte enforcement carry no
        `spec_version`, and that absence is authenticated: the field is inside
        the HMAC body, so it cannot be removed from a conforming record
        without breaking the chain. Those records are exempt from this check
        alone, which keeps a store written by an earlier build readable and
        extendable instead of failing closed on bytes it was never asked to
        produce. Every other rule still applies to them.
        """
        if "spec_version" not in record:
            return True
        return raw == canonical_bytes(record)

    @staticmethod
    def _checkpoint_shape_is_valid(checkpoint: dict) -> bool:
        """Validate the anchor's field shapes before its values are compared.

        `True == 1` in Python, so a checkpoint carrying `"count": true`
        compares equal to a one-record store (R3-F1, and F1 before it in the
        reference verifier).
        """
        count = checkpoint.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False
        for field in ("last_hmac", "checkpoint_hmac"):
            value = checkpoint.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                return False
        return True

    @staticmethod
    def _record_schema_is_valid(record: dict) -> bool:
        """Validate known stored schema contracts without rewriting history.

        Versions 3, 4 and 5 carry the same source-binding requirement and
        differ only in what `runtime_profile` means and, at version 5, in what
        the emitter validates before writing, so all three remain readable and
        all three are checked the same way here. Only version 5 may be
        written; that is enforced by `_validate_schema_fields`, not by this
        reader. A legacy record stays eligible for integrity checking under its
        historical rules and is never rewritten.
        """
        schema_version = record.get("schema_version")
        if schema_version == "2":
            return (
                record.get("runtime_source_digest") is None
                and record.get("runtime_source_scope") is None
            )
        if schema_version not in READABLE_SCHEMA_VERSIONS:
            return False
        digest = record.get("runtime_source_digest")
        return (
            isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest) is not None
            and record.get("runtime_source_scope") == RUNTIME_SOURCE_SCOPE
        )

    # Core fields the record declares as `int | None`. A non-integer here is a
    # type defect in the caller, not a portability problem to be converted
    # away: silently rendering it as a decimal string would change the field's
    # meaning and hide the defect from every later reader
    # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D6).
    _INTEGER_FIELDS = ("input_tokens", "output_tokens", "latency_us")

    def _validate_schema_fields(self, receipt: Receipt) -> None:
        """Fail closed unless a new receipt satisfies the current schema.

        Historical version 2, 3 and 4 records remain readable through chain
        verification. Current writes require version 5, a 64-character
        lowercase SHA-256 `runtime_source_digest` and exactly the declared
        `runtime_source_scope`. Every field the dataclass declares must match
        its declared type; `latency_us` is microseconds. Validation precedes
        any receipt or checkpoint change, and precedes float canonicalisation.
        """
        for field in self._INTEGER_FIELDS:
            value = getattr(receipt, field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ReceiptPersistenceError(
                    f"{field} must be an integer or null",
                    request_id=receipt.request_id,
                    field=field,
                    observed_type=type(value).__name__,
                )

        if receipt.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ReceiptPersistenceError(
                "new receipts require the current schema version",
                request_id=receipt.request_id,
                expected_schema_version=RECEIPT_SCHEMA_VERSION,
                observed_schema_version=receipt.schema_version,
            )
        digest = receipt.runtime_source_digest
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ReceiptPersistenceError(
                "a current-schema receipt requires a 64-character lowercase runtime_source_digest",
                request_id=receipt.request_id,
            )
        if receipt.runtime_source_scope != RUNTIME_SOURCE_SCOPE:
            raise ReceiptPersistenceError(
                "a current-schema receipt requires the declared runtime_source_scope",
                request_id=receipt.request_id,
            )

        self._validate_declared_types(receipt)

        for name in IMPLEMENTATION_DEFINED_BLOCKS:
            _reject_non_finite(getattr(receipt, name), receipt.request_id, name)

    @staticmethod
    def _validate_declared_types(receipt: Receipt) -> None:
        """Reject any field whose value does not match its declared type.

        Validating only the integer fields left every declared string and
        boolean field to `_canonicalise_floats`, which rewrote a float
        `request_id` as the string `'3.5'` and stored it
        (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F4). Conversion
        is for values this implementation holds no schema for. A field the
        dataclass declares is not one of those.
        """
        for name, (allowed, optional) in _receipt_field_types().items():
            value = getattr(receipt, name)
            if value is None:
                if optional:
                    continue
                raise ReceiptPersistenceError(
                    f"{name} must not be null",
                    request_id=receipt.request_id,
                    field=name,
                )
            # bool is a subclass of int, so an int field accepts True unless
            # the check rejects it before the isinstance test.
            if isinstance(value, bool) and bool not in allowed:
                raise ReceiptPersistenceError(
                    f"{name} must be {_type_names(allowed)}, not bool",
                    request_id=receipt.request_id,
                    field=name,
                    observed_type="bool",
                )
            if not isinstance(value, allowed):
                raise ReceiptPersistenceError(
                    f"{name} must be {_type_names(allowed)}",
                    request_id=receipt.request_id,
                    field=name,
                    observed_type=type(value).__name__,
                )

    def _reject_legacy_generation(self, count: int) -> None:
        """Refuse a current-schema append into a store holding legacy records.

        A store may hold one field contract. Appending a version 5 record to a
        store whose earlier records were written under version 2, 3 or 4 would
        put two contracts under one chain and make the store's conformance
        level a per-record property that no single claim describes. The legacy
        record is evidence and is never rewritten, so the append is refused and
        the operator is told which store to leave alone and what to run
        instead. Nothing is written: this runs before the first store byte and
        before the checkpoint is replaced.
        """
        for existing in self._iter_records():
            version = existing.get("schema_version")
            if version != RECEIPT_SCHEMA_VERSION:
                raise ReceiptMigrationRequiredError(
                    "this receipt store holds records written under an earlier schema "
                    f"version ({version!r}); it is historical evidence and is not rewritten. "
                    "Run `plag-in receipt --initialise-current-store` to create a separate "
                    "current-schema store beside it, then serve against that store.",
                    legacy_store_path=str(self.path),
                    legacy_schema_version=version,
                    required_schema_version=RECEIPT_SCHEMA_VERSION,
                    record_count=count,
                    operator_action="plag-in receipt --initialise-current-store",
                )

    def append(self, receipt: Receipt) -> dict:
        self._validate_schema_fields(receipt)
        body = asdict(receipt)
        # Every string value and object key is walked for lone surrogates
        # before any encoding is attempted, so an unencodable value produces a
        # named refusal rather than a `UnicodeEncodeError` traceback out of the
        # canonical encoder (work package A5).
        _reject_surrogates(body, receipt.request_id)
        # Conversion is confined to the blocks this implementation declares as
        # implementation-defined. Every other field has just been validated
        # against its declared type, so no float can reach the store through
        # one of them (F4).
        for name in IMPLEMENTATION_DEFINED_BLOCKS:
            if body.get(name) is not None:
                body[name] = _canonicalise_floats(body[name])
        body["integrity_mode"] = INTEGRITY_MODE

        with self._locked():
            valid, count, prev_hmac = self._verify_chain_detailed()
            if not valid:
                raise ReceiptChainError(
                    "receipt store failed integrity verification; refusing to extend it", record_count=count
                )

            self._reject_legacy_generation(count)

            # Section 2 requires request_id to be unique within the emitting
            # store. The reference verifier rejects a duplicate after the fact;
            # the emitter must not create one in the first place, and must not
            # write a byte or touch the checkpoint when it would
            # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_REPAIR_2026-08-26.md F6).
            for existing in self._iter_records():
                if existing.get("request_id") == receipt.request_id:
                    raise ReceiptPersistenceError(
                        "request_id is already present in this store",
                        request_id=receipt.request_id,
                        record_count=count,
                    )

            record = dict(body)
            record["prev_hmac"] = prev_hmac
            record["hmac"] = self._record_hmac(prev_hmac, record)
            # The stored line is the canonical serialisation, not a
            # convenience rendering of it: spec section 4 makes the stored
            # bytes normative, so a second implementation reading this store
            # must see the same bytes this one wrote
            # (CODEX_REVIEW_RECEIPT_SPEC_V0_1_2026-08-26.md D5). Binary append
            # keeps the LF terminator from being translated on a CRLF platform.
            # Backstop for A5: `_reject_surrogates` runs before the lock and
            # should already have refused anything unencodable, so reaching
            # this handler means the walk missed a case. It is caught rather
            # than left to propagate, because a `UnicodeEncodeError` escaping
            # here is an unnamed traceback at exactly the point where nothing
            # has been written yet and a named refusal is still possible.
            try:
                line = canonical_line(record)
            except UnicodeEncodeError as exc:
                raise ReceiptPersistenceError(
                    "receipt could not be encoded as UTF-8; no store byte was written",
                    request_id=receipt.request_id,
                    reason=str(exc),
                ) from exc
            try:
                with self.path.open("ab") as handle:
                    handle.write(line)
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


def _create_exclusive(path: Path, data: bytes) -> None:
    """Create `path` with mode 0600, failing if anything already occupies it."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _file_identity(path: Path) -> tuple[int, int]:
    """The device and inode that identify one file, independent of its name."""
    info = os.lstat(path)
    return (info.st_dev, info.st_ino)


def _link_into_place(staged: Path, final: Path) -> None:
    """Give a staged file its final name, failing if anything holds that name.

    `os.link` is the exclusive step: it refuses when the destination exists,
    including when the destination is a symbolic link, and it does not follow
    one. The linked name refers to the same inode as the staged name, which is
    what lets cleanup prove the file it removes is the one this call created.
    """
    os.link(staged, final)


def _unlink_if_ours(path: Path, identity: tuple[int, int]) -> bool:
    """Remove `path` only while it is still the file that `identity` names.

    Ownership is established by inode rather than by a list of pathnames. A
    pathname recorded before its creation attempt can be created by another
    process in the interval, and removing it on that ground deletes a file this
    call never made (independent review of the work package A repair). An inode
    recorded from a file this call created inside its own private directory
    cannot belong to anyone else.

    The declared boundary. This is a check followed by a removal, and Linux
    offers no unlink that takes an inode predicate, so the two steps cannot be
    made one.

    There are now two classes of caller, and they do not have the same premise
    (independent review of the third pass, T3-F3, which found this docstring
    describing only the first).

    Initialisation and rollback call this while holding the initialisation
    transaction lock. There the guarantee is against another PLAG IN process,
    because every such process takes that lock before touching these names.

    `ReceiptStore.__init__` calls it without that lock, to remove a store its
    own failed construction created. Its evidence is narrower and does not
    depend on the lock: the identity passed in comes from a descriptor this
    call obtained by winning an exclusive create, so the inode named is one
    that did not exist until this call made it, and no other party has a
    pathname for it except by having observed the name afterwards. If the name
    has since been replaced, the identity check fails and nothing is removed.

    In both classes, a process that does not take the lock and replaces a
    pathname between the check and the removal is outside the threat model.
    """
    try:
        if _file_identity(path) != identity:
            return False
    except OSError:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


INIT_TRANSACTION_LOCK_FILENAME = "receipts.init.lock"


@contextlib.contextmanager
def _initialisation_transaction(directory: Path):
    """Serialise initialisation and rollback across PLAG IN processes.

    Both operations create and remove several files under names they do not
    hold open, so without a lock two of them interleave and each sees the
    other's files under names it planned to use. This is the boundary the
    implementation claims: safety among cooperating PLAG IN processes, which
    all take this lock before touching a current-store name.

    The lock file is persistent and is never removed. A zero-byte mutex that
    nothing deletes raises no ownership question, which is the defect that
    removing the receipt store's own lock created.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INIT_TRANSACTION_LOCK_FILENAME
    # The lock is taken before any confinement check the caller performs on the
    # store, so it is the first name in the transaction an attacker can reach.
    # Opening it without O_NOFOLLOW created a regular file at the far end of a
    # planted link and then completed store creation, which is exactly the
    # escape the pointer rules refuse everywhere else (second pass, D1).
    fd = _open_confined(
        path, os.O_CREAT | os.O_RDWR, 0o600, "initialisation transaction lock"
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def initialise_current_store(store_path: Path, hmac_key_path: Path | None = None) -> dict:
    """Create a fresh current-schema store, key and genesis checkpoint.

    The operator command behind `plag-in receipt --initialise-current-store`.
    A legacy store is historical evidence, so it is neither rewritten nor
    migrated: a separate current-schema store is created beside it and the
    legacy files are not opened for writing at any point here.

    Every destination is created exclusively, so an existing store, key or
    checkpoint is refused rather than overwritten. If any step fails, the files
    this call created are removed before it returns, leaving the directory as
    it was found; the caller activates the store only after this returns.

    Each file is built inside a private staging directory and then hard-linked
    into place, so ownership is a property of the file rather than of a
    pathname this call intends to use. Cleanup removes a destination only while
    it is still the inode that was staged, so a file created by another process
    under a name this call planned to use is never removed by it.

    The whole transaction runs under the initialisation lock of
    `_initialisation_transaction`, which is the boundary this claim rests on:
    it holds among PLAG IN processes. The receipt store's own lock file is
    never removed here, whatever the outcome. It is a zero-byte mutex another
    process may be holding open, and removing it would split future users of
    that store across two lock inodes, which is worse than leaving an empty
    file behind (independent review of pass 2, D6).
    """
    store_path = Path(store_path)
    key_path = (
        Path(hmac_key_path) if hmac_key_path is not None
        else store_path.with_name(store_path.name + ".hmac_key")
    )
    checkpoint_path = store_path.with_name(store_path.name + ".checkpoint")

    store_path.parent.mkdir(parents=True, exist_ok=True)

    with _initialisation_transaction(store_path.parent):
        for role, candidate in (
            ("store", store_path),
            ("HMAC key", key_path),
            ("checkpoint", checkpoint_path),
        ):
            if candidate.exists() or candidate.is_symlink():
                raise ReceiptPersistenceError(
                    f"refusing to initialise over an existing {role}",
                    path=str(candidate),
                    role=role,
                )

        key = os.urandom(_HMAC_KEY_BYTES)
        message = f"checkpoint:0:{_GENESIS}".encode("utf-8")
        checkpoint_hmac = hmac_module.new(key, message, hashlib.sha256).hexdigest()
        checkpoint_payload = json.dumps(
            {"count": 0, "last_hmac": _GENESIS, "checkpoint_hmac": checkpoint_hmac},
            sort_keys=True,
        ).encode("utf-8")

        try:
            staging = Path(
                tempfile.mkdtemp(prefix=".plag-in-init-", dir=str(store_path.parent))
            )
        except OSError as exc:
            raise ReceiptPersistenceError(
                f"a staging directory for the new store could not be created: {exc}",
                path=str(store_path.parent),
            ) from exc

        # Destination, its identity as staged, and its role. Recorded before
        # any attempt on the destination name, because the identity is what
        # cleanup tests, not the name.
        planned: list[tuple[str, Path, tuple[int, int]]] = []
        try:
            for role, destination, payload in (
                ("HMAC key", key_path, key),
                ("store", store_path, b""),
                ("checkpoint", checkpoint_path, checkpoint_payload),
            ):
                staged = staging / destination.name
                _create_exclusive(staged, payload)
                # The staging directory was created by this call and is named
                # unpredictably, so a file inside it is this call's by
                # construction. That is what makes the identity below evidence
                # of ownership rather than of intent.
                planned.append((role, destination, _file_identity(staged)))

            for role, destination, _identity in planned:
                try:
                    _link_into_place(staging / destination.name, destination)
                except FileExistsError as exc:
                    # The pre-check refused every destination that existed when
                    # it ran; this is the same refusal for a name that was
                    # taken afterwards, and the file that took it is left alone.
                    raise ReceiptPersistenceError(
                        f"refusing to initialise over an existing {role}",
                        path=str(destination),
                        role=role,
                    ) from exc
                except OSError as exc:
                    raise ReceiptPersistenceError(
                        "the state directory does not support the hard link this "
                        f"transaction requires: {exc}",
                        path=str(destination),
                        role=role,
                    ) from exc

            store = ReceiptStore(store_path, hmac_key_path=key_path)
            valid, count = store.verify_chain()
            if not valid or count != 0:
                raise ReceiptPersistenceError(
                    "a freshly initialised store did not verify; nothing was activated",
                    path=str(store_path),
                    record_count=count,
                )
        except BaseException:
            for _role, destination, identity in reversed(planned):
                _unlink_if_ours(destination, identity)
            # The store's own lock file is deliberately left where it is. See
            # the docstring: an empty file nothing removes cannot be the wrong
            # file to remove.
            shutil.rmtree(staging, ignore_errors=True)
            raise

        # The staged names are additional names for the same inodes; removing
        # them leaves each destination as the only name for its file.
        shutil.rmtree(staging, ignore_errors=True)

        return {
            "store_path": str(store_path),
            "hmac_key_path": str(key_path),
            "checkpoint_path": str(checkpoint_path),
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "record_count": 0,
            # Ownership evidence for `rollback_current_store`, not operator
            # output.
            "created_identities": {
                role_key: list(identity)
                for role_key, identity in (
                    ("hmac_key_path", planned[0][2]),
                    ("store_path", planned[1][2]),
                    ("checkpoint_path", planned[2][2]),
                )
            },
        }


def rollback_current_store(created: dict) -> list[str]:
    """Remove a store that `initialise_current_store` built but nothing activated.

    Initialisation and activation are two steps: the store, its key and its
    checkpoint are created first, and the pointer that makes them the active
    store is written second. A failure in the second step used to leave the
    complete unactivated set on disk, so the operator was told the command had
    failed and found a store anyway (independent review of work package A, D2).

    Returns the paths removed. An empty store holds no evidence, so removing it
    loses nothing; a store that has acquired bytes is no longer the unactivated
    store this was given and is refused rather than deleted.

    Each removal is checked against the file identity `initialise_current_store`
    recorded, so a destination that has since become another process's file is
    left where it is. A dict carrying no identities removes nothing and says so,
    rather than falling back to removal by pathname.

    Runs under the same initialisation lock as `initialise_current_store`, and
    carries the same boundary: the guarantee holds among PLAG IN processes.

    The store's lock file is not removed. It is a zero-byte mutex another
    process may hold open, its identity was never recorded, and removing it
    would split later users of that store across two lock inodes. An empty file
    left in the state directory costs nothing by comparison (independent review
    of pass 2, D6).
    """
    store_path = Path(created["store_path"])

    with _initialisation_transaction(store_path.parent):
        try:
            size = store_path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size:
            raise ReceiptPersistenceError(
                "refusing to remove a receipt store that already holds bytes",
                path=str(store_path),
                size_bytes=size,
            )

        identities = created.get("created_identities")
        if not identities:
            raise ReceiptPersistenceError(
                "refusing to roll back without the file identities that prove ownership",
                path=str(store_path),
            )

        removed: list[str] = []
        for key in ("checkpoint_path", "store_path", "hmac_key_path"):
            identity = identities.get(key)
            if identity is None:
                continue
            path = Path(created[key])
            if _unlink_if_ours(path, tuple(identity)):
                removed.append(str(path))
        return removed
