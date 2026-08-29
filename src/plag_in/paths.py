"""State directory layout. Callers always pass an explicit base directory;
there is no implicit write outside what the caller names."""
from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from plag_in.aliasing import validate_alias
from plag_in.confinement import (
    ensure_state_dir,
    open_state_file,
    read_to_end,
    unlink_if_ours,
)
from plag_in.errors import ConfigurationError
from plag_in.receipts import CURRENT_STORE_FILENAME, RECEIPT_SCHEMA_VERSION

# A backend key is 64 hexadecimal characters and a newline. The read bound is
# far larger so that a wrong file is refused for what it is rather than
# truncated to a prefix that happens to parse.
_BACKEND_KEY_HEX_CHARS = 64
_BACKEND_KEY_READ_LIMIT = 4096


@dataclass(frozen=True)
class ActiveStoreSnapshot:
    """One reading of the pointer, and every path derived from that reading.

    The store, its key and its checkpoint are one selection, not three
    independent lookups. Reading the pointer once per path let a normal
    concurrent activation land between two of those reads and hand back a
    legacy store with a current-schema key, which either fails integrity or
    authenticates records under the wrong key (independent review of work
    package A, second pass, D3). Anything that needs more than one of these
    paths takes a snapshot and uses its fields.

    A snapshot is a reading, not a lease: it says what the pointer named at the
    moment it was read, and a later activation is not visible in it. That is
    the point. A caller writing a receipt uses one consistent set of paths for
    that write, and takes a fresh snapshot for the next one.
    """

    store: Path
    hmac_key: Path
    checkpoint: Path
    lock: Path
    schema_version: str | None
    is_legacy: bool


@dataclass(frozen=True)
class StateLayout:
    base: Path

    @property
    def sessions_dir(self) -> Path:
        return self.base / "sessions"

    @property
    def receipts_file(self) -> Path:
        return self.base / "receipts.jsonl"

    @property
    def receipts_hmac_key_file(self) -> Path:
        return self.base / "receipts.jsonl.hmac_key"

    @property
    def engine_keys_dir(self) -> Path:
        return self.base / "engine_keys"

    # -- receipt store roles -------------------------------------------
    #
    # `receipts_file` is the original store path. Once a current-schema store
    # has been initialised beside it, that store's path is recorded here, in a
    # separate pointer file, rather than by renaming or replacing anything: the
    # legacy store keeps its own path and stays byte-identical.

    @property
    def receipt_pointer_file(self) -> Path:
        return self.base / "receipts.active.json"

    @property
    def legacy_receipts_file(self) -> Path:
        return self.receipts_file

    @staticmethod
    def _refuse_symlink(path: Path, role: str):
        """Return the path's own stat, refusing a symbolic link at that name.

        `lstat` rather than `exists()`: a dangling symlink makes `exists()`
        false, so a pointer file replaced by a link to a missing target would
        read as an absent pointer and the caller would fall back to the legacy
        store, which is the opposite of failing closed. Returns None when
        nothing occupies the name.
        """
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ConfigurationError(
                f"the {role} could not be inspected",
                path=str(path),
                reason=str(exc),
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ConfigurationError(
                f"the {role} is a symbolic link; the receipt store set is regular files only",
                path=str(path),
                role=role,
            )
        return info

    def validate_active_store_path(self, store_path) -> Path:
        """Return the absolute active-store path, or refuse it.

        The active store is confined: a regular, non-symlink file, directly
        beneath this state directory, under the current-schema filename. The
        pointer is an ordinary file an operator or anything running as the
        operator can edit, so nothing about a path it names may be taken on
        trust. Without confinement an edited pointer redirects receipt creation
        and appends to a path the product never declared
        (independent review of work package A, D1).

        The files the store owns beside it are checked for the same thing: a
        symlinked key or checkpoint routes the store's own state elsewhere
        without touching the store path at all. `normpath` rather than
        `resolve`: resolution follows symlinks, which would accept exactly the
        indirection this refuses.
        """
        candidate = Path(os.path.abspath(str(store_path)))
        base = Path(os.path.abspath(str(self.base)))
        if candidate.name != CURRENT_STORE_FILENAME:
            raise ConfigurationError(
                "the active receipt store must carry the current-schema filename",
                path=str(candidate),
                expected_filename=CURRENT_STORE_FILENAME,
            )
        if candidate.parent != base:
            raise ConfigurationError(
                "the active receipt store must sit directly beneath the state directory",
                path=str(candidate),
                state_directory=str(base),
            )
        info = self._refuse_symlink(candidate, "active receipt store")
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ConfigurationError(
                "the active receipt store is not a regular file",
                path=str(candidate),
            )
        for suffix, role in (
            (".hmac_key", "active store HMAC key"),
            (".checkpoint", "active store checkpoint"),
            (".lock", "active store lock file"),
        ):
            self._refuse_symlink(candidate.with_name(candidate.name + suffix), role)
        return candidate

    def read_receipt_pointer(self) -> dict | None:
        """Return the recorded active-store pointer, or None when unset.

        A malformed, escaping, symlinked or unresolvable pointer is an error
        rather than an absence: silently falling back to the legacy store would
        serve a different store from the one the operator activated, without
        saying so. Only a name nothing occupies is an absence.
        """
        path = self.receipt_pointer_file
        self._refuse_symlink(path, "active receipt-store pointer")
        try:
            # O_NOFOLLOW closes the window between the lstat above and this
            # open; the lstat is what distinguishes a dangling link from an
            # absent file, and this is what makes the refusal race-free.
            fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ConfigurationError(
                "the active receipt-store pointer could not be opened",
                path=str(path),
                reason=str(exc),
            ) from exc
        try:
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ConfigurationError(
                        "the active receipt-store pointer is not a regular file",
                        path=str(path),
                    )
                raw = handle.read()
        except OSError as exc:
            raise ConfigurationError(
                "the active receipt-store pointer could not be read",
                path=str(path),
                reason=str(exc),
            ) from exc

        try:
            pointer = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigurationError(
                "the active receipt-store pointer could not be read",
                path=str(path),
                reason=str(exc),
            ) from exc
        if not isinstance(pointer, dict) or not isinstance(pointer.get("current_store"), str):
            raise ConfigurationError(
                "the active receipt-store pointer does not name a current store",
                path=str(path),
            )
        if pointer.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise ConfigurationError(
                "the active receipt-store pointer does not declare the current schema version",
                path=str(path),
                declared_schema_version=pointer.get("schema_version"),
                expected_schema_version=RECEIPT_SCHEMA_VERSION,
            )

        current = self.validate_active_store_path(pointer["current_store"])
        if not current.exists():
            raise ConfigurationError(
                "the active receipt-store pointer names a store that does not exist",
                path=str(path),
                current_store=str(current),
            )
        pointer["current_store"] = str(current)
        return pointer

    def write_receipt_pointer(self, store_path: Path, schema_version: str) -> dict:
        """Record which store new receipts are written to, atomically.

        The recorded path is validated before it is written, so a pointer this
        state directory would refuse to read is never produced. The temporary
        file is created exclusively under a fresh name and refuses to follow a
        link, so neither a planted temporary nor a symlinked pointer name can
        route the write outside the state directory (D1).

        Existence of the store is part of that validation, not a separate
        concern. The writer previously reported success for a store that was
        not there, and the very next read refused the pointer it had just
        written, which is a contract this class states and then breaks
        (independent review of work package A, second pass, implementation
        contract). The two sides now agree: what this writer produces, the
        reader accepts, for a store set that does not change between the check
        and the commit. Those are two operations, so a store removed in that
        window leaves a committed pointer the next read refuses (independent
        review of the second pass, F3). The window is the same one the
        cooperating-process boundary already excludes, and it is recorded here
        rather than described as an agreement without conditions.
        """
        current = self.validate_active_store_path(store_path)
        if not current.exists():
            raise ConfigurationError(
                "the active receipt-store pointer names a store that does not exist",
                path=str(self.receipt_pointer_file),
                current_store=str(current),
            )
        if schema_version != RECEIPT_SCHEMA_VERSION:
            raise ConfigurationError(
                "the active receipt-store pointer records the current schema version only",
                declared_schema_version=schema_version,
                expected_schema_version=RECEIPT_SCHEMA_VERSION,
            )
        pointer = {
            "current_store": str(current),
            "legacy_store": str(self.legacy_receipts_file),
            "schema_version": schema_version,
        }
        path = self.receipt_pointer_file
        path.parent.mkdir(parents=True, exist_ok=True)
        info = self._refuse_symlink(path, "active receipt-store pointer")
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ConfigurationError(
                "the active receipt-store pointer is not a regular file",
                path=str(path),
            )
        tmp_path = path.with_name(f"{path.name}.tmp.{secrets.token_hex(8)}")
        fd = os.open(
            str(tmp_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(pointer, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        os.chmod(path, 0o600)
        return pointer

    def active_store_snapshot(self) -> ActiveStoreSnapshot:
        """Resolve the active store, its key and its checkpoint from one read.

        This is the accessor to use whenever more than one of those paths is
        wanted. The pointer is read exactly once here, so the returned set
        cannot straddle an activation: it names the legacy store with the
        legacy key, or the current store with the current key, and never one of
        each (D3).
        """
        pointer = self.read_receipt_pointer()
        if pointer is None:
            store = self.receipts_file
            hmac_key = self.receipts_hmac_key_file
            schema_version = None
            is_legacy = True
        else:
            store = Path(pointer["current_store"])
            hmac_key = store.with_name(store.name + ".hmac_key")
            schema_version = pointer.get("schema_version")
            is_legacy = False
        return ActiveStoreSnapshot(
            store=store,
            hmac_key=hmac_key,
            checkpoint=store.with_name(store.name + ".checkpoint"),
            lock=store.with_name(store.name + ".lock"),
            schema_version=schema_version,
            is_legacy=is_legacy,
        )

    @property
    def active_receipts_file(self) -> Path:
        """The store new receipts are appended to, as at this call.

        The legacy store until a current-schema store has been activated, and
        that store afterwards.

        Reading this and `active_receipts_hmac_key_file` as a pair is the
        defect they were rewritten to remove: each reads the pointer again, so
        the two answers can come from different pointer states. Call
        `active_store_snapshot()` when both are wanted. This property remains
        for callers that genuinely need the store path alone.
        """
        return self.active_store_snapshot().store

    @property
    def active_receipts_hmac_key_file(self) -> Path:
        """The key of the active store, as at this call. See the warning above."""
        return self.active_store_snapshot().hmac_key

    def receipt_store_roles(self) -> dict:
        """Report every receipt store this state directory holds, and its role.

        Both paths are always reported, so an operator reading status can see
        that the legacy store still exists and is no longer being written to,
        rather than inferring it from the absence of a line.
        """
        pointer = self.read_receipt_pointer()
        legacy = self.legacy_receipts_file
        roles = {
            "legacy_store": {
                "path": str(legacy),
                "exists": legacy.exists(),
                "role": "historical evidence; readable and integrity-checkable, never written",
            },
            "current_store": None,
            "active_store": str(legacy),
        }
        if pointer is not None:
            current = Path(pointer["current_store"])
            roles["current_store"] = {
                "path": str(current),
                "exists": current.exists(),
                "schema_version": pointer.get("schema_version"),
                "role": "current schema; new receipts are appended here",
            }
            roles["active_store"] = str(current)
        else:
            roles["legacy_store"]["role"] = (
                "the only store; new receipts are appended here until a current-schema "
                "store is initialised"
            )
        return roles

    def backend_api_key_file(self, alias: str) -> Path:
        validate_alias(alias)
        return self.engine_keys_dir / f"{alias}.key"

    def load_or_create_backend_api_key(self, alias: str) -> tuple[Path, str]:
        """Return this alias's backend key, creating it on first use.

        The route opened the key by name at every step: the directory was
        created with `exist_ok`, which accepts a symbolic link pointing
        anywhere; the mode was set by name on both the directory and the key,
        which reaches the link's target; an existing name was adopted without
        being inspected, so a link, FIFO or directory was used as the key; and
        the read was an unbounded `read_text` that follows a link and blocks on
        a FIFO. Every one of those is refused here under the receipt store's
        rules, in `plag_in.confinement`.

        Ownership is the other half. A failure while writing a key this call
        created used to leave an empty file behind, and because a pre-existing
        key is adopted, the next call read that empty file and refused it: one
        interrupted write disabled the alias permanently. The created inode is
        registered by the exclusive create and removed if any later step fails,
        and nothing this call did not create is ever removed.
        """
        path = self.backend_api_key_file(alias)
        ensure_state_dir(self.base, "state directory", parents=True)
        ensure_state_dir(self.engine_keys_dir, "engine key directory")
        role = "backend API key file"
        created: list[tuple[Path, tuple[int, int]]] = []
        try:
            fd = open_state_file(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                role,
                created_registry=created,
            )
        except FileExistsError:
            pass
        else:
            secret = secrets.token_hex(_BACKEND_KEY_HEX_CHARS // 2)
            try:
                handle = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                self._remove_created(created)
                raise
            try:
                with handle:
                    # Through the descriptor, so it cannot reach another file,
                    # and after the create so the umask cannot leave it looser.
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(secret + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                self._remove_created(created)
                raise
            return path, secret

        # The name was already occupied. What occupies it is inspected before
        # it is used, and its bytes are left exactly as they are.
        fd = open_state_file(path, os.O_RDONLY, 0o600, role)
        try:
            os.fchmod(fd, 0o600)
            raw = read_to_end(fd, _BACKEND_KEY_READ_LIMIT)
        except OSError as exc:
            raise ConfigurationError(
                f"backend API key file could not be read: {path}",
                path=str(path),
                reason=str(exc),
            ) from exc
        finally:
            os.close(fd)
        if len(raw) > _BACKEND_KEY_READ_LIMIT:
            raise ConfigurationError(
                f"the {role} is larger than the {_BACKEND_KEY_READ_LIMIT} byte maximum",
                path=str(path),
                role=role,
                maximum_bytes=_BACKEND_KEY_READ_LIMIT,
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError(
                f"backend API key file could not be read: {path}",
                path=str(path),
                reason=str(exc),
            ) from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 1 or len(lines[0]) != _BACKEND_KEY_HEX_CHARS:
            raise ConfigurationError(
                "backend API key file must contain exactly one 64-character local key",
                path=str(path),
            )
        return path, lines[0]

    @staticmethod
    def _remove_created(created: list[tuple[Path, tuple[int, int]]]) -> None:
        """Remove what this call created, in reverse, and only while it is ours."""
        for created_path, identity in reversed(created):
            unlink_if_ours(created_path, identity)

    def ensure(self) -> "StateLayout":
        ensure_state_dir(self.base, "state directory", parents=True)
        ensure_state_dir(self.sessions_dir, "sessions directory")
        ensure_state_dir(self.engine_keys_dir, "engine key directory")
        return self
