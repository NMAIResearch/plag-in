"""Confined creation, mode setting and reading for state files outside the receipt set.

The receipt store's rules were established and independently reviewed over nine
passes: refuse a symbolic link in the same system call that would follow it,
refuse a non-regular file without ever blocking on it, take ownership from an
exclusive create, set the mode through the descriptor rather than the name,
read to the end of the file rather than trusting one `read(2)`, and remove only
the inode the failing operation created.

The backend API key and the supervisor's session records are state files under
the same directory and the same threat model, and they were opened by name. This
module gives them the receipt rules rather than a second copy of them: the open
and the read are `receipts._open_confined` and `receipts.read_confined_bytes`,
called with the vocabulary these callers refuse in. A second implementation is
what let a repaired read rule survive unrepaired one function away (independent
review of the seventh pass, R7-F1).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from plag_in.errors import ConfigurationError
from plag_in.receipts import (
    _open_confined,
    _read_to_end,
    _unlink_if_ours,
    read_confined_bytes,
)

STATE_FILE_NOUN = "the PLAG IN state directory"

read_to_end = _read_to_end
unlink_if_ours = _unlink_if_ours


def open_state_file(
    path: Path,
    flags: int,
    mode: int,
    role: str,
    created_registry=None,
    error_type: type = ConfigurationError,
) -> int:
    """Open a state file under the receipt store's confinement rules."""
    return _open_confined(
        path,
        flags,
        mode,
        role,
        created_registry,
        error_type=error_type,
        set_noun=STATE_FILE_NOUN,
    )


def read_state_file(
    path: Path,
    role: str,
    limit: int,
    error_type: type = ConfigurationError,
) -> bytes:
    """Read a state file whole, refusing one larger than `limit`."""
    return read_confined_bytes(
        path,
        role,
        limit,
        error_type=error_type,
        set_noun=STATE_FILE_NOUN,
    )


_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK


def _step_into(
    parent_fd: int,
    name: str,
    prefix: Path,
    role: str,
    may_create: bool,
    create_mode: int | None,
    error_type: type,
) -> int:
    """Open one path component relative to `parent_fd`, never following a link.

    `O_NOFOLLOW` refuses a symbolic link at this component with `ELOOP`, and
    `O_DIRECTORY` refuses anything that is not a directory with `ENOTDIR`.
    Because the open is relative to a descriptor, the component cannot be
    reinterpreted between the check and the use: the parent this opens from is
    the parent already proven link-free, not a name resolved again.
    """
    while True:
        try:
            return os.open(name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if not may_create:
                raise
            try:
                if create_mode is None:
                    os.mkdir(name, dir_fd=parent_fd)
                else:
                    os.mkdir(name, create_mode, dir_fd=parent_fd)
            except FileExistsError:
                # Another process created it in this interval. Open what is
                # there under the same refusals rather than assume it is ours.
                continue
            except OSError as exc:
                raise error_type(
                    f"the {role} could not be created",
                    path=str(prefix),
                    role=role,
                    component=name,
                    reason=str(exc),
                ) from exc
            continue
        except OSError as exc:
            raise error_type(
                f"a path component of the {role} is a symbolic link or not a directory; "
                "the product creates no state through one",
                path=str(prefix),
                role=role,
                component=name,
                reason=str(exc),
            ) from exc


def ensure_state_dir(
    path: Path,
    role: str,
    mode: int = 0o700,
    parents: bool = False,
    error_type: type = ConfigurationError,
) -> None:
    """Create a state directory if absent, then confirm and confine what is there.

    Every component of the declared path is walked from the filesystem root
    through directory descriptors, and each one is opened with `O_NOFOLLOW`, so
    no component is followed and none is resolved twice. The final directory's
    mode is set through the descriptor that proved it.

    The final component alone is not enough. `os.makedirs(path.parent)` followed
    a symbolic link at any ancestor and then created the state or sessions
    directory, the key directory and the key itself on the far side of it, so
    the product's state landed outside the directory the caller declared while
    every check on the final name passed (independent review of the state-file
    confinement packet, C1). Nothing above the declared path is trusted merely
    because it already exists.

    The consequence is stated rather than hidden: a declared path whose
    ancestors include a symbolic link is refused, and the refusal names the
    component. On a host where `/home` is a link to `/var/home`, a state path
    under `/home` is refused and the link-free path under `/var/home` is the one
    to declare.

    Ancestors this call has to create are created with the process umask, as
    `makedirs` did. Only the declared directory itself is given `mode`, because
    tightening a directory the operator owns above the state path is not this
    product's decision.
    """
    absolute = Path(os.path.abspath(str(path)))
    parts = absolute.parts
    if len(parts) < 2:
        raise error_type(
            f"the {role} may not be the filesystem root", path=str(absolute), role=role
        )
    try:
        fd = os.open(parts[0], _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise error_type(
            f"the {role} could not be resolved from the filesystem root",
            path=str(absolute),
            role=role,
            reason=str(exc),
        ) from exc
    last = len(parts) - 1
    try:
        for index in range(1, len(parts)):
            name = parts[index]
            final = index == last
            prefix = Path(*parts[: index + 1])
            child = _step_into(
                fd,
                name,
                prefix,
                role,
                may_create=final or parents,
                create_mode=mode if final else None,
                error_type=error_type,
            )
            os.close(fd)
            fd = child
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise error_type(
                f"the {role} is not a directory",
                path=str(absolute),
                role=role,
            )
        os.fchmod(fd, mode)
    finally:
        os.close(fd)
