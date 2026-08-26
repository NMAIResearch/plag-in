"""One alias validator shared by registry, supervisor, CLI and receipt lookup.

An alias becomes part of a session-record filename and appears in receipts
and API scopes; it must never be treated as a trusted path component
without validation first (CODEX_REVIEW_MVP_2026-08-25.md F2).
"""
from __future__ import annotations

import re

from plag_in.errors import InvalidAliasError

_MAX_ALIAS_LENGTH = 64
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_alias(alias: str) -> str:
    if not isinstance(alias, str) or not alias:
        raise InvalidAliasError("alias must be a non-empty string")
    if len(alias) > _MAX_ALIAS_LENGTH:
        raise InvalidAliasError(f"alias exceeds maximum length of {_MAX_ALIAS_LENGTH}", alias=alias)
    if alias in (".", ".."):
        raise InvalidAliasError("alias must not be '.' or '..'", alias=alias)
    if "/" in alias or "\\" in alias:
        raise InvalidAliasError("alias must not contain a path separator", alias=alias)
    if not _ALIAS_RE.match(alias):
        raise InvalidAliasError(
            "alias must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}", alias=alias
        )
    return alias
