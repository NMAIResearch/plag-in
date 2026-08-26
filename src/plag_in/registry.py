"""Stable aliases bound to exact model, engine and configuration digests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plag_in.aliasing import validate_alias
from plag_in.errors import ModelIdentityMismatchError
from plag_in.identity import ModelIdentity


@dataclass(frozen=True)
class RegistryEntry:
    alias: str
    identity: ModelIdentity
    engine: str
    engine_config_digest: str


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def bind(self, entry: RegistryEntry) -> None:
        validate_alias(entry.alias)
        self._entries[entry.alias] = entry

    def resolve(self, alias: str) -> RegistryEntry:
        try:
            return self._entries[alias]
        except KeyError as exc:
            raise ModelIdentityMismatchError(f"alias not registered: {alias}", alias=alias) from exc

    def verify_identity(self, alias: str) -> RegistryEntry:
        """Re-hash the bound file and refuse if it no longer matches the registered digest."""
        entry = self.resolve(alias)
        current = ModelIdentity.from_file(Path(entry.identity.path))
        if current.sha256 != entry.identity.sha256:
            raise ModelIdentityMismatchError(
                "model file hash changed since registration",
                alias=alias,
                expected=entry.identity.sha256,
                actual=current.sha256,
            )
        return entry

    def aliases(self) -> list[str]:
        return sorted(self._entries)
