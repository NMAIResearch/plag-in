"""State directory layout. Callers always pass an explicit base directory;
there is no implicit write outside what the caller names."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from plag_in.aliasing import validate_alias
from plag_in.errors import ConfigurationError


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

    def backend_api_key_file(self, alias: str) -> Path:
        validate_alias(alias)
        return self.engine_keys_dir / f"{alias}.key"

    def load_or_create_backend_api_key(self, alias: str) -> tuple[Path, str]:
        path = self.backend_api_key_file(alias)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            secret = secrets.token_hex(32)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(secret + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            raise ConfigurationError(f"backend API key file could not be read: {path}") from exc
        if len(lines) != 1 or len(lines[0]) != 64:
            raise ConfigurationError(
                "backend API key file must contain exactly one 64-character local key",
                path=str(path),
            )
        return path, lines[0]

    def ensure(self) -> "StateLayout":
        self.base.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.engine_keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.engine_keys_dir, 0o700)
        return self
