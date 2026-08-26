"""Process supervisor: launch, readiness, stop and stale-record recovery.

Every process is launched as an argument array with `shell=False`. An
executable must resolve to exactly the path configured in the engine
allowlist. A session record binds a PID to a kernel process-start-time
token (read from `/proc/<pid>/stat`) so a reused PID is never mistaken
for the process PLAG IN started, plus a SHA-256 digest of the argument
array and of the complete executable file (CODEX_REVIEW_MVP_2026-08-25.md
F5). A `Popen` handle for a process this instance started is retained and
reaped by a background thread so its terminal state is always collected
(F6); a session record from an earlier process is confirmed dead by
polling process identity rather than by `waitpid`, since only the actual
parent process may reap a child.

Alias lifecycle operations (start/status/stop) are serialised per alias
with an `fcntl.flock` on a dedicated lock file, so two threads or two
separate Linux processes racing to start the same alias cannot both
launch a process and leave the session record referring to only one of
them (CODEX_REVIEW_SECURITY_REPAIR_2026-08-25.md R6). `start()` holds that
lock for the full launch-and-ready-wait transaction, then refuses to
proceed if the existing record already names a live process; a record
whose identity is confirmed dead is cleared and replaced instead.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from plag_in.aliasing import validate_alias
from plag_in.config import is_loopback
from plag_in.errors import (
    AliasAlreadyRunningError,
    BackendUnavailableError,
    ExecutableNotAllowedError,
    LocalityPolicyError,
    PathContainmentError,
    PortConflictError,
    ProcessIdentityError,
)
from plag_in.identity import canonical_digest, hash_file


def _process_start_time_token(pid: int) -> str | None:
    """Read field 22 (starttime) of /proc/<pid>/stat; None if the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Field 2 (comm) may contain spaces/parens; split after the last ')'.
    tail = raw.rsplit(")", 1)[-1].split()
    return tail[19]  # starttime is field 22 overall, index 19 after comm


def check_port_available(host: str, port: int) -> None:
    if port == 0:
        return
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise PortConflictError(f"port already in use: {host}:{port}", host=host, port=port) from exc
    finally:
        probe.close()


@dataclass
class SessionRecord:
    alias: str
    pid: int
    executable: str
    executable_digest: str
    argv_digest: str
    start_time_token: str
    host: str
    port: int
    started_at: str

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionRecord":
        return cls(**data)


class Supervisor:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir).resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen] = {}
        self._processes_lock = threading.Lock()

    def _record_path(self, alias: str) -> Path:
        validate_alias(alias)
        candidate = (self.sessions_dir / f"{alias}.json").resolve()
        try:
            candidate.relative_to(self.sessions_dir)
        except ValueError as exc:
            raise PathContainmentError(
                "session record path escapes the sessions directory", alias=alias
            ) from exc
        return candidate

    @contextlib.contextmanager
    def _alias_lock(self, alias: str):
        """Serialise start/status/stop for one alias across threads and processes."""
        lock_path = self.sessions_dir / f"{alias}.alias_lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def load_record(self, alias: str) -> SessionRecord | None:
        path = self._record_path(alias)
        if not path.exists():
            return None
        return SessionRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_record(self, record: SessionRecord) -> None:
        path = self._record_path(record.alias)
        tmp_path = path.with_name(path.name + ".tmp")
        fd = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record.as_dict(), sort_keys=True))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        os.replace(tmp_path, path)

    def _clear_record(self, alias: str) -> None:
        path = self._record_path(alias)
        if path.exists():
            path.unlink()

    def start(
        self,
        alias: str,
        argv: list[str],
        allowlisted_executable: str,
        host: str,
        port: int,
        health_path: str,
        ready_timeout_s: float = 5.0,
    ) -> SessionRecord:
        validate_alias(alias)

        if not is_loopback(host):
            raise LocalityPolicyError(
                "refusing to launch an engine bound to a non-loopback address", host=host
            )

        resolved_argv0 = str(Path(argv[0]).resolve())
        resolved_allowlisted = str(Path(allowlisted_executable).resolve())
        if resolved_argv0 != resolved_allowlisted:
            raise ExecutableNotAllowedError(
                "executable is not on the configured allowlist",
                requested=resolved_argv0,
                allowed=resolved_allowlisted,
            )

        with self._alias_lock(alias):
            existing = self.load_record(alias)
            if existing is not None:
                if self._identity_matches(existing):
                    raise AliasAlreadyRunningError(
                        "alias already has a live process; refusing to start a second one",
                        alias=alias,
                        pid=existing.pid,
                    )
                # A stale record: its recorded identity is confirmed dead, so
                # it may be cleared before this alias is reused.
                self._clear_record(alias)
                with self._processes_lock:
                    self._processes.pop(alias, None)

            check_port_available(host, port)

            executable_digest = hash_file(Path(resolved_argv0))
            argv_digest = canonical_digest({"argv": argv})

            process = subprocess.Popen(  # noqa: S603 - argv is a fixed list, shell=False
                argv,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._processes_lock:
                self._processes[alias] = process
            # Reap in the background immediately so the terminal state is
            # always collected, whether or not `stop()` is ever called.
            threading.Thread(target=process.wait, daemon=True).start()

            start_token = _process_start_time_token(process.pid)
            record = SessionRecord(
                alias=alias,
                pid=process.pid,
                executable=resolved_argv0,
                executable_digest=executable_digest,
                argv_digest=argv_digest,
                start_time_token=start_token or "",
                host=host,
                port=port,
                started_at=str(time.time()),
            )
            self._write_record(record)

            try:
                self._wait_ready(host, port, health_path, ready_timeout_s)
            except BackendUnavailableError:
                self._terminate_and_reap(alias, process, record.pid)
                self._clear_record(alias)
                raise
            return record

    def _wait_ready(self, host: str, port: int, health_path: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        url = f"http://{host}:{port}{health_path}"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=0.5) as resp:  # noqa: S310
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = exc
            time.sleep(0.05)
        raise BackendUnavailableError(f"engine did not become ready: {url}") from last_error

    def _identity_matches(self, record: SessionRecord) -> bool:
        current_token = _process_start_time_token(record.pid)
        return current_token is not None and current_token == record.start_time_token

    def status(self, alias: str) -> dict:
        validate_alias(alias)
        with self._alias_lock(alias):
            record = self.load_record(alias)
            if record is None:
                return {"alias": alias, "state": "not_running"}

            process = self._processes.get(alias)
            if process is not None:
                if process.poll() is None:
                    return {"alias": alias, "state": "running", "pid": record.pid, "host": record.host, "port": record.port}
                return {"alias": alias, "state": "stale_record"}

            if self._identity_matches(record):
                return {"alias": alias, "state": "running", "pid": record.pid, "host": record.host, "port": record.port}
            return {"alias": alias, "state": "stale_record"}

    def _terminate_and_reap(self, alias: str, process: subprocess.Popen, pid: int) -> None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        with self._processes_lock:
            self._processes.pop(alias, None)

    def _confirm_terminated(self, pid: int, process: subprocess.Popen | None, timeout_s: float) -> bool:
        if process is not None:
            try:
                process.wait(timeout=timeout_s)
                return True
            except subprocess.TimeoutExpired:
                return False
        # No retained handle: this process did not start the child (a
        # later CLI invocation), so it cannot `waitpid` it. Fall back to
        # polling process identity; the actual parent's reaper collects
        # the zombie once the signal below takes effect.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _process_start_time_token(pid) is None:
                return True
            time.sleep(0.05)
        return False

    def stop(self, alias: str) -> dict:
        validate_alias(alias)
        with self._alias_lock(alias):
            record = self.load_record(alias)
            if record is None:
                return {"alias": alias, "state": "not_running"}

            current_token = _process_start_time_token(record.pid)
            if current_token is None:
                self._clear_record(alias)
                with self._processes_lock:
                    self._processes.pop(alias, None)
                return {"alias": alias, "state": "already_stopped"}

            if current_token != record.start_time_token:
                # The PID was reused by an unrelated process. Clear the stale
                # record but never signal a process PLAG IN did not start.
                self._clear_record(alias)
                raise ProcessIdentityError(
                    "recorded process identity does not match the running process; refusing to signal it",
                    alias=alias,
                    pid=record.pid,
                )

            process = self._processes.get(alias)
            os.kill(record.pid, signal.SIGTERM)
            terminated = self._confirm_terminated(record.pid, process, timeout_s=5.0)
            if not terminated:
                os.kill(record.pid, signal.SIGKILL)
                terminated = self._confirm_terminated(record.pid, process, timeout_s=5.0)
            if not terminated:
                raise ProcessIdentityError(
                    "process did not terminate after SIGKILL; refusing to report it stopped",
                    alias=alias,
                    pid=record.pid,
                )

            self._clear_record(alias)
            with self._processes_lock:
                self._processes.pop(alias, None)
            return {"alias": alias, "state": "stopped"}
