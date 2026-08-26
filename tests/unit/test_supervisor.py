import contextlib
import gc
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import warnings
from pathlib import Path

from plag_in.errors import ExecutableNotAllowedError, PortConflictError, ProcessIdentityError
from plag_in.identity import hash_file
from plag_in.supervisor import SessionRecord, Supervisor, _process_start_time_token

from tests.support import FAKE_ENGINE_PATH, free_port, write_fixture_weight


class SupervisorStartStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.tmp.name) / "sessions"
        self.supervisor = Supervisor(self.sessions_dir)
        self.model_path = write_fixture_weight(Path(self.tmp.name) / "model.gguf")
        self.host = "127.0.0.1"

    def tearDown(self):
        self.tmp.cleanup()

    def _argv(self, port: int, executable: Path | None = None) -> list[str]:
        exe = str(executable) if executable is not None else str(FAKE_ENGINE_PATH)
        return [exe, "--model", str(self.model_path), "--host", self.host, "--port", str(port)]

    def test_start_launches_process_and_waits_for_readiness(self):
        port = free_port()
        record = self.supervisor.start(
            alias="fixture-alias",
            argv=self._argv(port),
            allowlisted_executable=str(FAKE_ENGINE_PATH),
            host=self.host,
            port=port,
            health_path="/health",
            ready_timeout_s=5.0,
        )
        try:
            self.assertTrue(record.pid > 0)
            self.assertEqual(record.executable_digest, hash_file(FAKE_ENGINE_PATH))
            with urllib.request.urlopen(f"http://{self.host}:{port}/health", timeout=2) as resp:  # noqa: S310
                self.assertEqual(resp.status, 200)
            status = self.supervisor.status("fixture-alias")
            self.assertEqual(status["state"], "running")
        finally:
            self.supervisor.stop("fixture-alias")

    def test_non_loopback_engine_host_is_refused(self):
        port = free_port()
        with self.assertRaises(Exception):  # LocalityPolicyError; port bind may also fail first
            self.supervisor.start(
                alias="fixture-alias",
                argv=self._argv(port),
                allowlisted_executable=str(FAKE_ENGINE_PATH),
                host="203.0.113.10",
                port=port,
                health_path="/health",
            )
        self.assertIsNone(self.supervisor.load_record("fixture-alias"))

    def test_executable_not_on_allowlist_is_refused(self):
        port = free_port()
        with self.assertRaises(ExecutableNotAllowedError):
            self.supervisor.start(
                alias="fixture-alias",
                argv=self._argv(port),
                allowlisted_executable="/completely/different/binary",
                host=self.host,
                port=port,
                health_path="/health",
            )
        # Nothing should be listening: the refusal happened before launch.
        with self.assertRaises((ConnectionRefusedError, OSError)):
            with contextlib.closing(socket.create_connection((self.host, port), timeout=1)):
                pass
        self.assertIsNone(self.supervisor.load_record("fixture-alias"))

    def test_port_conflict_detected_before_launch(self):
        port = free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind((self.host, port))
        blocker.listen(1)
        try:
            with self.assertRaises(PortConflictError):
                self.supervisor.start(
                    alias="fixture-alias",
                    argv=self._argv(port),
                    allowlisted_executable=str(FAKE_ENGINE_PATH),
                    host=self.host,
                    port=port,
                    health_path="/health",
                )
        finally:
            blocker.close()

    def test_stop_terminates_only_the_owned_process(self):
        port = free_port()
        record = self.supervisor.start(
            alias="fixture-alias",
            argv=self._argv(port),
            allowlisted_executable=str(FAKE_ENGINE_PATH),
            host=self.host,
            port=port,
            health_path="/health",
        )
        result = self.supervisor.stop("fixture-alias")
        self.assertEqual(result["state"], "stopped")
        self.assertIsNone(_process_start_time_token(record.pid))
        self.assertIsNone(self.supervisor.load_record("fixture-alias"))

    def test_stop_produces_no_resource_warning_under_forced_gc(self):
        port = free_port()
        self.supervisor.start(
            alias="fixture-alias",
            argv=self._argv(port),
            allowlisted_executable=str(FAKE_ENGINE_PATH),
            host=self.host,
            port=port,
            health_path="/health",
        )
        self.supervisor.stop("fixture-alias")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            gc.collect()
            gc.collect()
        resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(resource_warnings, [])

    def test_stale_record_does_not_terminate_unrelated_process(self):
        # Fabricate a session record pointing at this test process's own
        # PID (very much alive and unrelated to anything PLAG IN started)
        # but with a start-time token that cannot match.
        fake_record = SessionRecord(
            alias="fixture-alias",
            pid=os.getpid(),
            executable=str(FAKE_ENGINE_PATH),
            executable_digest="f" * 64,
            argv_digest="irrelevant",
            start_time_token="not-a-real-token",
            host=self.host,
            port=free_port(),
            started_at=str(time.time()),
        )
        self.supervisor._write_record(fake_record)  # noqa: SLF001 - fabricating stale state deliberately

        with self.assertRaises(ProcessIdentityError):
            self.supervisor.stop("fixture-alias")

        # The unrelated process (this very test) must still be alive.
        os.kill(os.getpid(), 0)  # raises if the process no longer exists
        # The stale record is cleared so it cannot be reused.
        self.assertIsNone(self.supervisor.load_record("fixture-alias"))

    def test_status_not_running_when_no_record(self):
        status = self.supervisor.status("never-started")
        self.assertEqual(status["state"], "not_running")

    def test_executable_mutation_changes_the_session_digest(self):
        variant_a = Path(self.tmp.name) / "engine_a.py"
        variant_b = Path(self.tmp.name) / "engine_b.py"
        shutil.copy(FAKE_ENGINE_PATH, variant_a)
        shutil.copy(FAKE_ENGINE_PATH, variant_b)
        variant_b.write_bytes(variant_b.read_bytes() + b"\n# mutated variant\n")
        variant_a.chmod(0o755)
        variant_b.chmod(0o755)

        port_a, port_b = free_port(), free_port()
        record_a = self.supervisor.start(
            alias="engine-a",
            argv=self._argv(port_a, variant_a),
            allowlisted_executable=str(variant_a),
            host=self.host,
            port=port_a,
            health_path="/health",
        )
        record_b = self.supervisor.start(
            alias="engine-b",
            argv=self._argv(port_b, variant_b),
            allowlisted_executable=str(variant_b),
            host=self.host,
            port=port_b,
            health_path="/health",
        )
        try:
            self.assertNotEqual(record_a.executable_digest, record_b.executable_digest)
        finally:
            self.supervisor.stop("engine-a")
            self.supervisor.stop("engine-b")


class ArgvDigestStableAcrossInterpretersTest(unittest.TestCase):
    def test_same_argv_same_digest_in_two_fresh_interpreters(self):
        script = (
            "import sys; sys.path.insert(0, %r); "
            "from plag_in.identity import canonical_digest; "
            "print(canonical_digest({'argv': ['a', 'b; rm -rf /', 'c']}))"
        ) % str(Path(__file__).parent.parent.parent / "src")
        outputs = set()
        for _ in range(2):
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=10
            )
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1, f"digest was not stable across interpreters: {outputs}")


if __name__ == "__main__":
    unittest.main()
