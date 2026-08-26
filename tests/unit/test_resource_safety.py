import subprocess
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ConfigurationError
from plag_in.resource_safety import GIB, resource_preflight, verify_chat_cgroup


class ResourceSafetyTests(unittest.TestCase):
    def _host_files(self, root: Path, available_gib: int, full_avg10: float):
        meminfo = root / "meminfo"
        pressure = root / "pressure"
        meminfo.write_text(
            f"MemTotal:       {64 * 1024 * 1024} kB\n"
            f"MemAvailable:   {available_gib * 1024 * 1024} kB\n",
            encoding="utf-8",
        )
        pressure.write_text(
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
            f"full avg10={full_avg10:.2f} avg60=0.00 avg300=0.00 total=0\n",
            encoding="utf-8",
        )
        return meminfo, pressure

    @staticmethod
    def _gpu(free_mib=12000):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"Fixture GPU, 16376, {free_mib}\n",
                stderr="",
            )

        return run

    def test_preflight_passes_with_measured_ram_pressure_and_vram_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, pressure = self._host_files(Path(tmp), 32, 0.0)
            report = resource_preflight(
                2 * GIB,
                "all",
                meminfo_path=meminfo,
                pressure_path=pressure,
                command_runner=self._gpu(),
            )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["policy_validation_status"], "provisional_unvalidated")
        self.assertEqual(
            report["enforcement_boundary"], "cgroup_memory_and_swap_limits"
        )
        self.assertGreaterEqual(report["required_available_ram_bytes"], 20 * GIB)

    def test_preflight_refuses_low_system_ram(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, pressure = self._host_files(Path(tmp), 4, 0.0)
            report = resource_preflight(
                2 * GIB,
                "all",
                meminfo_path=meminfo,
                pressure_path=pressure,
                command_runner=self._gpu(),
            )
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("system RAM" in item for item in report["failures"]))

    def test_preflight_refuses_memory_pressure_and_low_vram(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, pressure = self._host_files(Path(tmp), 32, 1.0)
            report = resource_preflight(
                2 * GIB,
                "all",
                meminfo_path=meminfo,
                pressure_path=pressure,
                command_runner=self._gpu(free_mib=2048),
            )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["failures"]), 2)

    def test_preflight_refuses_unmeasured_gpu_headroom(self):
        def failed(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            meminfo, pressure = self._host_files(Path(tmp), 32, 0.0)
            with self.assertRaises(ConfigurationError):
                resource_preflight(
                    2 * GIB,
                    "all",
                    meminfo_path=meminfo,
                    pressure_path=pressure,
                    command_runner=failed,
                )

    def test_cgroup_verification_requires_hard_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "self.cgroup"
            proc.write_text("0::/user.slice/plag.scope\n", encoding="utf-8")
            scope = root / "cgroup" / "user.slice" / "plag.scope"
            scope.mkdir(parents=True)
            (scope / "memory.high").write_text(str(10 * GIB), encoding="utf-8")
            (scope / "memory.max").write_text(str(12 * GIB), encoding="utf-8")
            (scope / "memory.swap.max").write_text(str(2 * GIB), encoding="utf-8")
            observed = verify_chat_cgroup(proc_cgroup_path=proc, cgroup_root=root / "cgroup")
        self.assertEqual(observed["memory.max"], 12 * GIB)

    def test_cgroup_verification_refuses_unbounded_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "self.cgroup"
            proc.write_text("0::/user.slice/plag.scope\n", encoding="utf-8")
            scope = root / "cgroup" / "user.slice" / "plag.scope"
            scope.mkdir(parents=True)
            (scope / "memory.high").write_text(str(10 * GIB), encoding="utf-8")
            (scope / "memory.max").write_text("max", encoding="utf-8")
            (scope / "memory.swap.max").write_text(str(2 * GIB), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                verify_chat_cgroup(proc_cgroup_path=proc, cgroup_root=root / "cgroup")


if __name__ == "__main__":
    unittest.main()
