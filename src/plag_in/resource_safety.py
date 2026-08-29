"""Fail-closed host resource checks for interactive model loading."""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from plag_in.errors import ConfigurationError


MIB = 1024 * 1024
GIB = 1024 * MIB
# Provisional private-alpha policy. The cgroup maximum is the enforcement
# boundary. The reserve and working-set formula remain unvalidated until a
# current digest-bound live run records system RAM and VRAM measurements.
CHAT_MEMORY_HIGH_BYTES = 10 * GIB
CHAT_MEMORY_MAX_BYTES = 12 * GIB
CHAT_SWAP_MAX_BYTES = 2 * GIB
DESKTOP_RAM_RESERVE_BYTES = 8 * GIB
MIN_AVAILABLE_RAM_BYTES = CHAT_MEMORY_MAX_BYTES + DESKTOP_RAM_RESERVE_BYTES
MIN_FREE_VRAM_MIB = 4096
MAX_MEMORY_FULL_AVG10 = 0.10


def _key_values(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        number = value.strip().split()[0]
        values[key] = int(number) * 1024
    return values


def _memory_full_avg10(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("full "):
            match = re.search(r"\bavg10=([0-9.]+)", line)
            if match:
                return float(match.group(1))
    raise ConfigurationError("memory pressure full avg10 is unavailable")


def _nvidia_memory(command_runner=subprocess.run) -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command_runner(command, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError("NVIDIA memory headroom could not be measured") from exc
    if result.returncode != 0:
        raise ConfigurationError("NVIDIA memory headroom could not be measured")
    first = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        raise ConfigurationError("NVIDIA memory headroom returned an invalid record")
    try:
        total_mib = int(parts[1])
        free_mib = int(parts[2])
    except ValueError as exc:
        raise ConfigurationError("NVIDIA memory headroom returned non-integer values") from exc
    return {"name": parts[0], "total_mib": total_mib, "free_mib": free_mib}


def resource_preflight(
    model_size_bytes: int,
    gpu_layers: str,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    pressure_path: Path = Path("/proc/pressure/memory"),
    command_runner=subprocess.run,
) -> dict:
    """Measure immediate host headroom without loading native code or model bytes."""
    try:
        memory = _key_values(meminfo_path)
        full_avg10 = _memory_full_avg10(pressure_path)
    except (OSError, ValueError) as exc:
        raise ConfigurationError("host memory headroom could not be measured") from exc
    available = memory.get("MemAvailable")
    if available is None:
        raise ConfigurationError("MemAvailable is unavailable")
    estimated_working_set = model_size_bytes * 2 + 2 * GIB
    required_ram = max(MIN_AVAILABLE_RAM_BYTES, estimated_working_set + DESKTOP_RAM_RESERVE_BYTES)
    failures = []
    if available < required_ram:
        failures.append("available system RAM is below the model-load safety floor")
    if full_avg10 > MAX_MEMORY_FULL_AVG10:
        failures.append("recent full-memory pressure exceeds the safety threshold")
    if estimated_working_set > CHAT_MEMORY_MAX_BYTES:
        failures.append("estimated model working set exceeds the chat cgroup memory maximum")

    gpu = None
    required_free_vram_mib = None
    if str(gpu_layers).strip().lower() != "0":
        gpu = _nvidia_memory(command_runner)
        required_free_vram_mib = max(
            MIN_FREE_VRAM_MIB,
            math.ceil(model_size_bytes / MIB * 1.5) + 2048,
        )
        if gpu["free_mib"] < required_free_vram_mib:
            failures.append("free GPU memory is below the model-load safety floor")

    return {
        "status": "pass" if not failures else "fail",
        "policy_validation_status": "provisional_unvalidated",
        "enforcement_boundary": "cgroup_memory_and_swap_limits",
        "model_size_bytes": model_size_bytes,
        "available_ram_bytes": available,
        "required_available_ram_bytes": required_ram,
        "estimated_working_set_bytes": estimated_working_set,
        "memory_full_avg10": full_avg10,
        "maximum_memory_full_avg10": MAX_MEMORY_FULL_AVG10,
        "gpu": gpu,
        "required_free_vram_mib": required_free_vram_mib,
        "failures": failures,
    }


def evaluate_model_admission(
    model_size_bytes: int,
    gpu_layers: str,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    pressure_path: Path = Path("/proc/pressure/memory"),
    command_runner=subprocess.run,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict:
    """Report the preflight decision for one model without raising.

    `resource_preflight` and `verify_chat_cgroup` are the enforcement path
    and they raise. A compatibility matrix needs the same measurements for
    an entry it will not start, so this reports the decision instead. The
    formula, reserve and thresholds are the same values and remain
    provisional; nothing here weakens a check.

    `outcome` is `admitted`, `resource_refused` or `measurement_unavailable`.
    A measurement that could not be taken is never read as headroom.
    """
    record: dict = {
        "outcome": "admitted",
        "model_size_bytes": model_size_bytes,
        "requested_gpu_layers": gpu_layers,
        "policy_validation_status": "provisional_unvalidated",
        "enforcement_boundary": "cgroup_memory_and_swap_limits",
        "cgroup": None,
        "preflight": None,
        "failures": [],
        "unassessed": [],
    }
    try:
        record["cgroup"] = verify_chat_cgroup(
            proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root
        )
    except ConfigurationError as exc:
        record["outcome"] = "measurement_unavailable"
        record["unassessed"].append("chat_cgroup")
        record["failures"].append(exc.message)
        return record

    try:
        preflight = resource_preflight(
            model_size_bytes,
            gpu_layers,
            meminfo_path=meminfo_path,
            pressure_path=pressure_path,
            command_runner=command_runner,
        )
    except ConfigurationError as exc:
        record["outcome"] = "measurement_unavailable"
        record["unassessed"].append("host_headroom")
        record["failures"].append(exc.message)
        return record

    record["preflight"] = preflight
    if preflight["status"] != "pass":
        record["outcome"] = "resource_refused"
        record["failures"] = list(preflight["failures"])
    return record


def _cgroup_file(name: str, *, proc_cgroup_path: Path, cgroup_root: Path) -> Path:
    unified = None
    for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            unified = fields[2].lstrip("/")
            break
    if unified is None:
        raise ConfigurationError("the unified cgroup path is unavailable")
    return cgroup_root / unified / name


def verify_chat_cgroup(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict:
    """Verify that systemd applied the declared hard memory and swap ceilings."""
    values = {}
    for name in ("memory.high", "memory.max", "memory.swap.max"):
        path = _cgroup_file(name, proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root)
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigurationError(f"chat cgroup control is unreadable: {name}") from exc
        if raw == "max":
            raise ConfigurationError(f"chat cgroup control is unbounded: {name}")
        try:
            values[name] = int(raw)
        except ValueError as exc:
            raise ConfigurationError(f"chat cgroup control is invalid: {name}") from exc
    if values["memory.high"] > CHAT_MEMORY_HIGH_BYTES:
        raise ConfigurationError("chat cgroup memory.high exceeds the declared safety ceiling")
    if values["memory.max"] > CHAT_MEMORY_MAX_BYTES:
        raise ConfigurationError("chat cgroup memory.max exceeds the declared safety ceiling")
    if values["memory.swap.max"] > CHAT_SWAP_MAX_BYTES:
        raise ConfigurationError("chat cgroup memory.swap.max exceeds the declared safety ceiling")
    return values
