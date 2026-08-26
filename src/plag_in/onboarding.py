"""Confirmed first-run inspection, configuration and local test chat.

Inspection is read-only. Configuration writes and model loading are separate
choices with explicit confirmation. Locality controls remain separate from
runtime settings so a performance choice is never presented as privacy
evidence.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Callable, TextIO

from plag_in.config import is_loopback, load_config_file


def _platform_support(system: str) -> str:
    if system == "Linux":
        return "linux_mvp"
    if system in {"Darwin", "Windows"}:
        return "unsupported_untested"
    return "unsupported_unknown"


def doctor_report(config_path: Path | None = None) -> dict:
    """Inspect platform and declared configuration without changing the host."""
    system = platform.system()
    report = {
        "inspection": {
            "network_action": "none",
            "filesystem_changes": "none",
            "model_started": False,
        },
        "platform": {
            "system": system,
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "support_status": _platform_support(system),
        },
        "configuration": {"status": "not_supplied"},
        "locality_controls": {
            "status": "unconfigured",
            "active_evidence_level": "none",
            "external_inference_provider": "absent_from_mvp",
            "operating_system_egress_policy": "unassessed",
            "continuous_traffic_observation": "unassessed",
        },
        "runtime_settings": {
            "status": "unconfigured",
            "note": "context, sampling, GPU, thread and batch settings are not locality controls",
        },
        "discovery": {
            "default_ollama_store_present": (Path.home() / ".ollama" / "models").is_dir(),
            "configured_model_roots": 0,
            "present_model_roots": 0,
            "configured_engines": 0,
            "configured_profiles": 0,
            "tested_profiles": [],
        },
        "next_steps": [
            "Review or create a PLAG IN configuration without starting a model.",
            "Run plag-in doctor --config <config.json>.",
            "Run plag-in inspect --config <config.json> after reviewing the configuration.",
        ],
    }

    if config_path is None:
        return report

    config = load_config_file(config_path)
    loopback = is_loopback(config.bind.host)
    report["configuration"] = {"status": "loaded_and_validated"}
    report["locality_controls"] = {
        "status": "configuration_only",
        "active_evidence_level": "L0",
        "bind_scope": "loopback" if loopback else "private_network_requested",
        "authentication": config.security.auth_mode,
        "content_logging": config.security.content_logging,
        "external_inference_provider": "absent_from_mvp",
        "remote_fallback": "absent_from_mvp",
        "operating_system_egress_policy": "unassessed",
        "continuous_traffic_observation": "unassessed",
    }
    report["runtime_settings"] = {
        "status": "requested_configuration",
        "values": config.runtime.as_dict(),
        "note": "effective values are unavailable until a model is loaded and queried",
    }
    roots = tuple(config.model_roots)
    report["discovery"] = {
        "default_ollama_store_present": (Path.home() / ".ollama" / "models").is_dir(),
        "configured_ollama_store_present": (
            config.ollama_manifest_root.is_dir()
            if config.ollama_manifest_root is not None
            else False
        ),
        "configured_model_roots": len(roots),
        "present_model_roots": sum(root.is_dir() for root in roots),
        "configured_engines": len(config.engines),
        "declared_engine_kinds": sorted(engine.kind for engine in config.engines.values()),
        "configured_profiles": len(config.profiles),
        "tested_profiles": sorted(
            alias for alias, profile in config.profiles.items()
            if profile.compatibility_status == "tested"
        ),
    }
    report["next_steps"] = [
        "Review the locality controls and requested runtime settings as separate sections.",
        "Run plag-in inspect --config <same-config> to enumerate configured local models.",
        "Start a selected model only after reviewing its exact identity and runtime profile.",
    ]
    return report


def render_doctor(report: dict) -> str:
    """Render a stable human-readable doctor report."""
    platform_record = report["platform"]
    locality = report["locality_controls"]
    runtime = report["runtime_settings"]
    discovery = report["discovery"]

    lines = [
        "PLAG IN doctor",
        "",
        "Inspection effect",
        "  Network action: none",
        "  Filesystem changes: none",
        "  Model started: no",
        "",
        "Platform",
        f"  System: {platform_record['system']}",
        f"  Machine: {platform_record['machine']}",
        f"  Python: {platform_record['python_version']}",
        f"  Support: {platform_record['support_status']}",
        "",
        "Locality controls",
        f"  Status: {locality['status']}",
        f"  Evidence level: {locality['active_evidence_level']}",
        f"  External inference provider: {locality['external_inference_provider']}",
        f"  Operating-system egress policy: {locality['operating_system_egress_policy']}",
        f"  Continuous traffic observation: {locality['continuous_traffic_observation']}",
    ]
    if "bind_scope" in locality:
        lines.extend(
            [
                f"  Bind scope: {locality['bind_scope']}",
                f"  Authentication: {locality['authentication']}",
                f"  Content logging: {str(locality['content_logging']).lower()}",
                f"  Remote fallback: {locality['remote_fallback']}",
            ]
        )

    lines.extend(["", "Runtime settings", f"  Status: {runtime['status']}"])
    for key, value in runtime.get("values", {}).items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  Note: {runtime['note']}")

    lines.extend(
        [
            "",
            "Local discovery",
            f"  Default Ollama store present: {str(discovery['default_ollama_store_present']).lower()}",
            f"  Configured model roots: {discovery['configured_model_roots']}",
            f"  Present model roots: {discovery['present_model_roots']}",
            f"  Configured engines: {discovery['configured_engines']}",
            f"  Configured profiles: {discovery['configured_profiles']}",
            "  Tested profiles: " + (", ".join(discovery["tested_profiles"]) or "none"),
            "",
            "Next steps",
        ]
    )
    lines.extend(f"  {index}. {step}" for index, step in enumerate(report["next_steps"], 1))
    return "\n".join(lines)


_SETUP_MENU = """PLAG IN setup

Inspection is read-only. Configuration and model loading each require confirmation.
Inference uses an authenticated loopback connection with no remote provider.

1. Inspect this computer
2. Configure a tested existing local model
3. Start a private test chat
4. Connect an existing harness
5. Show the missing-model recommendation workflow
6. Explain privacy and locality controls
7. Show advanced commands
8. Exit
"""


def run_setup_assistant(
    config_path: Path | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    output: TextIO | None = None,
    configure_fn: Callable[[Path | None, Callable[[str], str], TextIO], Path | None] | None = None,
    chat_fn: Callable[[Path, Callable[[str], str], TextIO], int] | None = None,
    connect_fn: Callable[[Path, Callable[[str], str], TextIO], int] | None = None,
    max_cycles: int | None = None,
) -> int:
    """Keep the terminal menu active until the operator exits."""
    injected_io = input_fn is not None or output is not None
    if input_fn is None:
        input_fn = input
    if output is None:
        output = sys.stdout
    if max_cycles is None and injected_io:
        max_cycles = 32
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be positive")
    active_config = config_path
    cycles = 0
    while True:
        cycles += 1
        if max_cycles is not None and cycles > max_cycles:
            print("Setup stopped at the buffered-interface safety limit.", file=output)
            return 2
        print(_SETUP_MENU, file=output)
        try:
            choice = input_fn("Select 1-8: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup closed.", file=output)
            return 0

        if choice == "1":
            inspected_config = active_config if active_config and active_config.is_file() else None
            print("", file=output)
            print(render_doctor(doctor_report(inspected_config)), file=output)
            print("", file=output)
            continue
        if choice == "2":
            if configure_fn is None:
                print("Configuration builder is unavailable. No changes made.\n", file=output)
                continue
            try:
                configured = configure_fn(active_config, input_fn, output)
            except (EOFError, KeyboardInterrupt):
                print("\nAction cancelled. No change was made and no model was started.\n", file=output)
                continue
            if configured is not None:
                active_config = configured
            print("", file=output)
            continue
        if choice == "3":
            effective = active_config if active_config and active_config.is_file() else None
            if effective is None or chat_fn is None:
                print("Configure a tested local model first. No model was started.\n", file=output)
                continue
            try:
                chat_fn(effective, input_fn, output)
            except (EOFError, KeyboardInterrupt):
                print("\nAction cancelled. No model was started.\n", file=output)
            print("", file=output)
            continue
        if choice == "4":
            effective = active_config if active_config and active_config.is_file() else None
            if effective is None or connect_fn is None:
                print("Configure a tested local model first. No model was started.\n", file=output)
                continue
            try:
                connect_fn(effective, input_fn, output)
            except (EOFError, KeyboardInterrupt):
                print("\nAction cancelled. No model was started.\n", file=output)
            print("", file=output)
            continue
        if choice == "5":
            print(
                "\nMissing local model\n"
                "  Run plag-in recommend with a reviewed local catalogue.\n"
                "  Recommendation returns at most three options and never downloads a model.\n"
                "  Download remains a separate approval-gated action.\n",
                file=output,
            )
            continue
        if choice == "6":
            print(
                "\nPrivacy and locality controls\n"
                "  Locality: binding, authentication, remote fallback, telemetry, network policy, "
                "traffic observation and content logging.\n"
                "  Runtime: context, temperature, top-p, GPU layers, threads and batching.\n"
                "  Runtime settings are recorded for reproducibility but do not prove locality.\n",
                file=output,
            )
            continue
        if choice == "7":
            print(
                "\nAdvanced commands\n"
                "  plag-in --help\n"
                "  plag-in doctor --help\n"
                "  plag-in inspect --help\n"
                "  plag-in serve --help\n",
                file=output,
            )
            continue
        if choice == "8":
            print("Setup closed.", file=output)
            return 0

        print("Invalid selection. Choose a number from 1 to 8.\n", file=output)
