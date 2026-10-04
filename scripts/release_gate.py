#!/usr/bin/env python3
"""Verify version agreement and the declared public source boundary."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    ".gitignore",
    "CLIENT_INTERFACE.md",
    "LICENSE",
    "MANIFEST.in",
    "PRODUCT_SPEC.md",
    "README.md",
    "RELEASE_NOTES.md",
    "SECURITY.md",
    "docs/VERIFICATION.md",
    "pyproject.toml",
    "tools/verify_inference_receipts.py",
}
FORBIDDEN_TOP_LEVEL = {
    "evidence",
    "handoffs",
    "trial",
    "PLAG_IN_" + "GEMINI_BASELINE_STAGING_2026-08-25",
}
FORBIDDEN_NAMES = {
    "DECISIONS.md",
    "ROADMAP.md",
    "source_" + "register.csv",
    "source_" + "manifest.txt",
}
FORBIDDEN_PATTERNS = {
    "private_home_path": re.compile(r"/(?:var/)?home/[^/\s]+(?:/|\b)"),
    "restricted_folder": re.compile("Sensitive " + "data|claude " + "memory " + "folders", re.I),
    "github_token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    "huggingface_token": re.compile("hf" + r"_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "private_key": re.compile("BEGIN " + r"(?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
}
FORBIDDEN_TOP_LEVEL_PREFIXES = ("CLAUDE_", "CODEX_", "LIVE_")
FORBIDDEN_PARTS = {"__pycache__", ".venv", "build", "dist"}
TEXT_SUFFIXES = {".cfg", ".csv", ".ini", ".json", ".md", ".py", ".toml", ".txt"}


def source_version() -> str:
    tree = ast.parse((ROOT / "src/plag_in/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError("src/plag_in/__init__.py has no static __version__ string")


def release_paths() -> list[Path]:
    if (ROOT / ".git").is_dir():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not any(part in {".git", "__pycache__", ".venv", "build", "dist"} for part in path.parts)
        and path.name != "verify_release.py"
    ]


def main() -> int:
    failures: list[str] = []
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package_version = str(project["version"])
    module_version = source_version()
    if package_version != module_version:
        failures.append(
            f"version mismatch: pyproject={package_version!r}, source={module_version!r}"
        )

    paths = release_paths()
    relative_paths = {str(path.relative_to(ROOT)) for path in paths}
    for required in sorted(REQUIRED_FILES - relative_paths):
        failures.append(f"missing required release file: {required}")
    for name in sorted(FORBIDDEN_TOP_LEVEL):
        if any(relative == name or relative.startswith(name + "/") for relative in relative_paths):
            failures.append(f"forbidden top-level release path: {name}")
    for name in sorted(FORBIDDEN_NAMES & relative_paths):
        failures.append(f"forbidden internal release file: {name}")
    for relative in sorted(relative_paths):
        relative_path = Path(relative)
        if relative_path.parts and relative_path.parts[0].startswith(FORBIDDEN_TOP_LEVEL_PREFIXES):
            failures.append(f"forbidden top-level release path: {relative}")
        if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in relative_path.parts):
            failures.append(f"forbidden generated release path: {relative}")
        if relative.endswith(".pyc") or ".bak-" in relative:
            failures.append(f"forbidden generated release path: {relative}")

    scanned = 0
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"Release boundary valid: version {package_version}; "
        f"{len(paths)} files in scope; {scanned} text files scanned; 0 forbidden matches."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
