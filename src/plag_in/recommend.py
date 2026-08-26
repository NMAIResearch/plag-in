"""Deterministic recommendation over a versioned fixture catalogue.

This module has no download function. Recommendation reads a catalogue
and returns at most three ranked, source-recorded options; it never
performs a network action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecommendationQuery:
    task: str
    available_memory_gb: float
    preference: str = "quality"  # "speed" | "quality"
    context_tokens: int = 0
    licence_constraint: str | None = None


def load_catalogue(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["models"]


def recommend(query: RecommendationQuery, catalogue: list[dict]) -> list[dict]:
    if query.preference not in ("speed", "quality"):
        raise ValueError(f"unsupported preference: {query.preference!r}")

    candidates = []
    for entry in catalogue:
        if query.task not in entry.get("task_tags", []):
            continue
        if entry["memory_estimate_gb"] > query.available_memory_gb:
            continue
        if entry["context_tokens"] < query.context_tokens:
            continue
        if query.licence_constraint and entry["licence"] != query.licence_constraint:
            continue
        candidates.append(entry)

    score_field = "speed_score" if query.preference == "speed" else "quality_score"
    candidates.sort(key=lambda entry: (-entry[score_field], entry["id"]))
    return candidates[:3]
