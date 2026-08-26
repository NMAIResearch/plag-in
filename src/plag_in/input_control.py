"""Bounded terminal input helpers for confirmation and numbered choices."""
from __future__ import annotations

from typing import Callable, TextIO


DEFAULT_ATTEMPTS = 3


def _safe_token(value: str, limit: int = 40) -> str:
    clipped = value[:limit]
    rendered = ascii(clipped)
    if len(value) > limit:
        rendered += "..."
    return rendered


def confirm_action(
    input_fn: Callable[[str], str],
    output: TextIO,
    prompt: str,
    *,
    max_attempts: int = DEFAULT_ATTEMPTS,
) -> bool:
    """Accept explicit consent only and reprompt on an unrecognised token."""
    for _ in range(max_attempts):
        try:
            token = input_fn(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAction cancelled. No change was made and no model was started.", file=output)
            return False
        if token in {"y", "yes"}:
            return True
        if token in {"", "n", "no"}:
            return False
        print(
            f"Input not recognised: {_safe_token(token)}. Type y or yes to confirm, "
            "or n or no to decline.",
            file=output,
        )
    print("Confirmation attempt limit reached. The action was cancelled.", file=output)
    return False


def select_number(
    input_fn: Callable[[str], str],
    output: TextIO,
    prompt: str,
    *,
    minimum: int,
    maximum: int,
    max_attempts: int = DEFAULT_ATTEMPTS,
) -> int | None:
    """Return one bounded numbered selection or a clean cancellation."""
    for _ in range(max_attempts):
        try:
            token = input_fn(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled. No model was started.", file=output)
            return None
        if token.isdigit() and minimum <= int(token) <= maximum:
            return int(token)
        if token == "":
            print("Selection cancelled. No model was started.", file=output)
            return None
        print(
            f"Input not recognised: {_safe_token(token)}. Choose a number from "
            f"{minimum} to {maximum}.",
            file=output,
        )
    print("Selection attempt limit reached. No model was started.", file=output)
    return None
