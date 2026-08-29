"""Bounded terminal input helpers for confirmations and menu choices."""
from __future__ import annotations

import os
import select
import sys
import termios
import tty
from typing import Callable, TextIO


DEFAULT_ATTEMPTS = 3


def terminal_menu_available(
    input_fn: Callable[[str], str],
    output: TextIO,
) -> bool:
    """Return whether direct key navigation is available on this terminal."""
    return (
        input_fn is input
        and sys.stdin.isatty()
        and bool(getattr(output, "isatty", lambda: False)())
    )


def _read_terminal_key() -> str:
    """Read one key or one three-byte terminal escape sequence."""
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = os.read(fd, 1).decode("utf-8", errors="ignore")
        if first != "\x1b":
            return first
        suffix = b""
        for _ in range(2):
            ready, _, _ = select.select([fd], [], [], 0.04)
            if not ready:
                break
            suffix += os.read(fd, 1)
        return first + suffix.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def select_menu(
    options: list[str] | tuple[str, ...],
    output: TextIO,
    prompt: str,
    *,
    initial: int = 1,
    read_key: Callable[[], str] | None = None,
) -> int | None:
    """Select a one-based menu item with arrow keys and Enter."""
    if not options:
        raise ValueError("options must not be empty")
    if initial < 1 or initial > len(options):
        raise ValueError("initial selection is outside the menu")
    key_reader = read_key or _read_terminal_key
    selected = initial - 1
    rendered = False
    print(prompt, file=output)

    while True:
        if rendered:
            output.write(f"\x1b[{len(options)}A")
        for index, label in enumerate(options):
            marker = ">" if index == selected else " "
            if index == selected:
                output.write(f"\r\x1b[2K\x1b[7m{marker} {label}\x1b[0m\n")
            else:
                output.write(f"\r\x1b[2K{marker} {label}\n")
        output.flush()
        rendered = True

        try:
            key = key_reader()
        except (EOFError, KeyboardInterrupt):
            output.write("\nSelection cancelled. No model was started.\n")
            output.flush()
            return None
        if key in {"\r", "\n"}:
            output.write(f"Selected: {options[selected]}\n")
            output.flush()
            return selected + 1
        if key in {"\x1b[A", "k", "K"}:
            selected = (selected - 1) % len(options)
            continue
        if key in {"\x1b[B", "j", "J"}:
            selected = (selected + 1) % len(options)
            continue
        if key == "\x1b[H":
            selected = 0
            continue
        if key == "\x1b[F":
            selected = len(options) - 1
            continue
        if key in {"q", "Q", "\x1b"}:
            output.write("\nSelection cancelled. No model was started.\n")
            output.flush()
            return None
        if key.isdigit() and 1 <= int(key) <= len(options):
            selected = int(key) - 1


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
