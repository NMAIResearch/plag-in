"""Configuration-export commands: env, JSON and curl forms.

Secrets are omitted by default. A caller must pass `reveal_secret=True`
explicitly to include the real API key in an export.
"""
from __future__ import annotations

import json
import shlex

_REDACTED = "<redacted, pass --reveal-secret to include>"


def _key_value(api_key: str, reveal_secret: bool) -> str:
    return api_key if reveal_secret else _REDACTED


def emit_env(base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    key = _key_value(api_key, reveal_secret)
    return (
        f"PLAG_IN_BASE_URL={base_url}\n"
        f"PLAG_IN_API_KEY={key}\n"
        f"PLAG_IN_MODEL={model}\n"
    )


def emit_openai_env(base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    key = _key_value(api_key, reveal_secret)
    return (
        "# Local loopback gateway only. Remote provider: none.\n"
        f"OPENAI_BASE_URL={base_url}\n"
        f"OPENAI_API_KEY={key}\n"
        f"OPENAI_MODEL={model}\n"
    )


def emit_manual(base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    key = _key_value(api_key, reveal_secret)
    return (
        "Transport: local HTTP\n"
        "Protocol: Chat Completions v1 subset\n"
        f"Base URL: {base_url}\n"
        f"Authentication header: Authorization: Bearer {key}\n"
        f"Model alias: {model}\n"
        "Remote provider: none\n"
    )


def emit_json(base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    payload = {
        "base_url": base_url,
        "api_key": _key_value(api_key, reveal_secret),
        "model": model,
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def emit_curl(base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    key = _key_value(api_key, reveal_secret)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hello"}]})
    return (
        f"curl {shlex.quote(base_url.rstrip('/') + '/chat/completions')} "
        f"-H {shlex.quote('Authorization: Bearer ' + key)} "
        f"-H {shlex.quote('Content-Type: application/json')} "
        f"-d {shlex.quote(body)}"
    )


_EMITTERS = {
    "env": emit_env,
    "manual": emit_manual,
    "openai-env": emit_openai_env,
    "json": emit_json,
    "curl": emit_curl,
}


def emit(fmt: str, base_url: str, api_key: str, model: str, reveal_secret: bool = False) -> str:
    try:
        emitter = _EMITTERS[fmt]
    except KeyError as exc:
        raise ValueError(f"unsupported connect format: {fmt!r}") from exc
    return emitter(base_url, api_key, model, reveal_secret)
