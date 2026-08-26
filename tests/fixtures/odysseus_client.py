"""Odysseus-shaped OpenAI-compatible client fixture.

Mimics the minimal connection contract an Odysseus-style harness uses to
reach a local OpenAI-compatible endpoint: a base URL, an API key and a
model alias, resolving model IDs from `/v1/models` (CLIENT_INTERFACE.md
section 9). Contains no import from, and no code copied from, any
Odysseus source tree.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class OdysseusShapedClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        req.add_header("Authorization", f"Bearer {self.api_key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def list_models(self) -> tuple[int, dict]:
        # base_url already includes the /v1 prefix (CLIENT_INTERFACE.md
        # section 2 connection contract), so paths here are relative to it.
        return self._request("GET", "/models")

    def chat(self, messages: list[dict]) -> tuple[int, dict]:
        return self._request("POST", "/chat/completions", {"model": self.model, "messages": messages})
