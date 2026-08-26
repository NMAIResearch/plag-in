#!/usr/bin/env python3
"""Fixture fake inference engine used by the default test suite.

This is not llama-server. It is a minimal stdlib-only HTTP server that
speaks enough of the OpenAI-compatible surface for gateway and supervisor
tests to run without a real engine binary, a real model store or any host
service. It is invoked directly as an executable (shebang above) so it
matches the `LlamaServerAdapter` argv contract: [executable, "--model",
path, "--host", host, "--port", port].
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
from pathlib import Path


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": self.server.model_id}]})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not_found"})
            return
        if self.server.api_key is not None:
            expected = f"Bearer {self.server.api_key}"
            if self.headers.get("Authorization") != expected:
                self._json(401, {"error": "unauthorised_backend_fixture"})
                return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        request = json.loads(raw) if raw else {}
        messages = request.get("messages", [])
        echoed = messages[-1]["content"] if messages else ""
        message = {"role": "assistant", "content": f"fake response ({len(echoed)} chars received)"}
        if self.server.tool_calls is not None:
            message["tool_calls"] = self.server.tool_calls
        response = {
            "id": "fake-completion-1",
            "object": "chat.completion",
            "model": self.server.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        }
        self._json(200, response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--argv-dump")
    parser.add_argument("--with-tool-call", action="store_true")
    parser.add_argument("--api-key-file")
    args, _unknown = parser.parse_known_args()

    if args.argv_dump:
        Path(args.argv_dump).write_text(json.dumps(sys.argv))

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    server.model_id = args.model or "fake-model"
    server.api_key = (
        Path(args.api_key_file).read_text(encoding="utf-8").strip()
        if args.api_key_file
        else None
    )
    server.tool_calls = (
        [{"id": "call_1", "type": "function", "function": {"name": "not_executed", "arguments": "{}"}}]
        if args.with_tool_call
        else None
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
