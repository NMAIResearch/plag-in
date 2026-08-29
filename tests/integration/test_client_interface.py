"""CLIENT_INTERFACE.md acceptance over both implemented request protocols.

A generic client reaches a fake model using only base URL, key and model
alias. Every protocol is driven by a neutral raw client with an arbitrary
user-agent before any named harness, and no production module carries a
harness-name branch.
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import plag_in
from plag_in.config import ApiKey
from plag_in.gateway import MAX_REQUEST_BODY_BYTES

from tests.fixtures.odysseus_client import OdysseusShapedClient
from tests.support import build_gateway_stack

# Deliberately not the name of any real client. A conforming gateway must
# behave identically whatever a request calls itself.
NEUTRAL_USER_AGENT = "generic-conformance-probe/0"


def _request(base_url: str, path: str, payload, *, api_key=None, user_agent=NEUTRAL_USER_AGENT,
             headers=None):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", user_agent)
    if api_key is not None:
        request.add_header("Authorization", f"Bearer {api_key}")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


class OdysseusFirstClientRouteTest(unittest.TestCase):
    def test_odysseus_shaped_client_reaches_the_fake_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            api_key = ApiKey(id="odysseus", secret="odysseus-fixture-key", aliases=("fixture-alias",))
            stack = build_gateway_stack(
                Path(tmp), alias="fixture-alias", auth_mode="api_key", api_keys=(api_key,)
            )
            try:
                client = OdysseusShapedClient(
                    base_url=stack.server.base_url + "/v1",
                    api_key="odysseus-fixture-key",
                    model="fixture-alias",
                )
                status, models = client.list_models()
                self.assertEqual(status, 200)
                self.assertEqual([m["id"] for m in models["data"]], ["fixture-alias"])

                status, completion = client.chat([{"role": "user", "content": "hello from odysseus fixture"}])
                self.assertEqual(status, 200)
                self.assertIn("choices", completion)
            finally:
                stack.stop()

    def test_no_gateway_module_imports_odysseus(self):
        import plag_in.gateway as gateway_module

        source = Path(gateway_module.__file__).read_text()
        self.assertNotIn("odysseus", source.lower())


class ProductionSourceNeutralityTests(unittest.TestCase):
    """Probe 14 and handoff 43: no harness or vendor name is executable code.

    Comments and docstrings are excluded, because an audit reference to a
    review document is a record, not behaviour. Everything the interpreter
    acts on is scanned: identifiers, attributes, keywords and string
    literals alike.
    """

    # Harness and vendor names that must never reach executable code.
    # `openai` is deliberately absent: it names the wire protocol and the
    # environment-variable convention a connector emits, which handoff 43
    # permits. That permission is checked separately below rather than
    # assumed, so a real product preference could not hide behind it.
    FORBIDDEN = (
        "odysseus",
        "anthropic",
        "claude",
        "gemini",
        "google",
        "codex",
    )
    PERMITTED_OPENAI_CONTEXTS = (
        "openai-env",
        "emit_openai_env",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OpenAI-style environment values",
    )

    def _executable_text(self, path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in docstrings:
                    found.append(node.value)
            elif isinstance(node, ast.Name):
                found.append(node.id)
            elif isinstance(node, ast.Attribute):
                found.append(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                found.append(node.arg)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append(node.name)
            elif isinstance(node, ast.arg):
                found.append(node.arg)
            elif isinstance(node, ast.alias):
                found.append(node.name)
        return found

    def test_no_production_module_acts_on_a_harness_or_vendor_name(self):
        root = Path(plag_in.__file__).parent
        offenders = []
        for module_path in sorted(root.rglob("*.py")):
            for text in self._executable_text(module_path):
                lowered = text.lower()
                for name in self.FORBIDDEN:
                    if name in lowered:
                        offenders.append(f"{module_path.name}: {name} in {text!r}")
        self.assertEqual(offenders, [])

    def test_every_openai_reference_is_a_protocol_or_variable_name(self):
        root = Path(plag_in.__file__).parent
        offenders = []
        for module_path in sorted(root.rglob("*.py")):
            for text in self._executable_text(module_path):
                if "openai" not in text.lower():
                    continue
                if any(context in text for context in self.PERMITTED_OPENAI_CONTEXTS):
                    continue
                offenders.append(f"{module_path.name}: {text!r}")
        self.assertEqual(offenders, [])

    def test_the_scan_would_catch_a_planted_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            planted = Path(tmp) / "planted.py"
            planted.write_text(
                '"""A docstring naming Odysseus must not trip the scan."""\n'
                "def route(agent):\n"
                "    # A comment naming Odysseus must not trip it either.\n"
                "    if agent == 'odysseus':\n"
                "        return 'special'\n"
                "    return 'generic'\n",
                encoding="utf-8",
            )
            found = [
                text
                for text in self._executable_text(planted)
                if "odysseus" in text.lower()
            ]
        self.assertEqual(found, ["odysseus"])


class ResponsesProtocolClientTests(unittest.TestCase):
    """Requirements 6 and 7 of the Responses route, driven over real HTTP."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.api_key = ApiKey(
            id="local-operator", secret="conformance-fixture-key", aliases=("fixture-alias",)
        )
        self.stack = build_gateway_stack(
            Path(self._tmp.name),
            alias="fixture-alias",
            auth_mode="api_key",
            api_keys=(self.api_key,),
        )
        self.addCleanup(self.stack.stop)
        self.base_url = self.stack.server.base_url

    def _post(self, payload, **kwargs):
        kwargs.setdefault("api_key", self.api_key.secret)
        status, raw, headers = _request(self.base_url, "/v1/responses", payload, **kwargs)
        return status, raw, headers

    def test_non_streaming_response_is_a_response_object(self):
        status, raw, headers = self._post({"model": "fixture-alias", "input": "hello"})
        self.assertEqual(status, 200)
        document = json.loads(raw)
        self.assertEqual(document["object"], "response")
        self.assertEqual(document["status"], "completed")
        self.assertEqual(document["model"], "fixture-alias")
        self.assertIsInstance(document["output_text"], str)
        self.assertIn("X-PLAG-IN-Receipt-ID", headers)

    def test_streaming_response_is_an_ordered_event_stream(self):
        status, raw, headers = self._post(
            {"model": "fixture-alias", "input": "hello", "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        text = raw.decode("utf-8")
        self.assertTrue(text.startswith("event: response.created\n"))
        self.assertIn("event: response.output_text.delta\n", text)
        self.assertTrue(text.rstrip().endswith(text.rstrip().rsplit("\n", 1)[-1]))
        names = [line[len("event: "):] for line in text.splitlines() if line.startswith("event: ")]
        self.assertEqual(names[-1], "response.completed")

    def test_missing_authentication_is_refused_before_generation(self):
        status, raw, _headers = self._post({"model": "fixture-alias", "input": "x"}, api_key=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw)["error"]["type"], "authentication_error")

    def test_an_alias_the_key_does_not_permit_is_refused(self):
        status, raw, _headers = self._post({"model": "other-alias", "input": "x"})
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw)["error"]["type"], "authentication_error")

    def test_an_unserved_alias_is_an_invalid_request(self):
        key = ApiKey(id="wide", secret="wide-fixture-key")
        stack = build_gateway_stack(
            Path(self._tmp.name) / "wide",
            alias="fixture-alias",
            auth_mode="api_key",
            api_keys=(key,),
        )
        try:
            status, raw, _headers = _request(
                stack.server.base_url,
                "/v1/responses",
                {"model": "absent-alias", "input": "x"},
                api_key=key.secret,
            )
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(raw)["error"]["type"], "invalid_request")
        finally:
            stack.stop()

    def test_malformed_json_is_a_typed_invalid_request(self):
        status, raw, _headers = self._post(b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["type"], "invalid_request")

    def test_an_oversized_body_is_refused_with_a_typed_413(self):
        payload = json.dumps(
            {"model": "fixture-alias", "input": "x" * (MAX_REQUEST_BODY_BYTES + 1000)}
        ).encode("utf-8")
        status, raw, _headers = self._post(payload)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(raw)["error"]["type"], "payload_too_large")

    def test_a_malformed_content_length_is_refused_on_the_shared_handler(self):
        status, raw, _headers = self._post(
            {"model": "fixture-alias", "input": "x"}, headers={"Content-Length": "not-a-number"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["type"], "invalid_request")

    def test_tools_are_refused_over_the_wire(self):
        status, raw, _headers = self._post(
            {
                "model": "fixture-alias",
                "input": "x",
                "tools": [{"type": "function", "name": "f"}],
            }
        )
        self.assertEqual(status, 501)
        body = json.loads(raw)
        self.assertEqual(body["error"]["type"], "unsupported_capability")
        self.assertEqual(body["error"]["field"], "tools")

    def test_the_route_does_not_read_the_client_name(self):
        first = self._post({"model": "fixture-alias", "input": "hello"}, user_agent="probe-a/1")
        second = self._post({"model": "fixture-alias", "input": "hello"}, user_agent="probe-b/2")
        self.assertEqual(first[0], second[0])
        left, right = json.loads(first[1]), json.loads(second[1])
        for field in ("object", "status", "model", "output_text", "tools", "tool_choice"):
            self.assertEqual(left[field], right[field], field)

    def test_chat_completions_behaviour_is_unchanged_by_the_new_route(self):
        status, raw, _headers = _request(
            self.base_url,
            "/v1/chat/completions",
            {"model": "fixture-alias", "messages": [{"role": "user", "content": "hello"}]},
            api_key=self.api_key.secret,
        )
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["object"], "chat.completion")
        self.assertIn("choices", body)
        self.assertNotIn("output", body)

    def test_capabilities_publish_both_protocol_subsets(self):
        request = urllib.request.Request(self.base_url + "/plag-in/v1/capabilities")
        request.add_header("Authorization", f"Bearer {self.api_key.secret}")
        with urllib.request.urlopen(request, timeout=30) as response:
            document = json.loads(response.read())
        self.assertIn("chat_completions", document["protocol_subsets"])
        self.assertIn("responses", document["protocol_subsets"])
        self.assertEqual(
            document["endpoints"]["responses"], document["endpoints"]["chat_completions"]
        )
        self.assertEqual(
            document["protocol_subsets"]["responses"]["tool_execution"], "absent"
        )


class ConnectCommandFormatsTest(unittest.TestCase):
    def test_emits_env_json_and_curl(self):
        from plag_in.connect import emit

        for fmt in ("env", "manual", "openai-env", "json", "curl"):
            output = emit(fmt, "http://127.0.0.1:8080/v1", "k", "fixture-alias")
            self.assertIn("fixture-alias", output)

    def test_openai_style_export_states_local_only_boundary(self):
        from plag_in.connect import emit

        output = emit(
            "openai-env", "http://127.0.0.1:8080/v1", "k", "fixture-alias"
        )
        self.assertIn("Local loopback gateway only", output)
        self.assertIn("Remote provider: none", output)


if __name__ == "__main__":
    unittest.main()
