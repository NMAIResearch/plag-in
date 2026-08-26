"""CLIENT_INTERFACE.md acceptance: an Odysseus-shaped client reaches a fake
model through the OpenAI-compatible route, using only base URL, key and
model alias, with no gateway-side Odysseus code."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plag_in.config import ApiKey

from tests.fixtures.odysseus_client import OdysseusShapedClient
from tests.support import build_gateway_stack


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
