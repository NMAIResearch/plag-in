import unittest

from plag_in.config import RuntimeConfig
from plag_in.engines.llama_server import LlamaServerAdapter


class LlamaServerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = LlamaServerAdapter()

    def test_argv_is_a_plain_list_of_literal_strings(self):
        argv = self.adapter.build_argv("/opt/llama-server", "/models/a.gguf", "127.0.0.1", 8081)
        self.assertEqual(
            argv,
            [
                "/opt/llama-server", "--model", "/models/a.gguf", "--host", "127.0.0.1", "--port", "8081",
                "--ctx-size", "8192", "--n-gpu-layers", "auto", "--threads", "-1",
                "--batch-size", "2048", "--ubatch-size", "512", "--parallel", "1",
                "--temp", "0.8", "--top-p", "0.95",
                "--no-agent", "--no-ui-mcp-proxy", "--no-ui",
            ],
        )
        self.assertIsInstance(argv, list)
        self.assertTrue(all(isinstance(part, str) for part in argv))

    def test_metacharacters_in_model_path_stay_a_single_literal_element(self):
        hostile = "/models/a.gguf; rm -rf / #$(whoami)"
        argv = self.adapter.build_argv("/opt/llama-server", hostile, "127.0.0.1", 8081)
        self.assertIn(hostile, argv)
        self.assertEqual(argv.count(hostile), 1)
        # No element was split on the shell metacharacters.
        for part in argv:
            self.assertNotIn(";", part.replace(hostile, ""))

    def test_explicit_runtime_profile_is_materialised_in_argv(self):
        profile = RuntimeConfig(
            context_size=4096,
            gpu_layers="all",
            threads=6,
            batch_size=1024,
            ubatch_size=256,
            parallel=2,
            temperature=0.2,
            top_p=0.9,
        )
        argv = self.adapter.build_argv(
            "/opt/llama-server", "/models/a.gguf", "127.0.0.1", 8081, profile
        )
        for flag, value in (
            ("--ctx-size", "4096"),
            ("--n-gpu-layers", "all"),
            ("--threads", "6"),
            ("--batch-size", "1024"),
            ("--ubatch-size", "256"),
            ("--parallel", "2"),
            ("--temp", "0.2"),
            ("--top-p", "0.9"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_backend_key_file_is_passed_without_putting_the_secret_in_argv(self):
        argv = self.adapter.build_argv(
            "/opt/llama-server",
            "/models/a.gguf",
            "127.0.0.1",
            8081,
            backend_api_key_file="/state/engine_keys/a.key",
        )
        self.assertEqual(argv[argv.index("--api-key-file") + 1], "/state/engine_keys/a.key")
        self.assertNotIn("fixture-secret", argv)

    def test_capabilities_only_mark_what_is_actually_exercised(self):
        caps = self.adapter.capabilities().as_dict()
        self.assertEqual(caps["chat_completions"], "tested")
        self.assertEqual(caps["tool_call_transport"], "tested")
        self.assertEqual(caps["completions"], "unknown")
        self.assertEqual(caps["embeddings"], "unknown")


if __name__ == "__main__":
    unittest.main()
