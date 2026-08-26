"""Direct `llama-server` adapter.

The argument array is built as a plain list and handed to the supervisor,
which launches it without a shell. A value containing shell metacharacters
(for example a path with `;` or `$(...)`) is passed as one literal argv
element and is never interpreted.
"""
from __future__ import annotations

from plag_in.config import RuntimeConfig
from plag_in.engines.base import EngineAdapter, EngineCapabilities


class LlamaServerAdapter(EngineAdapter):
    name = "llama_server"

    def build_argv(
        self,
        executable: str,
        model_path: str,
        host: str,
        port: int,
        runtime: RuntimeConfig | None = None,
        backend_api_key_file: str | None = None,
    ) -> list[str]:
        profile = runtime or RuntimeConfig()
        argv = [
            executable,
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--ctx-size", str(profile.context_size),
            "--n-gpu-layers", profile.gpu_layers,
            "--threads", str(profile.threads),
            "--batch-size", str(profile.batch_size),
            "--ubatch-size", str(profile.ubatch_size),
            "--parallel", str(profile.parallel),
            "--temp", str(profile.temperature),
            "--top-p", str(profile.top_p),
            "--no-agent",
            "--no-ui-mcp-proxy",
            "--no-ui",
        ]
        if backend_api_key_file:
            argv.extend(["--api-key-file", backend_api_key_file])
        return argv

    def health_path(self) -> str:
        return "/health"

    def capabilities(self) -> EngineCapabilities:
        # Only the paths exercised against a fixture engine in the default
        # test suite are marked "tested". Everything else stays "unknown"
        # so the gateway reports a typed unsupported-capability error
        # instead of guessing.
        return EngineCapabilities(
            chat_completions="tested",
            completions="unknown",
            embeddings="unknown",
            streaming="unknown",
            tool_call_transport="tested",
        )
