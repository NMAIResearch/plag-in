"""Engine adapter contract.

Capability values are `tested`, `declared`, `inferred` or `unknown`. Only
`tested` capabilities may appear as supported in the client capability
document; PLAG IN returns a typed error for anything else rather than
translating a request silently.
"""
from __future__ import annotations

from dataclasses import dataclass

CAPABILITY_STATES = ("tested", "declared", "inferred", "unknown")


@dataclass(frozen=True)
class EngineCapabilities:
    chat_completions: str = "unknown"
    completions: str = "unknown"
    embeddings: str = "unknown"
    streaming: str = "unknown"
    tool_call_transport: str = "unknown"

    def as_dict(self) -> dict:
        return {
            "chat_completions": self.chat_completions,
            "completions": self.completions,
            "embeddings": self.embeddings,
            "streaming": self.streaming,
            "tool_call_transport": self.tool_call_transport,
        }


class EngineAdapter:
    name = "base"

    def build_argv(
        self,
        executable: str,
        model_path: str,
        host: str,
        port: int,
        runtime=None,
        backend_api_key_file: str | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def health_path(self) -> str:
        return "/health"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()
