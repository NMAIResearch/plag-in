"""Shared test helpers. Every helper here operates on scratch tempdirs only."""
from __future__ import annotations

import contextlib
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from plag_in.config import ApiKey, BindConfig, GatewayConfig, SecurityConfig
from plag_in.engines.llama_server import LlamaServerAdapter
from plag_in.gateway import BackendInfo, GatewayContext, GatewayServer
from plag_in.identity import ModelIdentity, canonical_digest
from plag_in.receipts import ReceiptStore
from plag_in.registry import Registry, RegistryEntry
from plag_in.supervisor import Supervisor

FAKE_ENGINE_PATH = Path(__file__).parent / "fixtures" / "fake_engine.py"
ENGINE_ADAPTERS = {"llama_server": LlamaServerAdapter()}


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_fixture_weight(path: Path, content: bytes = b"fixture-weight-bytes-v1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def python_executable() -> str:
    return sys.executable


@dataclass
class GatewayStack:
    supervisor: Supervisor
    context: GatewayContext
    server: GatewayServer
    identity: ModelIdentity
    alias: str
    engine_port: int
    receipts_path: Path
    hmac_key_path: Path

    def stop(self) -> None:
        self.server.stop()
        self.supervisor.stop(self.alias)


def build_gateway_stack(
    tmp_dir: Path,
    *,
    alias: str = "fixture-alias",
    model_content: bytes = b"fixture-weight-bytes-v1",
    bind_host: str = "127.0.0.1",
    auth_mode: str = "none",
    api_keys: tuple[ApiKey, ...] = (),
    with_tool_call: bool = False,
) -> GatewayStack:
    """Stand up a real (loopback) fake engine plus a real gateway server, entirely on scratch state."""
    tmp_dir = Path(tmp_dir)
    model_path = write_fixture_weight(tmp_dir / "model.gguf", model_content)
    identity = ModelIdentity.from_file(model_path)

    registry = Registry()
    config = GatewayConfig(
        model_roots=(tmp_dir,),
        bind=BindConfig(host=bind_host, port=0),
        security=SecurityConfig(auth_mode=auth_mode, api_keys=api_keys),
    )
    config_digest = canonical_digest(
        {"engine": "llama_server", "runtime": config.runtime.as_dict()}
    )
    registry.bind(
        RegistryEntry(alias=alias, identity=identity, engine="llama_server", engine_config_digest=config_digest)
    )

    supervisor = Supervisor(tmp_dir / "sessions")
    engine_port = free_port()
    backend_api_key = "fixture-backend-key-7f9b"
    backend_api_key_path = tmp_dir / "backend_api_key.txt"
    backend_api_key_path.write_text(backend_api_key + "\n", encoding="utf-8")
    backend_api_key_path.chmod(0o600)
    argv = ENGINE_ADAPTERS["llama_server"].build_argv(
        str(FAKE_ENGINE_PATH),
        str(model_path),
        "127.0.0.1",
        engine_port,
        config.runtime,
        str(backend_api_key_path),
    )
    if with_tool_call:
        argv.append("--with-tool-call")
    session = supervisor.start(
        alias=alias,
        argv=argv,
        allowlisted_executable=str(FAKE_ENGINE_PATH),
        host="127.0.0.1",
        port=engine_port,
        health_path="/health",
    )

    receipts_path = tmp_dir / "receipts.jsonl"
    hmac_key_path = tmp_dir / "receipts.jsonl.hmac_key"
    receipt_store = ReceiptStore(receipts_path, hmac_key_path=hmac_key_path)
    context = GatewayContext(config, registry, receipt_store, ENGINE_ADAPTERS)
    direct_record = config.runtime.as_direct_record()
    context.register_backend(
        alias,
        BackendInfo(
            engine="llama_server",
            engine_version="fixture",
            inference_mode="worker",
            host="127.0.0.1",
            port=engine_port,
            weight_digest=identity.sha256,
            config_digest=config_digest,
            executable_digest=session.executable_digest,
            argv_digest=session.argv_digest,
            requested_runtime_profile=direct_record.get("arguments"),
            measurement_status=direct_record.get("measurement_status"),
            gpu_offload=direct_record.get("gpu_offload"),
            runtime_profile=direct_record,
            backend_api_key=backend_api_key,
        ),
    )
    server = GatewayServer(context)
    server.start()

    return GatewayStack(
        supervisor=supervisor,
        context=context,
        server=server,
        identity=identity,
        alias=alias,
        engine_port=engine_port,
        receipts_path=receipts_path,
        hmac_key_path=hmac_key_path,
    )
