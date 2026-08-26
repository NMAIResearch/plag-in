"""Regressions 13 and 14: the embedded `serve` foreground path.

`EmbeddedLibLlama` is replaced with a fixture double so no native call is
ever made; these tests exercise `cli.cmd_serve`'s process- and
listener-lifecycle boundary only (one loopback listener, no inference
child, and clean release on a failed load).
"""
from __future__ import annotations

import argparse
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plag_in import cli
from plag_in.errors import NativeRuntimeError
from tests.support import free_port


class _FakeIdentity:
    def digest(self) -> str:
        return "e" * 64


class _FakeEmbedded:
    instances: list["_FakeEmbedded"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.load_called = False
        self.close_called = False
        self.identity = _FakeIdentity()
        type(self).instances.append(self)

    def load(self) -> None:
        self.load_called = True

    def ready(self) -> bool:
        return True

    @property
    def runtime_profile(self) -> dict:
        return {"engine": "libllama_embedded", "ollama_runtime_dependency": False}

    @property
    def template_digest(self) -> str:
        return "d" * 64

    def chat_completion(self, body: dict) -> dict:
        return {"choices": [], "usage": {}}

    def close(self) -> None:
        self.close_called = True


class _FailingEmbedded(_FakeEmbedded):
    def load(self) -> None:
        super().load()
        raise NativeRuntimeError("fixture load failure")


def _write_config(tmp_path: Path, port: int) -> Path:
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"fixture")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_roots": [str(tmp_path)],
                "bind": {"host": "127.0.0.1", "port": port},
                "engines": {
                    "libllama": {
                        "library": str(lib),
                        "library_sha256": "a" * 64,
                        "ggml_base_library": str(lib),
                        "ggml_base_library_sha256": "b" * 64,
                        "ggml_library": str(lib),
                        "ggml_library_sha256": "c" * 64,
                        "backend_libraries": [{"path": str(lib), "sha256": "d" * 64}],
                        "upstream_identity": "fixture-upstream",
                    }
                },
            }
        )
    )
    return config_path


def _serve_args(config_path: Path, model_path: Path, state_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config_path),
        alias="embedded-fixture",
        model_path=str(model_path),
        engine="libllama",
        engine_host="127.0.0.1",
        engine_port=None,
        ready_timeout=5.0,
        confirm_runtime_profile=True,
        state_dir=str(state_dir),
    )


class EmbeddedServeForegroundTests(unittest.TestCase):
    def setUp(self):
        _FakeEmbedded.instances = []

    def test_embedded_serve_has_one_listener_and_no_inference_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.gguf"
            model_path.write_bytes(b"GGUF-fixture")
            port = free_port()
            config_path = _write_config(tmp_path, port)
            state_dir = tmp_path / "state"
            args = _serve_args(config_path, model_path, state_dir)

            def _raise_keyboard_interrupt(_seconds):
                raise KeyboardInterrupt

            with patch("plag_in.cli.EmbeddedLibLlama", _FakeEmbedded), patch(
                "plag_in.cli.time.sleep", side_effect=_raise_keyboard_interrupt
            ), patch(
                "plag_in.cli._model_load_safety",
                return_value=(
                    {"memory.high": 1, "memory.max": 2, "memory.swap.max": 1},
                    {"available_ram_bytes": 4, "required_available_ram_bytes": 3,
                     "gpu": None, "required_free_vram_mib": None},
                ),
            ):
                result = cli.cmd_serve(args)

            self.assertEqual(result, {"alias": "embedded-fixture", "state": "stopped"})
            self.assertEqual(len(_FakeEmbedded.instances), 1)
            self.assertTrue(_FakeEmbedded.instances[0].load_called)
            self.assertTrue(_FakeEmbedded.instances[0].close_called)
            # The worker-adapter session mechanism is never touched by the
            # embedded route: no inference child, no session record.
            sessions_dir = state_dir / "sessions"
            self.assertTrue(sessions_dir.is_dir())
            self.assertEqual(list(sessions_dir.iterdir()), [])

    def test_failed_embedded_load_releases_the_listener_and_leaves_no_running_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.gguf"
            model_path.write_bytes(b"GGUF-fixture")
            port = free_port()
            config_path = _write_config(tmp_path, port)
            state_dir = tmp_path / "state"
            args = _serve_args(config_path, model_path, state_dir)

            with patch("plag_in.cli.EmbeddedLibLlama", _FailingEmbedded), patch(
                "plag_in.cli._model_load_safety",
                return_value=(
                    {"memory.high": 1, "memory.max": 2, "memory.swap.max": 1},
                    {"available_ram_bytes": 4, "required_available_ram_bytes": 3,
                     "gpu": None, "required_free_vram_mib": None},
                ),
            ):
                with self.assertRaises(NativeRuntimeError):
                    cli.cmd_serve(args)

            self.assertTrue(_FakeEmbedded.instances[0].load_called)
            self.assertTrue(_FakeEmbedded.instances[0].close_called)
            sessions_dir = state_dir / "sessions"
            self.assertTrue(not sessions_dir.is_dir() or list(sessions_dir.iterdir()) == [])

            # The listener must have been released: a fresh bind to the
            # exact same host:port must succeed.
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", port))
            finally:
                probe.close()


if __name__ == "__main__":
    unittest.main()
