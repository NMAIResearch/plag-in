import argparse
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from plag_in.cli import (
    _bounded_command,
    _model_load_safety,
    _require_runtime_confirmation,
    build_parser,
    cmd_doctor,
    cmd_inspect,
    main,
    run_private_chat,
    run_profile_gateway,
)
from plag_in.errors import ConfigurationError, ModelIdentityMismatchError
from plag_in.onboarding import doctor_report, render_doctor, run_setup_assistant


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class CliParserTests(unittest.TestCase):
    def test_expected_commands_present(self):
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
        commands = set(subparsers_action.choices.keys())
        self.assertEqual(
            commands,
            {
                "setup",
                "doctor",
                "inspect",
                "recommend",
                "serve",
                "chat",
                "gateway",
                "status",
                "stop",
                "verify",
                "receipt",
                "connect",
            },
        )

    def test_doctor_without_config_is_read_only_and_separates_controls(self):
        report = doctor_report()
        self.assertEqual(report["inspection"]["network_action"], "none")
        self.assertEqual(report["inspection"]["filesystem_changes"], "none")
        self.assertEqual(report["locality_controls"]["status"], "unconfigured")
        self.assertEqual(report["runtime_settings"]["status"], "unconfigured")
        self.assertNotIn("context_size", report["locality_controls"])

    def test_doctor_with_config_keeps_runtime_values_out_of_locality(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "bind": {"host": "127.0.0.1", "port": 19080},
                        "security": {"auth_mode": "none", "content_logging": False},
                        "runtime": {"context_size": 4096, "temperature": 0.2},
                    }
                )
            )
            report = cmd_doctor(argparse.Namespace(config=str(config_path)))
        self.assertEqual(report["locality_controls"]["active_evidence_level"], "L0")
        self.assertEqual(report["runtime_settings"]["values"]["context_size"], 4096)
        self.assertEqual(report["runtime_settings"]["values"]["temperature"], 0.2)
        self.assertNotIn("context_size", report["locality_controls"])
        self.assertNotIn("temperature", report["locality_controls"])

    def test_doctor_human_output_has_separate_headings(self):
        text = render_doctor(doctor_report())
        self.assertIn("Locality controls", text)
        self.assertIn("Runtime settings", text)

    def test_setup_explains_locality_without_writing(self):
        output = io.StringIO()
        choices = iter(["6", "8"])
        result = run_setup_assistant(input_fn=lambda _: next(choices), output=output)
        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("Privacy and locality controls", text)
        self.assertIn("Runtime settings are recorded for reproducibility", text)
        self.assertIn("Inspection is read-only", text)

    def test_buffered_setup_repetition_stops_at_safety_limit(self):
        output = io.StringIO()
        result = run_setup_assistant(input_fn=lambda _: "4", output=output, max_cycles=3)
        self.assertEqual(result, 2)
        self.assertIn("buffered-interface safety limit", output.getvalue())
        self.assertLess(len(output.getvalue()), 10_000)

    def test_nested_confirmation_interrupt_returns_to_menu_cleanly(self):
        output = io.StringIO()
        choices = iter(["2", "8"])

        def interrupted(config_path, input_fn, stream):
            raise KeyboardInterrupt

        result = run_setup_assistant(
            input_fn=lambda _: next(choices),
            output=output,
            configure_fn=interrupted,
        )
        self.assertEqual(result, 0)
        self.assertIn("Action cancelled", output.getvalue())
        self.assertIn("Setup closed", output.getvalue())

    def test_noninteractive_no_argument_invocation_prints_help(self):
        stderr = io.StringIO()
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stderr", stderr):
            result = main([])
        self.assertEqual(result, 2)
        self.assertIn("plag-in", stderr.getvalue())

    def test_noninteractive_setup_is_refused(self):
        stderr = io.StringIO()
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdout.isatty", return_value=False),
            patch("sys.stderr", stderr),
        ):
            result = main(["setup"])
        self.assertEqual(result, 2)
        self.assertIn("interactive_terminal_required", stderr.getvalue())

    def test_interactive_no_argument_invocation_opens_setup(self):
        stdout = _TTYBuffer()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout", stdout),
            patch("builtins.input", return_value="8"),
        ):
            result = main([])
        self.assertEqual(result, 0)
        self.assertIn("PLAG IN setup", stdout.getvalue())
        self.assertIn("Setup closed.", stdout.getvalue())

    def test_noninteractive_serve_requires_explicit_runtime_confirmation(self):
        args = argparse.Namespace(confirm_runtime_profile=False)
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConfigurationError) as caught:
                _require_runtime_confirmation(args, {"runtime": {"context_size": 8192}})
        self.assertIn("runtime_profile", caught.exception.fields)

    def test_explicit_runtime_confirmation_allows_noninteractive_launch_path(self):
        args = argparse.Namespace(confirm_runtime_profile=True)
        _require_runtime_confirmation(args, {"runtime": {"context_size": 8192}})

    def test_inspect_exposes_direct_profile_without_ollama_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"runtime": {"context_size": 4096}}))
            result = cmd_inspect(argparse.Namespace(config=str(config_path)))
        profile = result["direct_runtime_profile"]
        self.assertEqual(profile["engine"], "llama_server_direct")
        self.assertFalse(profile["ollama_runtime_dependency"])
        self.assertEqual(profile["arguments"]["context_size"], 4096)

    def test_inspect_requires_config(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["inspect"])

    def test_inspect_reports_embedded_engine_identity_without_loading_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lib = tmp_path / "libllama.so"
            lib.write_bytes(b"not-a-real-library")
            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
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
                        }
                    }
                )
            )
            result = cmd_inspect(argparse.Namespace(config=str(config_path)))
        report = result["engines"]["libllama"]
        self.assertEqual(report["kind"], "embedded")
        self.assertTrue(report["ggml_base_library_present"])
        self.assertEqual(report["declared_ggml_base_library_sha256"], "b" * 64)
        self.assertEqual(report["requested_runtime_profile"]["engine"], "libllama_embedded")
        self.assertFalse(report["requested_runtime_profile"]["ollama_runtime_dependency"])

    def _chat_config(self, root: Path) -> Path:
        model = root / "models" / "fixture.gguf"
        model.parent.mkdir()
        model.write_bytes(b"fixture-model")
        library = root / "libllama.so"
        library.write_bytes(b"fixture-library")
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model_roots": [str(model.parent)],
                    "engines": {
                        "libllama": {
                            "library": str(library),
                            "library_sha256": "a" * 64,
                            "ggml_base_library": str(library),
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": str(library),
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [
                                {"path": str(library), "sha256": "d" * 64}
                            ],
                            "upstream_identity": "fixture",
                        }
                    },
                    "profiles": {
                        "fixture": {
                            "model_path": str(model),
                            "engine": "libllama",
                            "display_name": "Fixture",
                            "compatibility_status": "tested",
                        }
                    },
                    "security": {
                        "auth_mode": "api_key",
                        "api_keys": [
                            {
                                "id": "operator",
                                "secret": "s" * 64,
                                "aliases": ["fixture"],
                                "origin": "loopback",
                            }
                        ],
                        "content_logging": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_private_chat_decline_starts_no_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = io.StringIO()
            with (
                patch("plag_in.cli._start_embedded_session") as start,
                patch("plag_in.cli._model_load_safety", return_value=(
                    {"memory.high": 10 * 1024**3, "memory.max": 12 * 1024**3,
                     "memory.swap.max": 2 * 1024**3},
                    {"available_ram_bytes": 32 * 1024**3,
                     "required_available_ram_bytes": 20 * 1024**3,
                     "gpu": None, "required_free_vram_mib": None},
                )),
            ):
                result = run_private_chat(
                    self._chat_config(root),
                    lambda _: "n",
                    output,
                    state_dir=root / "state",
                )
        self.assertEqual(result, 0)
        start.assert_not_called()
        self.assertIn("No listener or model was started", output.getvalue())

    def test_private_chat_sends_prompt_and_stops_session(self):
        class _Server:
            base_url = "http://127.0.0.1:19080"

        class _Receipts:
            @staticmethod
            def verify_chain():
                return True, 1

        class _Session:
            server = _Server()
            receipts = _Receipts()

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = io.StringIO()
            choices = iter(["yes", "hello", "/exit"])
            session = _Session()
            with (
                patch("plag_in.cli._start_embedded_session", return_value=session),
                patch("plag_in.cli._model_load_safety", return_value=(
                    {"memory.high": 10 * 1024**3, "memory.max": 12 * 1024**3,
                     "memory.swap.max": 2 * 1024**3},
                    {"available_ram_bytes": 32 * 1024**3,
                     "required_available_ram_bytes": 20 * 1024**3,
                     "gpu": None, "required_free_vram_mib": None},
                )),
                patch(
                    "plag_in.cli._local_chat_request",
                    return_value={"content": "local answer", "payload": {}},
                ) as request,
            ):
                result = run_private_chat(
                    self._chat_config(root),
                    lambda _: next(choices),
                    output,
                    state_dir=root / "state",
                )
        self.assertEqual(result, 0)
        self.assertTrue(session.closed)
        request.assert_called_once()
        self.assertIn("model> local answer", output.getvalue())
        self.assertIn("Receipt chain valid: true", output.getvalue())

    def test_bounded_chat_command_has_hard_memory_and_swap_controls(self):
        with patch("plag_in.cli.shutil.which", return_value="/usr/bin/systemd-run"):
            command = _bounded_command(["chat", "--config", "/tmp/config.json"])
        self.assertEqual(command[0], "/usr/bin/systemd-run")
        self.assertIn("--property=MemoryHigh=10737418240", command)
        self.assertIn("--property=MemoryMax=12884901888", command)
        self.assertIn("--property=MemorySwapMax=2147483648", command)
        self.assertIn("--property=OOMPolicy=kill", command)
        self.assertNotIn("sh", command)

    def test_chat_cli_relaunches_before_any_model_path_runs(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("plag_in.cli._launch_bounded_command", return_value=7) as launch,
        ):
            result = main(["chat", "--config", "/tmp/config.json"])
        self.assertEqual(result, 7)
        launch.assert_called_once_with(["chat", "--config", "/tmp/config.json"])

    def test_serve_cli_relaunches_before_argument_side_effects(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("plag_in.cli._launch_bounded_command", return_value=9) as launch,
        ):
            result = main([
                "serve", "--config", "/tmp/config.json", "--alias", "fixture",
                "--model-path", "/tmp/model.gguf", "--state-dir", "/tmp/state",
            ])
        self.assertEqual(result, 9)
        self.assertEqual(launch.call_args.args[0][0], "serve")

    def test_model_load_refuses_unbounded_cgroup_before_preflight(self):
        with (
            patch(
                "plag_in.cli.verify_chat_cgroup",
                side_effect=ConfigurationError("unbounded"),
            ),
            patch("plag_in.cli.resource_preflight") as preflight,
        ):
            with self.assertRaises(ConfigurationError):
                _model_load_safety(SimpleNamespace(size_bytes=1), "all")
        preflight.assert_not_called()

    def test_model_load_refuses_failed_resource_preflight(self):
        with (
            patch(
                "plag_in.cli.verify_chat_cgroup",
                return_value={"memory.high": 1, "memory.max": 2, "memory.swap.max": 1},
            ),
            patch(
                "plag_in.cli.resource_preflight",
                return_value={"status": "fail", "failures": ["fixture"]},
            ),
        ):
            with self.assertRaises(ConfigurationError):
                _model_load_safety(SimpleNamespace(size_bytes=1), "all")

    def test_harness_gateway_emits_generic_local_connector_and_stops(self):
        class _Server:
            base_url = "http://127.0.0.1:19080"

        class _Receipts:
            @staticmethod
            def verify_chain():
                return True, 0

        class _Session:
            server = _Server()
            receipts = _Receipts()

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = io.StringIO()
            session = _Session()
            with (
                patch("plag_in.cli._model_load_safety", return_value=(
                    {"memory.high": 10 * 1024**3, "memory.max": 12 * 1024**3,
                     "memory.swap.max": 2 * 1024**3},
                    {"available_ram_bytes": 32 * 1024**3,
                     "gpu": {"free_mib": 12000}},
                )),
                patch("plag_in.cli._start_embedded_session", return_value=session),
                patch("plag_in.cli.time.sleep", side_effect=KeyboardInterrupt),
            ):
                result = run_profile_gateway(
                    self._chat_config(root),
                    lambda _: "yes",
                    output,
                    state_dir=root / "state",
                    connector_format="manual",
                )
        self.assertEqual(result, 0)
        self.assertTrue(session.closed)
        text = output.getvalue()
        self.assertIn("Transport: local HTTP", text)
        self.assertIn("Protocol: Chat Completions v1 subset", text)
        self.assertIn("Remote provider: none", text)

    def test_setup_can_route_a_configured_profile_to_harness_connector(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            choices = iter(["4", "8"])
            calls = []

            def connect(path, input_fn, stream):
                calls.append(path)
                return 0

            result = run_setup_assistant(
                config_path,
                input_fn=lambda _: next(choices),
                output=output,
                connect_fn=connect,
            )
        self.assertEqual(result, 0)
        self.assertEqual(calls, [config_path])


class CliInspectScopeTests(unittest.TestCase):
    def _embedded_engine(self, library: Path) -> dict:
        return {
            "library": str(library),
            "library_sha256": "a" * 64,
            "ggml_base_library": str(library),
            "ggml_base_library_sha256": "b" * 64,
            "ggml_library": str(library),
            "ggml_library_sha256": "c" * 64,
            "backend_libraries": [{"path": str(library), "sha256": "d" * 64}],
            "upstream_identity": "fixture",
        }

    def _config_with_profile(self, root: Path) -> tuple[Path, Path, Path]:
        models = root / "models"
        models.mkdir()
        profile_model = models / "profile.gguf"
        profile_model.write_bytes(b"the-configured-profile-model")
        unrelated = models / "unrelated-large.gguf"
        unrelated.write_bytes(b"an unrelated held model that must not be hashed")
        library = root / "libllama.so"
        library.write_bytes(b"lib")
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model_roots": [str(models)],
                    "engines": {"libllama": self._embedded_engine(library)},
                    "profiles": {
                        "fixture": {
                            "model_path": str(profile_model),
                            "engine": "libllama",
                            "display_name": "Fixture",
                            "compatibility_status": "tested",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path, profile_model, unrelated

    def test_default_scope_inspects_only_configured_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, profile_model, _unrelated = self._config_with_profile(Path(tmp))
            with (
                patch("plag_in.cli.discover_gguf", side_effect=AssertionError("scanned all models")),
                patch(
                    "plag_in.cli.discover_ollama_models",
                    side_effect=AssertionError("scanned all ollama models"),
                ),
            ):
                result = cmd_inspect(argparse.Namespace(config=str(config_path)))
        self.assertEqual(result["scope"], "configured_profiles")
        self.assertNotIn("gguf_models", result)
        self.assertNotIn("ollama_models", result)
        self.assertEqual(len(result["profiles"]), 1)
        entry = result["profiles"][0]
        self.assertEqual(entry["alias"], "fixture")
        self.assertEqual(entry["status"], "inspected")
        self.assertEqual(
            entry["sha256"],
            hashlib.sha256(b"the-configured-profile-model").hexdigest(),
        )

    def test_all_models_scope_retains_full_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _profile_model, _unrelated = self._config_with_profile(Path(tmp))
            result = cmd_inspect(argparse.Namespace(config=str(config_path), all_models=True))
        self.assertEqual(result["scope"], "all_models")
        self.assertIn("gguf_models", result)
        hashed_paths = {item["path"] for item in result["gguf_models"]}
        self.assertTrue(any(p.endswith("unrelated-large.gguf") for p in hashed_paths))
        self.assertTrue(any(p.endswith("profile.gguf") for p in hashed_paths))

    def test_missing_profile_model_is_unassessed_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            library = root / "libllama.so"
            library.write_bytes(b"lib")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_roots": [str(models)],
                        "engines": {"libllama": self._embedded_engine(library)},
                        "profiles": {
                            "fixture": {
                                "model_path": str(models / "absent.gguf"),
                                "engine": "libllama",
                                "display_name": "Fixture",
                                "compatibility_status": "tested",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = cmd_inspect(argparse.Namespace(config=str(config_path)))
        entry = result["profiles"][0]
        self.assertEqual(entry["status"], "unassessed")
        self.assertEqual(entry["reason"], "path_containment_violation")
        self.assertNotIn("sha256", entry)

    def test_configured_ollama_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_root = root / "models"
            empty_root.mkdir()
            library = root / "libllama.so"
            library.write_bytes(b"lib")
            ollama_root = root / "ollama"
            manifests = ollama_root / "manifests" / "library" / "m"
            blobs = ollama_root / "blobs"
            manifests.mkdir(parents=True)
            blobs.mkdir(parents=True)
            declared = hashlib.sha256(b"the-real-bytes").hexdigest()
            blob_path = blobs / f"sha256-{declared}"
            blob_path.write_bytes(b"TAMPERED-bytes-that-do-not-match")
            (manifests / "latest").write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": f"sha256:{declared}",
                            }
                        ]
                    }
                )
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_roots": [str(empty_root)],
                        "ollama_manifest_root": str(ollama_root),
                        "engines": {"libllama": self._embedded_engine(library)},
                        "profiles": {
                            "fixture": {
                                "model_path": str(blob_path),
                                "engine": "libllama",
                                "display_name": "Fixture",
                                "compatibility_status": "tested",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ModelIdentityMismatchError):
                cmd_inspect(argparse.Namespace(config=str(config_path)))

    def test_inspect_all_models_parser_flag_present(self):
        parser = build_parser()
        args = parser.parse_args(["inspect", "--config", "/tmp/config.json", "--all-models"])
        self.assertTrue(args.all_models)
        default_args = parser.parse_args(["inspect", "--config", "/tmp/config.json"])
        self.assertFalse(default_args.all_models)


if __name__ == "__main__":
    unittest.main()
