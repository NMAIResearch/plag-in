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
    _assess_trial_admission,
    _bounded_command,
    _identity_instruction,
    _inventory_rows,
    _model_load_safety,
    _prepare_receipt_state,
    _setup_configure,
    _require_runtime_confirmation,
    _trial_alias_for,
    build_parser,
    cmd_doctor,
    cmd_inspect,
    main,
    run_private_chat,
    run_profile_gateway,
)
from plag_in.config import load_config_file
from plag_in.errors import ConfigurationError, ModelIdentityMismatchError
from plag_in.inventory import (
    AVAILABILITY_LOCAL_COMPLETE,
    AVAILABILITY_LOCAL_INCOMPLETE,
    AVAILABILITY_REMOTE_ONLY,
    HeldModel,
)
from plag_in.onboarding import doctor_report, render_doctor, run_setup_assistant
from plag_in.identity import hash_file
from tests.support import FIXTURE_COMPATIBILITY_RECORD, registered_fixture_compatibility
from tests.unit.test_gguf_metadata import build_gguf


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class CliParserTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(registered_fixture_compatibility())

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
            patch(
                "plag_in.input_control._read_terminal_key",
                side_effect=["8", "\r"],
            ),
        ):
            result = main([])
        self.assertEqual(result, 0)
        self.assertIn("PLAG IN setup", stdout.getvalue())
        self.assertIn("Setup closed.", stdout.getvalue())

    def test_receipt_migration_decline_occurs_before_model_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "receipts.jsonl").write_text(
                json.dumps({"schema_version": "3"}) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("plag_in.cli._initialise_receipt_state") as initialise:
                ready = _prepare_receipt_state(state, lambda _: "n", output)
        self.assertFalse(ready)
        initialise.assert_not_called()
        self.assertIn("Receipt migration required before inference", output.getvalue())
        self.assertIn("No listener or model was started", output.getvalue())

    def test_receipt_migration_confirmation_activates_current_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "receipts.jsonl").write_text(
                json.dumps({"schema_version": "3"}) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            activated = {
                "active_pointer": {"current_store": str(state / "receipts.v5.jsonl")}
            }
            with patch(
                "plag_in.cli._initialise_receipt_state", return_value=activated
            ) as initialise:
                ready = _prepare_receipt_state(state, lambda _: "yes", output)
        self.assertTrue(ready)
        initialise.assert_called_once()
        self.assertIn("Historical receipt bytes were not modified", output.getvalue())

    def test_setup_lists_unverified_held_model_without_configuring_it(self):
        held = HeldModel(
            tag="other:7b",
            source="ollama_manifest",
            availability=AVAILABILITY_LOCAL_COMPLETE,
            reason="",
            blob_path="/tmp/other",
            declared_digest="sha256:" + "b" * 64,
            declared_size_bytes=7 * 1024**3,
            held_size_bytes=7 * 1024**3,
        )
        candidate = {
            "available": True,
            "model": {"declared_sha256": "a" * 64},
        }
        output = io.StringIO()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[held]),
            patch("plag_in.cli.detect_tested_candidate", return_value=candidate),
            patch("plag_in.cli.configure_tested_model") as configure,
        ):
            result = _setup_configure(None, lambda _: "1", output)
        self.assertIsNone(result)
        configure.assert_not_called()
        self.assertIn("held locally", output.getvalue())
        self.assertIn("No configuration was written", output.getvalue())
        self.assertIn("other:7b", output.getvalue())

    def test_setup_keeps_a_non_startable_entry_visible_with_its_reason(self):
        remote_only = HeldModel(
            tag="remote:cloud",
            source="ollama_manifest",
            availability=AVAILABILITY_REMOTE_ONLY,
            reason="manifest_declares_no_model_layer",
            blob_path="",
            declared_digest="",
            declared_size_bytes=None,
            held_size_bytes=None,
        )
        output = io.StringIO()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[remote_only]),
            patch("plag_in.cli.detect_tested_candidate", return_value={"available": False}),
            patch("plag_in.cli.configure_tested_model") as configure,
        ):
            _setup_configure(None, lambda _: "1", output)
        configure.assert_not_called()
        text = output.getvalue()
        self.assertIn("remote:cloud", text)
        self.assertIn("remote_only", text)
        self.assertIn("manifest_declares_no_model_layer", text)
        self.assertIn("not the same as the model being unsupported", text)

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
                            "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
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
        # A1: the shutdown line reports storage integrity and names its scope.
        # It must not read as a specification-conformance result.
        self.assertIn("Receipt chain integrity valid: true", output.getvalue())
        self.assertIn("not specification conformance", output.getvalue())

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


_SAFETY = (
    {"memory.high": 10 * 1024**3, "memory.max": 12 * 1024**3, "memory.swap.max": 2 * 1024**3},
    {
        "available_ram_bytes": 32 * 1024**3,
        "required_available_ram_bytes": 20 * 1024**3,
        "gpu": None,
        "required_free_vram_mib": None,
    },
)

_ADMITTED_RESOURCES = {
    "outcome": "admitted",
    "policy_validation_status": "provisional_unvalidated",
    "cgroup": _SAFETY[0],
    "preflight": _SAFETY[1],
    "failures": [],
    "unassessed": [],
}


class _FakeSession:
    """Records what a session was asked to serve, without loading anything."""

    class _Server:
        base_url = "http://127.0.0.1:19080"

    class _Receipts:
        @staticmethod
        def verify_chain():
            return True, 1

    def __init__(self):
        self.server = self._Server()
        self.receipts = self._Receipts()
        self.closed = False

    def close(self):
        self.closed = True


class TrialAdmissionTests(unittest.TestCase):
    """Probes 1, 2, 11 and 12: the unverified trial route.

    Every fixture is a scratch GGUF file. No native library, no held model
    and no real host measurement is used.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.tested = build_gguf(self.models / "tested.gguf", architecture="tested-arch")
        self.held = build_gguf(self.models / "held.gguf", architecture="held-arch")
        # The record is registered against the bytes actually written, so
        # the tested route is exercised on matching evidence rather than on
        # a stored label the derivation would refuse.
        self.enterContext(
            registered_fixture_compatibility(model_sha256=hash_file(self.tested))
        )

    def _config_path(self) -> Path:
        library = self.root / "libllama.so"
        library.write_bytes(b"fixture-library")
        path = self.root / "config.json"
        path.write_text(
            json.dumps(
                {
                    "model_roots": [str(self.models)],
                    "engines": {
                        "libllama": {
                            "library": str(library),
                            "library_sha256": "a" * 64,
                            "ggml_base_library": str(library),
                            "ggml_base_library_sha256": "b" * 64,
                            "ggml_library": str(library),
                            "ggml_library_sha256": "c" * 64,
                            "backend_libraries": [{"path": str(library), "sha256": "d" * 64}],
                            "upstream_identity": "fixture",
                        }
                    },
                    "profiles": {
                        "tested-fixture": {
                            "model_path": str(self.tested),
                            "engine": "libllama",
                            "display_name": "Tested Fixture",
                            "compatibility_status": "tested",
                            "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
                        }
                    },
                    "security": {
                        "auth_mode": "api_key",
                        "api_keys": [
                            {"id": "operator", "secret": "s" * 64, "origin": "loopback"}
                        ],
                        "content_logging": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def _held_entry(self, path: Path | None = None, **overrides) -> HeldModel:
        target = path or self.held
        fields = {
            "tag": target.name,
            "source": "gguf_root",
            "availability": AVAILABILITY_LOCAL_COMPLETE,
            "reason": "",
            "blob_path": str(target),
            "declared_digest": "",
            "declared_size_bytes": target.stat().st_size,
            "held_size_bytes": target.stat().st_size,
        }
        fields.update(overrides)
        return HeldModel(**fields)

    # -- probe 1: the tested route is unchanged ---------------------------

    def test_tested_profiles_are_listed_first(self):
        config = load_config_file(self._config_path())
        rows = _inventory_rows(config, None)
        self.assertEqual(rows[0].compatibility_status, "tested")
        self.assertEqual(rows[0].alias, "tested-fixture")
        self.assertTrue(rows[0].startable)
        self.assertIn("tested-fixture", rows[0].label)

    def test_the_configured_tested_model_is_not_repeated_as_unverified(self):
        config = load_config_file(self._config_path())
        rows = _inventory_rows(config, None)
        self.assertEqual([row.compatibility_status for row in rows], ["tested", "unverified"])
        self.assertNotIn(str(self.tested), [row.entry.blob_path for row in rows if row.entry])

    def test_a_chat_without_a_trial_selects_the_tested_profile_unchanged(self):
        output = io.StringIO()
        session = _FakeSession()
        with (
            patch("plag_in.cli._start_embedded_session", return_value=session) as start,
            patch("plag_in.cli._model_load_safety", return_value=_SAFETY),
            patch(
                "plag_in.cli._local_chat_request",
                return_value={"content": "answer", "payload": {}},
            ),
        ):
            choices = iter(["yes", "/exit"])
            run_private_chat(
                self._config_path(),
                lambda _: next(choices),
                output,
                state_dir=self.root / "state",
            )
        _config, alias, profile, _state = start.call_args.args
        self.assertEqual(alias, "tested-fixture")
        self.assertEqual(profile.compatibility_status, "tested")
        self.assertIn("Compatibility: tested", output.getvalue())

    # -- probe 2: an unverified model runs without a persistent change ----

    def test_an_approved_trial_carries_an_unverified_ephemeral_profile(self):
        config_path = self._config_path()
        before = config_path.read_bytes()
        output = io.StringIO()
        session = _FakeSession()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[self._held_entry()]),
            patch(
                "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
            ),
            patch("plag_in.cli._start_embedded_session", return_value=session) as start,
            patch("plag_in.cli._model_load_safety", return_value=_SAFETY),
            patch(
                "plag_in.cli._local_chat_request",
                return_value={"content": "answer", "payload": {}},
            ),
        ):
            choices = iter(["yes", "yes", "/exit"])
            run_private_chat(
                config_path,
                lambda _: next(choices),
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        trial_config, alias, profile, _state = start.call_args.args
        self.assertEqual(alias, "trial-held.gguf")
        self.assertEqual(profile.compatibility_status, "unverified")
        self.assertEqual(str(profile.model_path), str(self.held))
        self.assertEqual(config_path.read_bytes(), before, "no configuration may be written")
        self.assertEqual(sorted(trial_config.profiles), ["tested-fixture", "trial-held.gguf"])

    def test_the_trial_disclosure_states_every_confirmed_value(self):
        output = io.StringIO()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[self._held_entry()]),
            patch(
                "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
            ),
            patch("plag_in.cli._start_embedded_session") as start,
        ):
            run_private_chat(
                self._config_path(),
                lambda _: "n",
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        text = output.getvalue()
        start.assert_not_called()
        for expected in (
            "Complete SHA-256:",
            "Held size:",
            "Architecture: held-arch",
            "Declared context length: 4096",
            "Chat template present: true",
            "Unassessed metadata:",
            "Requested runtime:",
            "Cgroup memory maximum:",
            "Available RAM:",
            "This is an unverified local trial",
            "No listener or model was started",
        ):
            self.assertIn(expected, text, expected)

    def test_a_declined_trial_starts_nothing(self):
        output = io.StringIO()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[self._held_entry()]),
            patch(
                "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
            ),
            patch("plag_in.cli._start_embedded_session") as start,
        ):
            result = run_private_chat(
                self._config_path(),
                lambda _: "n",
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        self.assertEqual(result, 0)
        start.assert_not_called()

    def test_a_trial_alias_is_derived_from_a_tag_without_inventing_identity(self):
        self.assertEqual(_trial_alias_for("qwen2.5:3b"), "trial-qwen2.5-3b")
        self.assertEqual(_trial_alias_for("a/b"), "trial-a-b")
        self.assertLessEqual(len(_trial_alias_for("x" * 200)), 64)

    # -- probe 11: a refused trial starts and records nothing -------------

    def test_a_remote_only_entry_is_refused_before_any_load(self):
        output = io.StringIO()
        entry = self._held_entry(
            availability=AVAILABILITY_REMOTE_ONLY,
            reason="manifest_declares_no_model_layer",
            blob_path="",
        )
        with (
            patch("plag_in.cli.discover_held_models", return_value=[entry]),
            patch("plag_in.cli._start_embedded_session") as start,
        ):
            result = run_private_chat(
                self._config_path(),
                lambda _: "yes",
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        self.assertEqual(result, 0)
        start.assert_not_called()
        self.assertIn("manifest_declares_no_model_layer", output.getvalue())
        self.assertIn("No listener, engine or model was started", output.getvalue())

    def test_a_non_gguf_selection_is_refused_as_unsupported(self):
        other = self.models / "not-a-model.bin"
        other.write_bytes(b"this is not a GGUF file")
        record = _assess_trial_admission(
            load_config_file(self._config_path()), self._held_entry(other), "libllama"
        )
        self.assertEqual(record["admission"], "refused")
        self.assertEqual(record["compatibility"], "unsupported")
        self.assertEqual(record["reason"], "not_gguf")

    def test_a_model_without_a_chat_template_is_refused_as_unsupported(self):
        bare = build_gguf(self.models / "bare.gguf", chat_template=None)
        record = _assess_trial_admission(
            load_config_file(self._config_path()), self._held_entry(bare), "libllama"
        )
        self.assertEqual(record["compatibility"], "unsupported")
        self.assertEqual(record["reason"], "gguf_declares_no_chat_template")

    def test_a_resource_refusal_is_not_reported_as_incompatibility(self):
        refused = dict(_ADMITTED_RESOURCES, outcome="resource_refused", failures=["no RAM"])
        with patch("plag_in.cli.evaluate_model_admission", return_value=refused):
            record = _assess_trial_admission(
                load_config_file(self._config_path()), self._held_entry(), "libllama"
            )
        self.assertEqual(record["admission"], "refused")
        self.assertEqual(record["reason"], "resource_refused")
        self.assertEqual(record["compatibility"], "unverified")
        self.assertEqual(record["availability"], AVAILABILITY_LOCAL_COMPLETE)

    def test_an_absent_blob_is_availability_not_compatibility(self):
        entry = self._held_entry(
            availability=AVAILABILITY_LOCAL_INCOMPLETE, reason="declared_blob_absent"
        )
        record = _assess_trial_admission(
            load_config_file(self._config_path()), entry, "libllama"
        )
        self.assertEqual(record["reason"], "declared_blob_absent")
        self.assertEqual(record["compatibility"], "unverified")

    def test_an_unknown_trial_model_is_refused(self):
        with patch("plag_in.cli.discover_held_models", return_value=[]):
            with self.assertRaises(ConfigurationError):
                run_private_chat(
                    self._config_path(),
                    lambda _: "yes",
                    io.StringIO(),
                    state_dir=self.root / "state",
                    trial_model="absent:tag",
                )

    # -- probe 12: the identity instruction -------------------------------

    def test_the_identity_instruction_uses_verified_local_metadata(self):
        config = load_config_file(self._config_path())
        with patch(
            "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
        ):
            record = _assess_trial_admission(config, self._held_entry(), "libllama")
        instruction = _identity_instruction("trial-held", record)
        self.assertIn("trial-held", instruction)
        self.assertIn(record["identity"]["sha256"], instruction)
        self.assertIn("held-arch", instruction)
        self.assertIn("4096", instruction)

    def test_the_identity_instruction_names_no_provider(self):
        config = load_config_file(self._config_path())
        with patch(
            "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
        ):
            record = _assess_trial_admission(config, self._held_entry(), "libllama")
        lowered = _identity_instruction("trial-held", record).lower()
        # "meta" is excluded from this list: it is a substring of "metadata",
        # which the instruction states truthfully about what it read.
        for name in ("openai", "anthropic", "claude", "google", "gemini", "mistral", "qwen"):
            self.assertNotIn(name, lowered, name)

    def test_the_identity_instruction_leads_the_private_chat_conversation(self):
        output = io.StringIO()
        session = _FakeSession()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[self._held_entry()]),
            patch(
                "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
            ),
            patch("plag_in.cli._start_embedded_session", return_value=session),
            patch("plag_in.cli._model_load_safety", return_value=_SAFETY),
            patch(
                "plag_in.cli._local_chat_request",
                return_value={"content": "answer", "payload": {}},
            ) as request,
        ):
            choices = iter(["yes", "yes", "who are you", "/exit"])
            run_private_chat(
                self._config_path(),
                lambda _: next(choices),
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        messages = request.call_args.args[3]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("trial-held.gguf", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "who are you"})

    def test_a_tested_chat_carries_no_identity_instruction(self):
        output = io.StringIO()
        session = _FakeSession()
        with (
            patch("plag_in.cli._start_embedded_session", return_value=session),
            patch("plag_in.cli._model_load_safety", return_value=_SAFETY),
            patch(
                "plag_in.cli._local_chat_request",
                return_value={"content": "answer", "payload": {}},
            ) as request,
        ):
            choices = iter(["yes", "hello", "/exit"])
            run_private_chat(
                self._config_path(),
                lambda _: next(choices),
                output,
                state_dir=self.root / "state",
            )
        messages = request.call_args.args[3]
        self.assertEqual(messages[0], {"role": "user", "content": "hello"})

    def test_the_chat_states_that_self_description_is_not_identity_evidence(self):
        output = io.StringIO()
        session = _FakeSession()
        with (
            patch("plag_in.cli.discover_held_models", return_value=[self._held_entry()]),
            patch(
                "plag_in.cli.evaluate_model_admission", return_value=dict(_ADMITTED_RESOURCES)
            ),
            patch("plag_in.cli._start_embedded_session", return_value=session),
            patch("plag_in.cli._model_load_safety", return_value=_SAFETY),
            patch(
                "plag_in.cli._local_chat_request",
                return_value={"content": "answer", "payload": {}},
            ),
        ):
            choices = iter(["yes", "yes", "/exit"])
            run_private_chat(
                self._config_path(),
                lambda _: next(choices),
                output,
                state_dir=self.root / "state",
                trial_model="held.gguf",
            )
        self.assertIn("not identity evidence", output.getvalue())

    def test_the_harness_gateway_never_injects_an_identity_instruction(self):
        source = Path(run_profile_gateway.__code__.co_filename).read_text(encoding="utf-8")
        gateway_body = source.split("def run_profile_gateway", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("_identity_instruction", gateway_body)

    def test_the_trial_flags_are_available_on_both_serving_commands(self):
        parser = build_parser()
        for command, extra in (("chat", []), ("gateway", [])):
            args = parser.parse_args(
                [command, "--config", "/tmp/c.json", "--state-dir", "/tmp/s",
                 "--trial-model", "held:latest", *extra]
            )
            self.assertEqual(args.trial_model, "held:latest")
            self.assertIsNone(args.trial_alias)


class CliInspectScopeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(registered_fixture_compatibility())

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
                            "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
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
                                "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
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
                                "compatibility_record": FIXTURE_COMPATIBILITY_RECORD.record_id,
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
