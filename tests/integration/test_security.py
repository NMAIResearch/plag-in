"""Required security tests.

Covers both the original handoff bullets (02_SONNET_MVP_IMPLEMENTATION.md)
and the F1-F9 defects plus the twenty numbered regressions from
CODEX_REVIEW_MVP_2026-08-25.md / 03_SONNET_MVP_SECURITY_REPAIR.md. Each
test method name states what it covers; classes are grouped by defect
where practical. All fixtures are scratch: a tempdir, the fake_engine.py
fixture process and real loopback sockets. No host-installed service or
live model store is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from plag_in.config import ApiKey, BindConfig, GatewayConfig, SecurityConfig, load_config
from plag_in.engines.llama_server import LlamaServerAdapter
from plag_in.errors import (
    ConfigurationError,
    ExecutableNotAllowedError,
    InvalidAliasError,
    LocalityPolicyError,
    ModelIdentityMismatchError,
    PathContainmentError,
    ProcessIdentityError,
    ReceiptPersistenceError,
    UnknownConfigurationFieldError,
)
from plag_in.identity import ModelIdentity, canonical_digest
from plag_in.model_paths import resolve_contained_model
from plag_in.receipts import Receipt, ReceiptStore
from plag_in.registry import Registry, RegistryEntry
from plag_in.supervisor import Supervisor

import plag_in.gateway as gateway_module
import plag_in.recommend as recommend_module
import plag_in.cli as cli_module
from plag_in.gateway import BackendInfo, GatewayContext, GatewayServer

from tests.support import FAKE_ENGINE_PATH, build_gateway_stack, free_port, write_fixture_weight

_SECRET_FIXTURE = "sk-fixture-secret-3e9f7a1c"
_PROMPT_FIXTURE = "the confidential prompt fixture marker 7a2c9e"


def _post(base_url, path, body=None, headers=None, raw_body=None):
    data = raw_body if raw_body is not None else json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=hdrs, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read()), exc.headers
        finally:
            exc.close()


def _get(base_url, path, headers=None):
    req = urllib.request.Request(f"{base_url}{path}", headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read()), exc.headers
        finally:
            exc.close()


def _make_receipt(rid, **overrides):
    base = dict(
        request_id=rid, timestamp="t", gateway_version="v", engine="llama_server",
        engine_version="v", model_alias="a", weight_digest="d" * 64, template_digest="d" * 64,
        config_digest="d" * 64, engine_executable_digest="d" * 64, argv_digest="d" * 64,
        locality_level="L1", route="local", listen_address="x",
        backend_address="y", input_tokens=1, output_tokens=1, latency_ms=1.0, status="completed",
    )
    base.update(overrides)
    return Receipt(**base)


# ---------------------------------------------------------------------------
# Original handoff bullets (unchanged behaviour, updated for new dataclasses)
# ---------------------------------------------------------------------------


class CommandMetacharactersRemainLiteralTest(unittest.TestCase):
    def test_metacharacters_reach_the_backend_as_one_literal_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hostile_name = "model; touch pwned.txt && echo $(id).gguf"
            model_path = tmp_path / "safe-on-disk.gguf"
            model_path.write_bytes(b"weights")

            adapter = LlamaServerAdapter()
            port = free_port()
            dump_path = tmp_path / "argv_dump.json"
            argv = adapter.build_argv(str(FAKE_ENGINE_PATH), hostile_name, "127.0.0.1", port)
            argv += ["--argv-dump", str(dump_path)]

            supervisor = Supervisor(tmp_path / "sessions")
            supervisor.start(
                alias="hostile",
                argv=argv,
                allowlisted_executable=str(FAKE_ENGINE_PATH),
                host="127.0.0.1",
                port=port,
                health_path="/health",
            )
            try:
                received_argv = json.loads(dump_path.read_text())
                self.assertIn(hostile_name, received_argv)
                marker = tmp_path / "pwned.txt"
                self.assertFalse(marker.exists(), "a shell metacharacter must never be interpreted")
            finally:
                supervisor.stop("hostile")


class ExecutableAllowlistRefusalTest(unittest.TestCase):
    def test_non_allowlisted_executable_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = Supervisor(Path(tmp) / "sessions")
            port = free_port()
            with self.assertRaises(ExecutableNotAllowedError):
                supervisor.start(
                    alias="a1",
                    argv=[str(FAKE_ENGINE_PATH), "--model", "x", "--host", "127.0.0.1", "--port", str(port)],
                    allowlisted_executable="/bin/not-the-configured-engine",
                    host="127.0.0.1",
                    port=port,
                    health_path="/health",
                )


class TraversalAndSymlinkEscapeRefusalTest(unittest.TestCase):
    def test_symlink_escaping_declared_root_is_ignored(self):
        from plag_in.inventory import discover_gguf

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outside = tmp_path / "outside"
            outside.mkdir()
            (outside / "secret.gguf").write_bytes(b"outside-weights")

            root = tmp_path / "declared_root"
            root.mkdir()
            (root / "escape.gguf").symlink_to(outside / "secret.gguf")

            self.assertEqual(discover_gguf([root]), [])


class ModelMutationInvalidatesIdentityTest(unittest.TestCase):
    def test_hash_mismatch_after_mutation_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"original")
            registry = Registry()
            registry.bind(
                RegistryEntry(
                    alias="a1", identity=ModelIdentity.from_file(path), engine="llama_server", engine_config_digest="d"
                )
            )
            path.write_bytes(b"mutated-content")
            with self.assertRaises(ModelIdentityMismatchError):
                registry.verify_identity("a1")


class UnknownSecurityConfigurationFieldRefusalTest(unittest.TestCase):
    def test_unknown_security_field_fails_closed(self):
        with self.assertRaises(UnknownConfigurationFieldError):
            load_config({"security": {"auth_mode": "none", "undocumented_backdoor_flag": True}})


class LoopbackDefaultTest(unittest.TestCase):
    def test_default_config_binds_loopback(self):
        config = load_config({})
        self.assertEqual(config.bind.host, "127.0.0.1")

    def test_gateway_config_default_is_loopback(self):
        self.assertEqual(BindConfig().host, "127.0.0.1")


class NonLoopbackAuthenticationRefusalTest(unittest.TestCase):
    def test_config_load_refuses_non_loopback_without_auth(self):
        with self.assertRaises(ConfigurationError):
            load_config({"bind": {"host": "0.0.0.0", "port": 8080}, "security": {"auth_mode": "none"}})

    def test_gateway_server_refuses_non_loopback_without_auth_even_if_constructed_directly(self):
        config = GatewayConfig(bind=BindConfig(host="0.0.0.0", port=free_port()), security=SecurityConfig(auth_mode="none"))
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            context = GatewayContext(config, Registry(), store, {})
            with self.assertRaises(LocalityPolicyError):
                GatewayServer(context)


class SecretAndPromptFixtureStringsAbsentTest(unittest.TestCase):
    def test_secrets_and_prompt_content_absent_from_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_key = ApiKey(id="k1", secret=_SECRET_FIXTURE)
            stack = build_gateway_stack(tmp_path, alias="fixture-alias", auth_mode="api_key", api_keys=(api_key,))
            try:
                status, _body, _headers = _post(
                    stack.server.base_url,
                    "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": _PROMPT_FIXTURE}]},
                    headers={"Authorization": f"Bearer {_SECRET_FIXTURE}"},
                )
                self.assertEqual(status, 200)

                receipts_text = stack.receipts_path.read_text()
                self.assertNotIn(_SECRET_FIXTURE, receipts_text)
                self.assertNotIn(_PROMPT_FIXTURE, receipts_text)
            finally:
                stack.stop()


class ReceiptMutationAndReorderingDetectionTest(unittest.TestCase):
    def test_mutation_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            store.append(_make_receipt("r1"))
            store.append(_make_receipt("r2"))
            lines = store.path.read_text().splitlines()
            store.path.write_text("\n".join(reversed(lines)) + "\n")
            valid, _count = store.verify_chain()
            self.assertFalse(valid)


class DownloaderActionAbsentTest(unittest.TestCase):
    def test_no_download_command_in_cli(self):
        parser = cli_module.build_parser()
        subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
        self.assertNotIn("download", subparsers_action.choices.keys())

    def test_recommend_module_has_no_download_capability(self):
        names = dir(recommend_module)
        self.assertFalse(any("download" in n.lower() for n in names))

    def test_gateway_has_no_download_route(self):
        source = Path(gateway_module.__file__).read_text()
        self.assertNotIn("download", source.lower())


class ToolCallTransportCannotExecuteTest(unittest.TestCase):
    def test_tool_calls_are_forwarded_but_never_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stack = build_gateway_stack(tmp_path, alias="fixture-alias", with_tool_call=True)
            try:
                status, body, _headers = _post(
                    stack.server.base_url,
                    "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "call a tool"}]},
                )
                self.assertEqual(status, 200)
                tool_calls = body["choices"][0]["message"]["tool_calls"]
                self.assertEqual(tool_calls[0]["function"]["name"], "not_executed")
                marker = tmp_path / "tool_was_executed.marker"
                self.assertFalse(marker.exists())

                source = Path(gateway_module.__file__).read_text()
                self.assertNotIn("subprocess", source)
                self.assertNotIn("eval(", source)
                self.assertNotIn("exec(", source)
            finally:
                stack.stop()


class PromptDerivedDigestsAbsentTest(unittest.TestCase):
    def test_no_hash_of_the_prompt_appears_in_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                prompt = "a distinctive prompt used only to compute a digest fixture"
                status, _body, headers = _post(
                    stack.server.base_url,
                    "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": prompt}]},
                )
                self.assertEqual(status, 200)
                request_id = headers.get("X-PLAG-IN-Request-ID")
                status, receipt, _headers = _get(stack.server.base_url, f"/plag-in/v1/receipts/{request_id}")
                self.assertEqual(status, 200)
                prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
                serialized = json.dumps(receipt)
                self.assertNotIn(prompt_digest, serialized)
                self.assertNotIn(canonical_digest({"prompt": prompt}), serialized)
            finally:
                stack.stop()


class UnsupportedClientCapabilityTypedErrorTest(unittest.TestCase):
    def test_unsupported_endpoints_return_typed_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                for path in ("/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/messages"):
                    status, body, _headers = _post(stack.server.base_url, path, {"model": "fixture-alias"})
                    self.assertEqual(status, 501, path)
                    self.assertEqual(body["error"]["type"], "unsupported_capability")
                    self.assertEqual(body["error"]["endpoint"], path)
            finally:
                stack.stop()


# ---------------------------------------------------------------------------
# F1 / regression 1, 2: locality can no longer be reported falsely
# ---------------------------------------------------------------------------


class NonLoopbackBackendRejectedTest(unittest.TestCase):
    def test_register_backend_rejects_non_loopback_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            config = GatewayConfig(bind=BindConfig(host="127.0.0.1", port=0))
            context = GatewayContext(config, Registry(), store, {})
            with self.assertRaises(LocalityPolicyError):
                context.register_backend(
                    "fixture-alias",
                    BackendInfo(
                        engine="llama_server", engine_version="v", host="203.0.113.10", port=443,
                        weight_digest="d" * 64, config_digest="d" * 64,
                        executable_digest="d" * 64, argv_digest="d" * 64,
                    ),
                )
            self.assertEqual(context.locality_level(), "L0")

    def test_cli_serve_rejects_non_loopback_engine_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "models"
            model_root.mkdir()
            model_path = write_fixture_weight(model_root / "m.gguf")
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "model_roots": [str(model_root)],
                "engines": {"llama_server": {"executable": str(FAKE_ENGINE_PATH)}},
                "bind": {"host": "127.0.0.1", "port": 0},
                "security": {"auth_mode": "none"},
            }))
            parser = cli_module.build_parser()
            args = parser.parse_args([
                "serve", "--config", str(config_path), "--alias", "a1",
                "--model-path", str(model_path), "--engine-port", str(free_port()),
                "--engine-host", "203.0.113.10", "--state-dir", str(tmp_path / "state"),
            ])
            with self.assertRaises(LocalityPolicyError):
                cli_module.cmd_serve(args)
            self.assertFalse((tmp_path / "state" / "sessions").exists() and
                              any((tmp_path / "state" / "sessions").iterdir()))


class ConfigOnlyVerificationReportsL0Test(unittest.TestCase):
    def test_verify_without_a_running_session_reports_l0(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"bind": {"host": "127.0.0.1", "port": 0}, "security": {"auth_mode": "none"}}))
            parser = cli_module.build_parser()
            args = parser.parse_args(["verify", "--config", str(config_path)])
            result = cli_module.cmd_verify(args)
            self.assertEqual(result["locality_enforcement_level"], "L0")
            self.assertTrue(result["measured_loopback_bind"])
            self.assertFalse(result["measured_running_loopback_session"])

    def test_verify_reports_l1_only_for_a_confirmed_running_loopback_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = write_fixture_weight(tmp_path / "m.gguf")
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "model_roots": [str(tmp_path)],
                "engines": {"llama_server": {"executable": str(FAKE_ENGINE_PATH)}},
                "bind": {"host": "127.0.0.1", "port": 0},
                "security": {"auth_mode": "none"},
            }))
            supervisor = Supervisor(tmp_path / "state" / "sessions")
            port = free_port()
            supervisor.start(
                alias="a1",
                argv=[str(FAKE_ENGINE_PATH), "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port)],
                allowlisted_executable=str(FAKE_ENGINE_PATH), host="127.0.0.1", port=port, health_path="/health",
            )
            try:
                parser = cli_module.build_parser()
                args = parser.parse_args([
                    "verify", "--config", str(config_path),
                    "--state-dir", str(tmp_path / "state"), "--alias", "a1",
                ])
                result = cli_module.cmd_verify(args)
                self.assertEqual(result["locality_enforcement_level"], "L1")
            finally:
                supervisor.stop("a1")


# ---------------------------------------------------------------------------
# F2 / regression 3: alias cannot escape the session directory
# ---------------------------------------------------------------------------


class AliasEscapeCreatesNoFileOutsideStateDirTest(unittest.TestCase):
    def test_traversal_alias_is_refused_and_creates_no_stray_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions_dir = tmp_path / "state" / "sessions"
            supervisor = Supervisor(sessions_dir)
            port = free_port()
            with self.assertRaises(InvalidAliasError):
                supervisor.start(
                    alias="../escaped",
                    argv=[str(FAKE_ENGINE_PATH), "--model", "x", "--host", "127.0.0.1", "--port", str(port)],
                    allowlisted_executable=str(FAKE_ENGINE_PATH), host="127.0.0.1", port=port, health_path="/health",
                )
            self.assertFalse((tmp_path / "state" / "escaped.json").exists())
            self.assertFalse((tmp_path / "escaped.json").exists())

    def test_various_hostile_aliases_are_all_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = Supervisor(Path(tmp) / "sessions")
            for hostile in ("", ".", "..", "../x", "a/b", "a\\b", "x" * 100):
                with self.assertRaises(InvalidAliasError, msg=hostile):
                    supervisor.load_record(hostile)


# ---------------------------------------------------------------------------
# F7 / regression 4: `serve` model-root containment
# ---------------------------------------------------------------------------


class ServeRejectsPathsOutsidePermittedRootsTest(unittest.TestCase):
    def test_direct_path_outside_model_roots_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "roots"
            model_root.mkdir()
            outside = write_fixture_weight(tmp_path / "outside" / "m.gguf")
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(outside, [model_root], None)

    def test_symlink_inside_root_pointing_outside_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "roots"
            model_root.mkdir()
            outside = write_fixture_weight(tmp_path / "outside" / "m.gguf")
            symlink = model_root / "escape.gguf"
            symlink.symlink_to(outside)
            with self.assertRaises(PathContainmentError):
                resolve_contained_model(symlink, [model_root], None)

    def test_cli_serve_rejects_before_process_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "roots"
            model_root.mkdir()
            outside = write_fixture_weight(tmp_path / "outside" / "m.gguf")
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "model_roots": [str(model_root)],
                "engines": {"llama_server": {"executable": str(FAKE_ENGINE_PATH)}},
                "bind": {"host": "127.0.0.1", "port": 0},
                "security": {"auth_mode": "none"},
            }))
            port = free_port()
            parser = cli_module.build_parser()
            args = parser.parse_args([
                "serve", "--config", str(config_path), "--alias", "a1", "--model-path", str(outside),
                "--engine-port", str(port), "--state-dir", str(tmp_path / "state"),
            ])
            with self.assertRaises(PathContainmentError):
                cli_module.cmd_serve(args)
            # No process was launched: the port is still free.
            import socket
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                self.fail("a process was launched despite the containment violation")
            finally:
                probe.close()


# ---------------------------------------------------------------------------
# F4 / regressions 9, 10, 11, 12: key expiry, origin, endpoint and alias scope
# ---------------------------------------------------------------------------


class ExpiredKeyRejectedTest(unittest.TestCase):
    def test_expired_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            key = ApiKey(id="k1", secret="s1", expiry="2000-01-01T00:00:00+00:00")
            config = GatewayConfig(security=SecurityConfig(auth_mode="api_key", api_keys=(key,)))
            context = GatewayContext(config, Registry(), store, {})
            with self.assertRaises(Exception):
                context.authenticate({"Authorization": "Bearer s1"}, "127.0.0.1")

    def test_invalid_expiry_configuration_fails_closed_at_load(self):
        with self.assertRaises(ConfigurationError):
            load_config({
                "security": {
                    "auth_mode": "api_key",
                    "api_keys": [{"id": "k1", "secret": "s1", "expiry": "not-a-timestamp"}],
                },
            })

    def test_naive_expiry_without_utc_offset_fails_closed(self):
        with self.assertRaises(ConfigurationError):
            load_config({
                "security": {
                    "auth_mode": "api_key",
                    "api_keys": [{"id": "k1", "secret": "s1", "expiry": "2099-01-01T00:00:00"}],
                },
            })


class LoopbackOnlyKeyRejectedForNonLoopbackClientTest(unittest.TestCase):
    def test_loopback_key_rejected_for_remote_client_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            key = ApiKey(id="k1", secret="s1", origin="loopback")
            config = GatewayConfig(security=SecurityConfig(auth_mode="api_key", api_keys=(key,)))
            context = GatewayContext(config, Registry(), store, {})
            with self.assertRaises(Exception):
                context.authenticate({"Authorization": "Bearer s1"}, "203.0.113.10")
            # Loopback clients remain fine.
            context.authenticate({"Authorization": "Bearer s1"}, "127.0.0.1")

    def test_any_origin_key_accepted_for_remote_client_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReceiptStore(Path(tmp) / "receipts.jsonl", hmac_key_path=Path(tmp) / "key")
            key = ApiKey(id="k1", secret="s1", origin="any")
            config = GatewayConfig(security=SecurityConfig(auth_mode="api_key", api_keys=(key,)))
            context = GatewayContext(config, Registry(), store, {})
            context.authenticate({"Authorization": "Bearer s1"}, "203.0.113.10")


class EndpointScopeEnforcedOnEveryProtectedRouteTest(unittest.TestCase):
    def test_key_scoped_to_chat_only_is_rejected_on_other_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = ApiKey(id="k1", secret=_SECRET_FIXTURE, endpoints=("chat_completions",))
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias", auth_mode="api_key", api_keys=(key,))
            try:
                headers = {"Authorization": f"Bearer {_SECRET_FIXTURE}"}
                status, _body, _h = _get(stack.server.base_url, "/v1/models", headers)
                self.assertEqual(status, 401)
                status, _body, _h = _get(stack.server.base_url, "/plag-in/v1/capabilities", headers)
                self.assertEqual(status, 401)
                status, _body, _h = _get(stack.server.base_url, "/plag-in/v1/status", headers)
                self.assertEqual(status, 401)
                status, _body, _h = _post(
                    stack.server.base_url, "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]}, headers,
                )
                self.assertEqual(status, 200)
            finally:
                stack.stop()


class AliasScopedKeyReceivesNoDataForOtherAliasTest(unittest.TestCase):
    def test_key_scoped_to_alias_a_sees_nothing_of_alias_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = Registry()
            store = ReceiptStore(tmp_path / "receipts.jsonl", hmac_key_path=tmp_path / "key")
            config = GatewayConfig(security=SecurityConfig(auth_mode="api_key", api_keys=(ApiKey(id="k", secret="s", aliases=("a",)),)))
            context = GatewayContext(config, registry, store, {"llama_server": LlamaServerAdapter()})

            for alias in ("a", "b"):
                model_path = write_fixture_weight(tmp_path / f"{alias}.gguf", alias.encode())
                identity = ModelIdentity.from_file(model_path)
                registry.bind(RegistryEntry(alias=alias, identity=identity, engine="llama_server", engine_config_digest="d"))
                context.register_backend(alias, BackendInfo(
                    engine="llama_server", engine_version="v", host="127.0.0.1", port=free_port(),
                    weight_digest=identity.sha256, config_digest="d", executable_digest="d" * 64, argv_digest="d" * 64,
                ))

            api_key = config.security.api_keys[0]
            self.assertEqual(context.authorized_aliases(api_key), ["a"])
            caps = context.capabilities_document(api_key)
            self.assertEqual(set(caps["models"]), {"a"})
            status_doc = context.status_document(api_key)
            self.assertEqual(status_doc["served_aliases"], ["a"])

            with self.assertRaises(Exception):
                context.check_scope(api_key, "chat_completions", alias="b")


# ---------------------------------------------------------------------------
# F5 / regressions 13, 14: process identity digests
# ---------------------------------------------------------------------------
# Covered in tests/unit/test_supervisor.py (ArgvDigestStableAcrossInterpretersTest,
# test_executable_mutation_changes_the_session_digest).


# ---------------------------------------------------------------------------
# F6 / regression 15: process ownership and clean stop
# ---------------------------------------------------------------------------
# Covered in tests/unit/test_supervisor.py (test_stop_produces_no_resource_warning_under_forced_gc).


# ---------------------------------------------------------------------------
# F8 / regressions 19, 20: receipt integrity on failure paths
# ---------------------------------------------------------------------------


class RoutedBackendFailureCreatesFailedReceiptTest(unittest.TestCase):
    def test_backend_failure_returns_typed_error_with_ids_and_a_failed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                stack.supervisor.stop(stack.alias)  # backend is now unreachable
                status, body, _headers = _post(
                    stack.server.base_url,
                    "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                )
                self.assertEqual(status, 502)
                self.assertEqual(body["error"]["type"], "backend_unavailable")
                request_id = body["error"]["request_id"]
                receipt_id = body["error"]["receipt_id"]
                self.assertTrue(request_id)
                self.assertEqual(request_id, receipt_id)

                status, receipt, _headers = _get(stack.server.base_url, f"/plag-in/v1/receipts/{request_id}")
                self.assertEqual(status, 200)
                self.assertEqual(receipt["status"], "failed")
                self.assertIsNone(receipt["input_tokens"])
                self.assertIsNone(receipt["output_tokens"])

                # No remote-route code path exists in the gateway at all.
                source = Path(gateway_module.__file__).read_text()
                self.assertNotIn("remote_provider", source)
                self.assertNotIn("openai.com", source)
            finally:
                stack.stop()


class ReceiptWriteFailureCannotReturnSuccessTest(unittest.TestCase):
    def test_receipt_persistence_failure_surfaces_instead_of_a_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                with patch.object(
                    stack.context.receipts, "append", side_effect=ReceiptPersistenceError("disk full (fixture)")
                ):
                    status, body, _headers = _post(
                        stack.server.base_url,
                        "/v1/chat/completions",
                        {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                    )
                self.assertEqual(status, 500)
                self.assertEqual(body["error"]["type"], "receipt_persistence_failed")
            finally:
                stack.stop()


# ---------------------------------------------------------------------------
# F9 / regressions 16, 17, 18: request bounds and typed failures
# ---------------------------------------------------------------------------


class MalformedJsonReturnsTyped400Test(unittest.TestCase):
    def test_malformed_request_json_returns_typed_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                status, body, _headers = _post(
                    stack.server.base_url, "/v1/chat/completions", raw_body=b"{not valid json"
                )
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["type"], "invalid_request")
            finally:
                stack.stop()


class OversizedInputReturnsTyped413Test(unittest.TestCase):
    def test_oversized_body_returns_typed_413_before_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                before_count = stack.receipts_path.read_text().count("\n") if stack.receipts_path.exists() else 0
                oversized = b"x" * (gateway_module.MAX_REQUEST_BODY_BYTES + 1)
                status, body, _headers = _post(
                    stack.server.base_url, "/v1/chat/completions", raw_body=oversized
                )
                self.assertEqual(status, 413)
                self.assertEqual(body["error"]["type"], "payload_too_large")
                after_count = stack.receipts_path.read_text().count("\n") if stack.receipts_path.exists() else 0
                self.assertEqual(before_count, after_count, "an oversized body must never reach the backend")
            finally:
                stack.stop()


class InvalidBackendJsonReturnsTypedBackendErrorTest(unittest.TestCase):
    def test_non_json_backend_response_returns_typed_backend_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                # Point the registered backend at a plain TCP echo-nothing
                # port that is not the fake engine at all, so the "JSON"
                # response is actually empty/invalid rather than well formed.
                import socket
                import threading

                bogus_port = free_port()

                def serve_garbage():
                    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind(("127.0.0.1", bogus_port))
                    srv.listen(1)
                    srv.settimeout(5)
                    try:
                        conn, _ = srv.accept()
                    except OSError:
                        return
                    conn.recv(65536)
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\nnot json!!!")
                    conn.close()
                    srv.close()

                thread = threading.Thread(target=serve_garbage, daemon=True)
                thread.start()

                backend = stack.context.backends["fixture-alias"]
                stack.context.backends["fixture-alias"] = BackendInfo(
                    engine=backend.engine, engine_version=backend.engine_version,
                    host="127.0.0.1", port=bogus_port, weight_digest=backend.weight_digest,
                    config_digest=backend.config_digest, executable_digest=backend.executable_digest,
                    argv_digest=backend.argv_digest, template_digest=backend.template_digest,
                )

                status, body, _headers = _post(
                    stack.server.base_url,
                    "/v1/chat/completions",
                    {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                )
                thread.join(timeout=5)
                self.assertEqual(status, 502)
                self.assertEqual(body["error"]["type"], "backend_response_invalid")
            finally:
                stack.stop()


class ConcurrencyBoundProducesTypedOverloadTest(unittest.TestCase):
    def test_exhausted_inflight_slots_return_typed_429(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                acquired = []
                for _ in range(gateway_module.MAX_CONCURRENT_CHAT_REQUESTS):
                    self.assertTrue(stack.context._inflight.acquire(blocking=False))  # noqa: SLF001
                    acquired.append(True)
                try:
                    status, body, _headers = _post(
                        stack.server.base_url,
                        "/v1/chat/completions",
                        {"model": "fixture-alias", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    self.assertEqual(status, 429)
                    self.assertEqual(body["error"]["type"], "overloaded")
                finally:
                    for _ in acquired:
                        stack.context._inflight.release()  # noqa: SLF001
            finally:
                stack.stop()


class FiniteBodyReadTimeoutTest(unittest.TestCase):
    def test_handler_declares_a_finite_socket_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack = build_gateway_stack(Path(tmp), alias="fixture-alias")
            try:
                handler_cls = stack.server._httpd.RequestHandlerClass  # noqa: SLF001
                self.assertIsNotNone(handler_cls.timeout)
                self.assertGreater(handler_cls.timeout, 0)
            finally:
                stack.stop()


if __name__ == "__main__":
    unittest.main()
