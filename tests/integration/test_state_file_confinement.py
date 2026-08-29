"""Confinement and ownership for the state files outside the receipt set.

Two routes opened files by name: `StateLayout.load_or_create_backend_api_key`
and the session-record and alias-lock routes in `supervisor.py`. Each probe here
is one member of the classes the receipt store was repaired against over nine
passes, applied to those two routes: a symbolic link at a name, a non-regular
file at a name, an unbounded read, a mode set by name, a predictable temporary
name, and cleanup that removes a file the failing operation did not create.

Every probe runs against a scratch directory. Nothing here touches the live
state directory, starts a model or opens a socket. Probes that would block on a
FIFO run in a subprocess under a timeout, so a blocking regression fails the
test rather than hanging the suite.
"""
import contextlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from plag_in import cli
from plag_in import supervisor as supervisor_module
from plag_in.cli import run_private_chat, run_profile_gateway
from plag_in.errors import ConfigurationError, PlagInError
from plag_in.paths import StateLayout
from plag_in.setup_flow import default_state_path
from plag_in.supervisor import SessionRecord, Supervisor
from tests.support import FIXTURE_COMPATIBILITY_RECORD, registered_fixture_compatibility

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

STATE_ENVIRONMENT_VARIABLES = ("HOME", "XDG_STATE_HOME")


@contextlib.contextmanager
def scratch_environment(values: dict):
    """Apply environment changes and restore the previous state exactly.

    A value of `None` removes the variable for the duration. Both the set and
    the removal are undone on exit, and a variable that was absent before is
    absent afterwards, so a probe cannot change the conditions a later test runs
    under (independent review of the C1-R2 repair, R2-F2).
    """
    names = set(values) | set(STATE_ENVIRONMENT_VARIABLES)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

VALID_KEY = "a" * 64


def run_bounded(body: str, timeout_s: float = 10.0) -> subprocess.CompletedProcess:
    """Run a probe body in a subprocess, so a blocking open fails rather than hangs."""
    script = textwrap.dedent(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


class ScratchStateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "state"
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def state(self) -> StateLayout:
        return StateLayout(self.base).ensure()


class BackendKeyConfinementTests(ScratchStateTestCase):
    """The key route: every create, mode change and read in it."""

    def test_a_symlinked_key_is_refused_and_the_target_is_untouched(self):
        state = self.state()
        target = self.outside / "someone-elses-file"
        target.write_text("not a key\n", encoding="utf-8")
        os.chmod(target, 0o644)
        link = state.backend_api_key_file("model-a")
        link.symlink_to(target)

        with self.assertRaises(ConfigurationError):
            state.load_or_create_backend_api_key("model-a")

        self.assertEqual(target.read_text(encoding="utf-8"), "not a key\n")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644, "no mode reached the target")
        self.assertTrue(link.is_symlink(), "the link itself is left in place, not removed")

    def test_a_symlinked_key_directory_is_refused(self):
        elsewhere = self.outside / "engine_keys"
        elsewhere.mkdir()
        self.base.mkdir(parents=True)
        (self.base / "engine_keys").symlink_to(elsewhere)
        state = StateLayout(self.base)

        with self.assertRaises(ConfigurationError):
            state.load_or_create_backend_api_key("model-a")

        self.assertEqual(list(elsewhere.iterdir()), [], "no key was created through the link")

    def test_a_fifo_at_the_key_name_is_refused_without_blocking(self):
        state = self.state()
        fifo = state.backend_api_key_file("model-a")
        os.mkfifo(fifo)

        result = run_bounded(
            f"""
            from plag_in.paths import StateLayout
            from plag_in.errors import ConfigurationError
            from pathlib import Path
            state = StateLayout(Path({str(self.base)!r}))
            try:
                state.load_or_create_backend_api_key("model-a")
                print("ACCEPTED")
            except ConfigurationError:
                print("REFUSED")
            """
        )
        self.assertEqual(result.stdout.strip(), "REFUSED", result.stderr)
        self.assertTrue(stat.S_ISFIFO(os.lstat(fifo).st_mode), "the FIFO is left in place")

    def test_a_directory_at_the_key_name_is_refused(self):
        state = self.state()
        state.backend_api_key_file("model-a").mkdir()
        with self.assertRaises(ConfigurationError):
            state.load_or_create_backend_api_key("model-a")

    def test_a_pre_existing_key_is_returned_and_left_byte_identical(self):
        state = self.state()
        path = state.backend_api_key_file("model-a")
        path.write_text(VALID_KEY + "\n", encoding="utf-8")
        before = path.read_bytes()

        returned_path, secret = state.load_or_create_backend_api_key("model-a")

        self.assertEqual(returned_path, path)
        self.assertEqual(secret, VALID_KEY)
        self.assertEqual(path.read_bytes(), before, "a pre-existing key is never rewritten")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, "the mode is tightened")

    def test_an_oversized_key_file_is_refused_by_the_bounded_read(self):
        state = self.state()
        path = state.backend_api_key_file("model-a")
        path.write_bytes(b"x" * 5000)
        with self.assertRaises(ConfigurationError) as caught:
            state.load_or_create_backend_api_key("model-a")
        self.assertIn("maximum", caught.exception.message)

    def test_a_key_file_at_the_bound_is_still_read(self):
        state = self.state()
        path = state.backend_api_key_file("model-a")
        padding = b"\n" * (4096 - (len(VALID_KEY) + 1))
        path.write_bytes(VALID_KEY.encode("utf-8") + b"\n" + padding)
        self.assertEqual(len(path.read_bytes()), 4096)

        _, secret = state.load_or_create_backend_api_key("model-a")
        self.assertEqual(secret, VALID_KEY)

    def test_a_failure_while_writing_a_new_key_removes_the_created_file(self):
        """The poisoned-key case: an interrupted first write disabled the alias."""
        state = self.state()
        path = state.backend_api_key_file("model-a")
        real_fsync = os.fsync

        def fail_once(fd):
            raise OSError("simulated failure after the key file was created")

        with patch.object(os, "fsync", fail_once):
            with self.assertRaises(OSError):
                state.load_or_create_backend_api_key("model-a")

        self.assertFalse(path.exists(), "a key this call created must not survive its failure")
        self.assertEqual(os.fsync, real_fsync)

        # And the alias is still usable afterwards, which the leftover empty
        # file prevented permanently.
        _, secret = state.load_or_create_backend_api_key("model-a")
        self.assertEqual(len(secret), 64)

    def test_a_read_failure_is_a_named_refusal_and_keeps_the_key(self):
        """The injection reaches `os.read` only because the read is confined.

        On the reverted tree the key is read by `Path.read_text`, whose buffered
        reader does not call `os.read`, so this injection does not fire there at
        all. What the probe establishes is therefore about this tree: a failure
        inside the bounded reader surfaces as a named refusal and leaves a key
        this call did not create byte-identical. The preservation half alone
        holds on both trees, because the reverted route has no removal on this
        branch to begin with.
        """
        state = self.state()
        path = state.backend_api_key_file("model-a")
        path.write_text(VALID_KEY + "\n", encoding="utf-8")
        before = path.read_bytes()

        def fail_read(fd, *args, **kwargs):
            raise OSError("simulated read failure")

        with patch.object(os, "read", fail_read):
            with self.assertRaises(ConfigurationError):
                state.load_or_create_backend_api_key("model-a")

        self.assertEqual(path.read_bytes(), before, "a pre-existing key is never removed")

    def test_the_key_and_its_directory_carry_private_modes(self):
        state = self.state()
        path, _ = state.load_or_create_backend_api_key("model-a")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(state.engine_keys_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(state.sessions_dir.stat().st_mode), 0o700)

    def test_a_loose_key_directory_is_tightened_through_the_descriptor(self):
        self.base.mkdir(parents=True)
        loose = self.base / "engine_keys"
        loose.mkdir(mode=0o755)
        state = StateLayout(self.base)

        state.load_or_create_backend_api_key("model-a")

        self.assertEqual(stat.S_IMODE(loose.stat().st_mode), 0o700)


class AncestorLinkTests(ScratchStateTestCase):
    """C1: a symbolic link anywhere above the declared directory.

    The final component was opened with `O_NOFOLLOW`, but the ancestors were
    resolved by `makedirs`, so the whole state set was created on the far side
    of a link while every check on the final name passed. These probes plant the
    link at the parent and at a grandparent, and assert that nothing at all
    appears at the link's target.
    """

    def declared_through_link(self, link_name: str, depth: int) -> Path:
        """Return a declared path whose ancestor at `depth` levels up is a link."""
        target = self.outside / "target"
        target.mkdir()
        link = self.root / link_name
        link.symlink_to(target)
        return link.joinpath(*[f"level{n}" for n in range(depth)])

    def assert_nothing_escaped(self):
        target = self.outside / "target"
        self.assertEqual(
            sorted(p.name for p in target.iterdir()),
            [],
            "nothing may be created through a linked ancestor",
        )

    def test_the_key_route_refuses_a_linked_parent(self):
        declared = self.declared_through_link("declared-key-parent", 1)
        with self.assertRaises(ConfigurationError) as caught:
            StateLayout(declared).load_or_create_backend_api_key("model-a")
        self.assertEqual(caught.exception.fields.get("component"), "declared-key-parent")
        self.assert_nothing_escaped()

    def test_the_key_route_refuses_a_linked_grandparent(self):
        declared = self.declared_through_link("declared-key-grandparent", 2)
        with self.assertRaises(ConfigurationError):
            StateLayout(declared).load_or_create_backend_api_key("model-a")
        self.assert_nothing_escaped()

    def test_the_key_route_refuses_a_linked_ancestor_on_ensure(self):
        declared = self.declared_through_link("declared-key-ensure", 1)
        with self.assertRaises(ConfigurationError):
            StateLayout(declared).ensure()
        self.assert_nothing_escaped()

    def test_the_supervisor_refuses_a_linked_parent(self):
        declared = self.declared_through_link("declared-sessions-parent", 1)
        with self.assertRaises(ConfigurationError) as caught:
            Supervisor(declared / "sessions")
        self.assertEqual(caught.exception.fields.get("component"), "declared-sessions-parent")
        self.assert_nothing_escaped()

    def test_the_supervisor_refuses_a_linked_grandparent(self):
        declared = self.declared_through_link("declared-sessions-grandparent", 2)
        with self.assertRaises(ConfigurationError):
            Supervisor(declared / "sessions")
        self.assert_nothing_escaped()

    def test_a_link_planted_at_an_ancestor_that_already_exists_is_still_refused(self):
        """The ancestor is not trusted merely because it was there first."""
        (self.outside / "target").mkdir()
        real = self.root / "declared-existing"
        real.mkdir()
        (real / "state").mkdir()
        state = StateLayout(real / "state").ensure()
        state.load_or_create_backend_api_key("model-a")

        real.rename(self.root / "declared-moved")
        (self.root / "declared-existing").symlink_to(self.outside / "target")

        with self.assertRaises(ConfigurationError):
            StateLayout(real / "state").load_or_create_backend_api_key("model-b")
        self.assert_nothing_escaped()

    def test_a_link_free_declared_path_is_still_accepted(self):
        """The guard against the walk refusing an ordinary nested path."""
        declared = self.root / "a" / "b" / "c" / "state"
        state = StateLayout(declared).ensure()
        path, secret = state.load_or_create_backend_api_key("model-a")

        self.assertTrue(path.is_file())
        self.assertEqual(len(secret), 64)
        self.assertEqual(stat.S_IMODE(declared.stat().st_mode), 0o700)
        # The ancestors this call had to create are the operator's, not the
        # product's, so they keep the process umask rather than 0o700.
        self.assertTrue((self.root / "a").is_dir())

    def test_a_non_directory_ancestor_is_refused(self):
        blocker = self.root / "not-a-directory"
        blocker.write_text("a regular file where a directory was declared\n", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            StateLayout(blocker / "state").ensure()


class ServingEntryPointPathTests(ScratchStateTestCase):
    """C1-R1: the declared state path must reach the walk unresolved.

    `run_private_chat` and `run_profile_gateway` resolved the state path before
    anything confined it, so a link in the declared path was followed and its
    target became the effective state path. The walk then had nothing left to
    refuse. These probes drive both entry points with a linked ancestor and stop
    at the point the real session would build its state, before any registry,
    listener or model work.
    """

    def chat_config(self, root: Path) -> Path:
        model = root / "models" / "fixture.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
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
                            "backend_libraries": [{"path": str(library), "sha256": "d" * 64}],
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

    SAFETY = (
        {
            "memory.high": 10 * 1024**3,
            "memory.max": 12 * 1024**3,
            "memory.swap.max": 2 * 1024**3,
        },
        {
            "available_ram_bytes": 32 * 1024**3,
            "required_available_ram_bytes": 20 * 1024**3,
            "gpu": None,
            "required_free_vram_mib": None,
        },
    )

    def drive_default(self, entry_point, environment: dict):
        """Run one entry point with no declared path, so the default route runs.

        `state_dir=None` is what makes this the default seam: the entry point
        calls `default_state_path()` itself. The same stand-in stops at the
        session boundary, so nothing after `cli.py:151` runs.
        """
        seen = {}

        def stop_at_session_start(config, alias, profile, state_dir, **_derived):
            seen["state_dir"] = Path(state_dir)
            StateLayout(Path(state_dir)).ensure()
            raise AssertionError("the walk must refuse before this point is passed")

        with (
            registered_fixture_compatibility(),
            scratch_environment(environment),
            patch("plag_in.cli._start_embedded_session", stop_at_session_start),
            patch("plag_in.cli._model_load_safety", return_value=self.SAFETY),
        ):
            with self.assertRaises(ConfigurationError) as caught:
                entry_point(
                    self.chat_config(self.root / "fixture"),
                    lambda _: "y",
                    io.StringIO(),
                    state_dir=None,
                )
        return seen, caught.exception

    def home_with_linked(self, component: str) -> dict:
        """A scratch HOME in which one component of the default path is a link.

        The mapping is returned rather than applied. Removing `XDG_STATE_HOME`
        here, before the restoration context was entered, left it removed for
        every later test in the process: `patch.dict` restores what changed
        inside it and cannot restore a value deleted before it (independent
        review of the C1-R2 repair, R2-F2). `None` asks the context to remove
        the variable and to put it back exactly as it was.
        """
        home = self.root / "home"
        target = self.outside / "target"
        target.mkdir(parents=True, exist_ok=True)
        if component == ".local":
            home.mkdir(parents=True)
            (home / ".local").symlink_to(target)
        elif component == "state":
            (home / ".local").mkdir(parents=True)
            (home / ".local" / "state").symlink_to(target)
        else:
            raise AssertionError(f"unhandled component {component}")
        return {"HOME": str(home), "XDG_STATE_HOME": None}

    def drive(self, entry_point, declared: Path):
        """Run one entry point to the point of session start, and no further.

        The stand-in for `_start_embedded_session` records the path it was
        handed and performs the one state operation the real function performs
        first, `StateLayout(state_dir).ensure()` at `cli.py:151`. Nothing after
        that line runs, so no registry is bound, no listener is opened and no
        model is loaded.
        """
        seen = {}

        def stop_at_session_start(config, alias, profile, state_dir, **_derived):
            seen["state_dir"] = Path(state_dir)
            StateLayout(Path(state_dir)).ensure()
            raise AssertionError("the walk must refuse before this point is passed")

        with (
            registered_fixture_compatibility(),
            patch("plag_in.cli._start_embedded_session", stop_at_session_start),
            patch("plag_in.cli._model_load_safety", return_value=self.SAFETY),
        ):
            with self.assertRaises(ConfigurationError) as caught:
                entry_point(
                    self.chat_config(self.root / "fixture"),
                    lambda _: "y",
                    io.StringIO(),
                    state_dir=declared,
                )
        return seen, caught.exception

    def linked_declared_path(self, name: str) -> Path:
        target = self.outside / "target"
        target.mkdir()
        link = self.root / name
        link.symlink_to(target)
        return link / "state"

    def assert_nothing_escaped(self):
        self.assertEqual(
            sorted(p.name for p in (self.outside / "target").iterdir()),
            [],
            "no state may be created through the linked ancestor",
        )

    def test_private_chat_refuses_a_linked_ancestor(self):
        declared = self.linked_declared_path("declared-chat")
        seen, error = self.drive(run_private_chat, declared)

        self.assertEqual(
            seen["state_dir"],
            declared,
            "the declared path must arrive unresolved, not as the link's target",
        )
        self.assertEqual(error.fields.get("component"), "declared-chat")
        self.assert_nothing_escaped()

    def test_profile_gateway_refuses_a_linked_ancestor(self):
        declared = self.linked_declared_path("declared-gateway")
        seen, error = self.drive(run_profile_gateway, declared)

        self.assertEqual(seen["state_dir"], declared)
        self.assertEqual(error.fields.get("component"), "declared-gateway")
        self.assert_nothing_escaped()

    def test_neither_entry_point_resolves_a_state_path(self):
        """The structural guard against the call being reintroduced."""
        source = (Path(cli.__file__)).read_text(encoding="utf-8")
        for name in ("run_private_chat", "run_profile_gateway"):
            start = source.index(f"def {name}(")
            end = source.index("\ndef ", start + 1)
            # Comment lines are stripped: the rule is about what the function
            # executes, and the comment recording why it does not resolve names
            # the call it is forbidding.
            body = "\n".join(
                line
                for line in source[start:end].splitlines()
                if not line.lstrip().startswith("#")
            )
            self.assertIn("effective_state", body)
            for call in (".resolve()", "realpath"):
                self.assertNotIn(
                    call,
                    body,
                    f"{name} must not resolve a path before it is confined",
                )

    def test_the_default_route_refuses_a_linked_local_component(self):
        environment = self.home_with_linked(".local")
        seen, error = self.drive_default(run_private_chat, environment)

        self.assertEqual(error.fields.get("component"), ".local")
        self.assertIn(".local", seen["state_dir"].parts, "the spelling must survive selection")
        self.assert_nothing_escaped()

    def test_the_default_route_refuses_a_linked_state_component(self):
        environment = self.home_with_linked("state")
        seen, error = self.drive_default(run_profile_gateway, environment)

        self.assertEqual(error.fields.get("component"), "state")
        self.assert_nothing_escaped()

    def test_the_default_route_refuses_a_linked_xdg_state_home(self):
        target = self.outside / "target"
        target.mkdir(parents=True, exist_ok=True)
        link = self.root / "xdg-link"
        link.symlink_to(target)
        seen, error = self.drive_default(run_private_chat, {"XDG_STATE_HOME": str(link)})

        self.assertEqual(error.fields.get("component"), "xdg-link")
        self.assertEqual(
            seen["state_dir"],
            link / "plag-in",
            "XDG_STATE_HOME is used exactly as supplied",
        )
        self.assert_nothing_escaped()

    def test_the_default_resolves_the_home_anchor_and_nothing_below_it(self):
        """Provenance, not appearance.

        A link-free result is not evidence that no link was followed, because
        following one produces a link-free result too. This asserts where each
        part of the default came from: the anchor is translated, and the three
        components below it are the literal spellings.
        """
        home = self.root / "home-anchor"
        real_home = self.outside / "real-home"
        real_home.mkdir(parents=True)
        home.symlink_to(real_home)
        (real_home / ".local" / "state").mkdir(parents=True)
        linked_local = self.outside / "elsewhere"
        linked_local.mkdir()

        with scratch_environment({"HOME": str(home), "XDG_STATE_HOME": None}):
            default = default_state_path()

        self.assertEqual(
            default,
            Path(os.path.realpath(str(home))) / ".local" / "state" / "plag-in",
            "only the home anchor is translated",
        )
        self.assertEqual(default.parts[-3:], (".local", "state", "plag-in"))

    def test_a_non_empty_xdg_state_home_is_never_canonicalised(self):
        declared = self.root / "xdg-plain"
        declared.mkdir()
        with scratch_environment({"XDG_STATE_HOME": str(declared)}):
            self.assertEqual(default_state_path(), declared / "plag-in")

    def test_an_empty_xdg_state_home_selects_the_home_fallback(self):
        """The rule the previous report stated without this qualification.

        An empty value is treated as unset. The alternative, taking the empty
        string as supplied, selects a bare relative `plag-in` beneath the
        working directory, which is not a state location the product should
        choose for itself.
        """
        home = self.root / "empty-xdg-home"
        (home / ".local" / "state").mkdir(parents=True)

        with scratch_environment({"HOME": str(home), "XDG_STATE_HOME": ""}):
            with_empty = default_state_path()
        with scratch_environment({"HOME": str(home), "XDG_STATE_HOME": None}):
            with_absent = default_state_path()

        self.assertEqual(with_empty, with_absent)
        self.assertEqual(with_empty.parts[-3:], (".local", "state", "plag-in"))
        self.assertNotEqual(with_empty, Path("plag-in"))

    def test_a_home_probe_leaves_the_environment_byte_identical(self):
        """R2-F2: the regression that begins with both variables set.

        The previous probes removed `XDG_STATE_HOME` before entering the
        restoration context, so it stayed removed for the rest of the process.
        This starts with both variables set to sentinels, runs each HOME-based
        probe body, and requires both to be unchanged afterwards.
        """
        sentinels = {
            "HOME": str(self.root / "sentinel-home"),
            "XDG_STATE_HOME": str(self.root / "sentinel-xdg"),
        }
        with scratch_environment(sentinels):
            for component, entry_point in (
                (".local", run_private_chat),
                ("state", run_profile_gateway),
            ):
                with self.subTest(component=component):
                    try:
                        self.drive_default(entry_point, self.home_with_linked(component))
                        for name, expected in sentinels.items():
                            self.assertEqual(
                                os.environ.get(name),
                                expected,
                                f"{name} must be byte-identical after the probe",
                            )
                    finally:
                        # Unconditional, so a failed subtest leaves the next one
                        # a clean tree and reports its own result rather than a
                        # consequential error.
                        self.reset_scratch_tree()

    def reset_scratch_tree(self):
        """Clear the scratch home and outside target between probe bodies."""
        for path in (self.root / "home", self.outside / "target"):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        (self.outside / "target").mkdir(parents=True, exist_ok=True)

    def test_a_declared_path_is_never_canonicalised(self):
        declared = self.linked_declared_path("declared-not-canonicalised")
        with self.assertRaises(ConfigurationError):
            StateLayout(declared).ensure()
        self.assert_nothing_escaped()


class SessionRecordConfinementTests(ScratchStateTestCase):
    """The supervisor's record, temporary file and alias lock."""

    def sessions(self) -> Path:
        return self.base / "sessions"

    def record(self, alias="model-a") -> SessionRecord:
        return SessionRecord(
            alias=alias,
            pid=1,
            executable="/bin/true",
            executable_digest="0" * 64,
            argv_digest="1" * 64,
            start_time_token="123",
            host="127.0.0.1",
            port=1,
            started_at="0",
        )

    def test_a_symlinked_alias_lock_is_refused_and_creates_nothing_outside(self):
        supervisor = Supervisor(self.sessions())
        target = self.outside / "planted-lock"
        (self.sessions() / "model-a.alias_lock").symlink_to(target)

        with self.assertRaises(ConfigurationError):
            supervisor.status("model-a")

        self.assertFalse(target.exists(), "no file was created at the link's target")

    def test_a_fifo_at_the_alias_lock_is_refused_without_blocking(self):
        supervisor = Supervisor(self.sessions())
        os.mkfifo(self.sessions() / "model-a.alias_lock")

        result = run_bounded(
            f"""
            from plag_in.supervisor import Supervisor
            from plag_in.errors import ConfigurationError
            from pathlib import Path
            supervisor = Supervisor(Path({str(self.sessions())!r}))
            try:
                supervisor.status("model-a")
                print("ACCEPTED")
            except ConfigurationError:
                print("REFUSED")
            """
        )
        self.assertEqual(result.stdout.strip(), "REFUSED", result.stderr)

    def test_a_symlinked_session_record_is_refused(self):
        supervisor = Supervisor(self.sessions())
        target = self.outside / "planted-record"
        target.write_text(json.dumps({"alias": "model-a"}), encoding="utf-8")
        (self.sessions() / "model-a.json").symlink_to(target)

        with self.assertRaises(ConfigurationError):
            supervisor.load_record("model-a")

    def test_a_fifo_at_the_session_record_is_refused_without_blocking(self):
        supervisor = Supervisor(self.sessions())
        os.mkfifo(self.sessions() / "model-a.json")

        result = run_bounded(
            f"""
            from plag_in.supervisor import Supervisor
            from plag_in.errors import ConfigurationError
            from pathlib import Path
            supervisor = Supervisor(Path({str(self.sessions())!r}))
            try:
                supervisor.load_record("model-a")
                print("ACCEPTED")
            except ConfigurationError:
                print("REFUSED")
            """
        )
        self.assertEqual(result.stdout.strip(), "REFUSED", result.stderr)

    def test_an_absent_record_is_an_absence_not_a_refusal(self):
        supervisor = Supervisor(self.sessions())
        self.assertIsNone(supervisor.load_record("model-a"))

    def test_an_oversized_session_record_is_refused(self):
        supervisor = Supervisor(self.sessions())
        (self.sessions() / "model-a.json").write_bytes(b"x" * 70000)
        with self.assertRaises(ConfigurationError):
            supervisor.load_record("model-a")

    def test_a_malformed_record_raises_a_typed_refusal(self):
        supervisor = Supervisor(self.sessions())
        (self.sessions() / "model-a.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            supervisor.load_record("model-a")

    def test_a_record_missing_its_fields_raises_a_typed_refusal(self):
        supervisor = Supervisor(self.sessions())
        (self.sessions() / "model-a.json").write_text('{"alias": "model-a"}', encoding="utf-8")
        with self.assertRaises(PlagInError):
            supervisor.load_record("model-a")

    def test_a_planted_temporary_file_is_preserved(self):
        supervisor = Supervisor(self.sessions())
        planted = self.sessions() / "model-a.json.tmp"
        sentinel = b"a file the product did not create\n"
        planted.write_bytes(sentinel)

        supervisor._write_record(self.record())

        self.assertTrue(planted.exists(), "the old predictable name must not be reused")
        self.assertEqual(planted.read_bytes(), sentinel)

    def test_the_temporary_name_is_fresh_for_each_write(self):
        supervisor = Supervisor(self.sessions())
        seen = set()
        real_open_state_file = supervisor_module.open_state_file

        def record_names(path, flags, mode, role, created_registry=None, **kwargs):
            if "temporary" in role:
                seen.add(Path(path).name)
            return real_open_state_file(
                path, flags, mode, role, created_registry=created_registry, **kwargs
            )

        with patch.object(supervisor_module, "open_state_file", record_names):
            supervisor._write_record(self.record())
            supervisor._write_record(self.record())

        self.assertEqual(len(seen), 2, f"each write needs its own temporary name: {seen}")
        for name in seen:
            self.assertNotEqual(name, "model-a.json.tmp")

    def test_no_temporary_file_survives_a_successful_write(self):
        supervisor = Supervisor(self.sessions())
        supervisor._write_record(self.record())
        leftovers = [p.name for p in self.sessions().iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_a_failure_before_the_replacement_keeps_the_established_record(self):
        supervisor = Supervisor(self.sessions())
        supervisor._write_record(self.record())
        path = self.sessions() / "model-a.json"
        before = path.read_bytes()

        def failing_replace(src, dst):
            raise OSError("simulated rename failure")

        with patch.object(os, "replace", failing_replace):
            with self.assertRaises(OSError):
                supervisor._write_record(self.record(alias="model-a"))

        self.assertEqual(path.read_bytes(), before, "the established record is untouched")
        leftovers = [p.name for p in self.sessions().iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [], "the temporary file is removed")

    def test_a_failure_after_a_committed_replacement_keeps_the_new_record(self):
        """R8-F1 applied here: a committed replacement is the coherent state."""
        supervisor = Supervisor(self.sessions())
        supervisor._write_record(self.record())
        path = self.sessions() / "model-a.json"
        before = path.read_bytes()

        real_replace = os.replace
        renamed = {"done": False}

        def replace_then_fail(src, dst):
            real_replace(src, dst)
            renamed["done"] = True
            raise OSError("simulated failure in the interval after the rename")

        updated = self.record()
        updated.pid = 4242
        with patch.object(os, "replace", replace_then_fail):
            with self.assertRaises(OSError):
                supervisor._write_record(updated)

        self.assertTrue(renamed["done"], "the rename must have completed")
        self.assertTrue(path.exists(), "the installed record must survive")
        self.assertNotEqual(path.read_bytes(), before, "and must be the new record")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["pid"], 4242)
        leftovers = [p.name for p in self.sessions().iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_the_record_carries_a_private_mode(self):
        supervisor = Supervisor(self.sessions())
        supervisor._write_record(self.record())
        path = self.sessions() / "model-a.json"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_a_symlinked_sessions_directory_is_refused(self):
        elsewhere = self.outside / "sessions"
        elsewhere.mkdir()
        self.base.mkdir(parents=True)
        (self.base / "sessions").symlink_to(elsewhere)

        with self.assertRaises(ConfigurationError):
            Supervisor(self.base / "sessions")

        self.assertEqual(list(elsewhere.iterdir()), [], "nothing was created through the link")

    def test_clearing_a_record_removes_the_link_not_its_target(self):
        supervisor = Supervisor(self.sessions())
        target = self.outside / "not-ours"
        target.write_text("keep me\n", encoding="utf-8")
        link = self.sessions() / "model-a.json"
        link.symlink_to(target)

        supervisor._clear_record("model-a")

        self.assertFalse(link.is_symlink(), "the name is cleared")
        self.assertTrue(target.exists(), "what the link pointed at is untouched")
        self.assertEqual(target.read_text(encoding="utf-8"), "keep me\n")

    def test_a_containment_escape_is_still_refused(self):
        supervisor = Supervisor(self.sessions())
        for alias in ("../escape", "sub/dir", "/absolute"):
            with self.assertRaises(PlagInError):
                supervisor.load_record(alias)


if __name__ == "__main__":
    unittest.main()
