"""Regressions for the two counterexamples that rejected Phase 1.

F1: an arbitrary configuration asserted `compatibility_status: tested` for a
scratch GGUF, and the serve path derived `tested` from it, with no reviewed
model digest and no decision record anywhere in the chain.

F2: an Ollama manifest declaring `sha256:abc`, with a one-byte
`blobs/sha256-abc` beside it, was reported `local_complete`. A prefix test
cannot bind a complete SHA-256 identity.

Both probes here run against scratch temporary directories. No held model,
native library, GPU or listener is used.
"""
import argparse
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from plag_in.cli import _serve_compatibility_state
from plag_in.config import load_config
from plag_in.errors import ConfigurationError
from plag_in.identity import ModelIdentity
from plag_in.inventory import (
    AVAILABILITY_LOCAL_COMPLETE,
    AVAILABILITY_LOCAL_INCOMPLETE,
    discover_held_models,
    discover_ollama_model_summaries,
    discover_ollama_models,
)
from plag_in.model_paths import resolve_contained_model
from tests.support import (
    FIXTURE_COMPATIBILITY_RECORD,
    fixture_engine_config,
    registered_fixture_compatibility,
)

_MALFORMED_DIGESTS = {
    "short": "sha256:abc",
    "long": "sha256:" + "a" * 65,
    "non_hex": "sha256:" + "g" * 64,
    "upper_case": "sha256:" + "A" * 64,
}


def _base_config(engines: dict, profiles: dict) -> dict:
    return {
        "engines": engines,
        "profiles": profiles,
        "security": {
            "auth_mode": "api_key",
            "api_keys": [
                {
                    "id": "operator",
                    "secret": "s" * 64,
                    "aliases": list(profiles),
                    "origin": "loopback",
                }
            ],
            "content_logging": False,
        },
    }


def _profile(record: str | None = None, status: str = "tested") -> dict:
    profile = {
        "model_path": "/tmp/models/scratch.gguf",
        "engine": "libllama",
        "display_name": "Scratch",
        "compatibility_status": status,
    }
    if record is not None:
        profile["compatibility_record"] = record
    return profile


class TestedRequiresAReviewedRecordTests(unittest.TestCase):
    """F1 at configuration parsing: the status string is not the authority."""

    def setUp(self):
        self.enterContext(registered_fixture_compatibility())
        self.engine = fixture_engine_config(Path("/tmp/lib/libllama.so"))

    def test_the_reproduced_counterexample_is_refused(self):
        """The exact shape the review reproduced: `tested` and nothing else."""
        with self.assertRaises(ConfigurationError) as caught:
            load_config(_base_config({"libllama": self.engine}, {"scratch": _profile()}))
        self.assertIn("compatibility_record", caught.exception.message)

    def test_an_unregistered_record_is_refused(self):
        with self.assertRaises(ConfigurationError) as caught:
            load_config(
                _base_config(
                    {"libllama": self.engine},
                    {"scratch": _profile(record="invented-by-the-operator")},
                )
            )
        self.assertIn("no reviewed compatibility record", caught.exception.message)

    def test_an_engine_that_is_not_the_recorded_native_bundle_is_refused(self):
        engine = dict(self.engine, library_sha256="9" * 64)
        with self.assertRaises(ConfigurationError) as caught:
            load_config(
                _base_config(
                    {"libllama": engine},
                    {"scratch": _profile(record=FIXTURE_COMPATIBILITY_RECORD.record_id)},
                )
            )
        self.assertIn("native bundle", caught.exception.message)
        self.assertEqual(
            caught.exception.fields.get("observed_native_abi_profile"), None
        )

    def test_an_executable_engine_cannot_carry_a_record(self):
        """An external process declares no native component digests to match."""
        with self.assertRaises(ConfigurationError) as caught:
            load_config(
                _base_config(
                    {"libllama": {"executable": "/bin/false"}},
                    {"scratch": _profile(record=FIXTURE_COMPATIBILITY_RECORD.record_id)},
                )
            )
        self.assertIn("no native component digests", caught.exception.message)

    def test_an_unverified_profile_may_not_carry_a_record(self):
        with self.assertRaises(ConfigurationError):
            load_config(
                _base_config(
                    {"libllama": self.engine},
                    {
                        "scratch": _profile(
                            record=FIXTURE_COMPATIBILITY_RECORD.record_id,
                            status="unverified",
                        )
                    },
                )
            )

    def test_the_recorded_pair_is_accepted(self):
        config = load_config(
            _base_config(
                {"libllama": self.engine},
                {"scratch": _profile(record=FIXTURE_COMPATIBILITY_RECORD.record_id)},
            )
        )
        profile = config.profiles["scratch"]
        self.assertEqual(profile.compatibility_status, "tested")
        self.assertEqual(
            profile.compatibility_record, FIXTURE_COMPATIBILITY_RECORD.record_id
        )


class ServeDerivedStatusTests(unittest.TestCase):
    """F1 at the serve path: the bytes present must be the recorded bytes."""

    def setUp(self):
        self.enterContext(registered_fixture_compatibility())
        self.engine = fixture_engine_config(Path("/tmp/lib/libllama.so"))
        self.config = load_config(
            _base_config(
                {"libllama": self.engine},
                {"scratch": _profile(record=FIXTURE_COMPATIBILITY_RECORD.record_id)},
            )
        )

    def _identity(self, sha256: str) -> ModelIdentity:
        return ModelIdentity(
            path="/tmp/models/scratch.gguf", sha256=sha256, size_bytes=1
        )

    def test_other_bytes_at_the_recorded_path_are_not_tested(self):
        status, disclosure = _serve_compatibility_state(
            self.config, self._identity("1" * 64), "libllama"
        )
        self.assertEqual(status, "unverified")
        self.assertEqual(
            disclosure["reason"], "model_digest_does_not_match_reviewed_record"
        )
        self.assertEqual(disclosure["observed_model_sha256"], "1" * 64)

    def test_the_recorded_bytes_are_tested(self):
        status, disclosure = _serve_compatibility_state(
            self.config,
            self._identity(FIXTURE_COMPATIBILITY_RECORD.model_sha256),
            "libllama",
        )
        self.assertEqual(status, "tested")
        self.assertEqual(
            disclosure["compatibility_record"], FIXTURE_COMPATIBILITY_RECORD.record_id
        )
        self.assertEqual(
            disclosure["record_binding_digest"],
            FIXTURE_COMPATIBILITY_RECORD.binding_digest(),
        )

    def test_a_path_with_no_profile_is_unverified(self):
        status, disclosure = _serve_compatibility_state(
            self.config,
            ModelIdentity(path="/tmp/models/other.gguf", sha256="2" * 64, size_bytes=1),
            "libllama",
        )
        self.assertEqual(status, "unverified")
        self.assertEqual(
            disclosure["reason"], "no_configured_profile_for_this_model_path_and_engine"
        )

    def test_another_engine_serving_the_same_path_is_unverified(self):
        status, _ = _serve_compatibility_state(
            self.config,
            self._identity(FIXTURE_COMPATIBILITY_RECORD.model_sha256),
            "some-other-engine",
        )
        self.assertEqual(status, "unverified")

    def test_a_withdrawn_record_falls_back_to_unverified(self):
        """Removing a record from the reviewed source demotes the profile."""
        config = self.config
        with patch(
            "plag_in.compatibility_records.REGISTERED_COMPATIBILITY_RECORDS", ()
        ):
            status, disclosure = _serve_compatibility_state(
                config,
                self._identity(FIXTURE_COMPATIBILITY_RECORD.model_sha256),
                "libllama",
            )
        self.assertEqual(status, "unverified")
        self.assertEqual(
            disclosure["reason"], "reviewed_compatibility_record_absent"
        )


class MalformedManifestDigestTests(unittest.TestCase):
    """F2: a digest that cannot bind an identity is never local completeness."""

    def _store(self, root: Path, digest: str, blob_name: str) -> Path:
        manifests = root / "manifests" / "registry.example" / "library" / "fixture"
        manifests.mkdir(parents=True)
        blobs = root / "blobs"
        blobs.mkdir()
        (blobs / blob_name).write_bytes(b"x")
        (manifests / "latest").write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": digest,
                            "size": 1,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_every_malformed_form_stays_local_incomplete(self):
        for name, digest in _MALFORMED_DIGESTS.items():
            with self.subTest(form=name), tempfile.TemporaryDirectory() as tmp:
                root = self._store(
                    Path(tmp), digest, f"sha256-{digest.split(':', 1)[1]}"
                )
                held = discover_held_models(root, [])
                self.assertEqual(len(held), 1)
                self.assertEqual(held[0].availability, AVAILABILITY_LOCAL_INCOMPLETE)
                self.assertEqual(held[0].reason, "manifest_digest_malformed")
                self.assertEqual(held[0].blob_path, "")

    def test_the_reproduced_counterexample_is_not_local_complete(self):
        """`sha256:abc` with a matching one-byte `blobs/sha256-abc` beside it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._store(Path(tmp), "sha256:abc", "sha256-abc")
            held = discover_held_models(root, [])
            self.assertNotEqual(held[0].availability, AVAILABILITY_LOCAL_COMPLETE)
            self.assertEqual(held[0].reason, "manifest_digest_malformed")

    def test_the_quick_summary_reports_no_held_entry(self):
        for name, digest in _MALFORMED_DIGESTS.items():
            with self.subTest(form=name), tempfile.TemporaryDirectory() as tmp:
                root = self._store(
                    Path(tmp), digest, f"sha256-{digest.split(':', 1)[1]}"
                )
                self.assertEqual(discover_ollama_model_summaries(root), [])

    def test_manifest_resolution_reports_no_entry(self):
        for name, digest in _MALFORMED_DIGESTS.items():
            with self.subTest(form=name), tempfile.TemporaryDirectory() as tmp:
                root = self._store(
                    Path(tmp), digest, f"sha256-{digest.split(':', 1)[1]}"
                )
                self.assertEqual(discover_ollama_models(root), [])

    def test_a_malformed_blob_is_not_a_servable_model_path(self):
        from plag_in.errors import PathContainmentError

        for name, digest in _MALFORMED_DIGESTS.items():
            with self.subTest(form=name), tempfile.TemporaryDirectory() as tmp:
                blob_name = f"sha256-{digest.split(':', 1)[1]}"
                root = self._store(Path(tmp), digest, blob_name)
                with self.assertRaises(PathContainmentError):
                    resolve_contained_model(root / "blobs" / blob_name, [], root)

    def test_a_well_formed_held_manifest_is_still_local_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            digest = hashlib.sha256(b"x").hexdigest()
            root = self._store(Path(tmp), f"sha256:{digest}", f"sha256-{digest}")
            held = discover_held_models(root, [])
            self.assertEqual(held[0].availability, AVAILABILITY_LOCAL_COMPLETE)
            self.assertEqual(held[0].reason, "")


class _FakeEmbedded:
    """A stand-in for the native backend. No native call is made."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self) -> None:
        pass

    def ready(self) -> bool:
        return True

    @property
    def runtime_profile(self) -> dict:
        return {"engine": "libllama_embedded", "ollama_runtime_dependency": False}

    @property
    def template_digest(self) -> str:
        return "d" * 64

    def close(self) -> None:
        pass


class _MismatchedBytesFixture(unittest.TestCase):
    """A valid reviewed record, and model bytes that are not its bytes.

    The configuration is otherwise valid: the profile is `tested`, the
    record is registered, and the engine is the recorded native bundle. Only
    the bytes differ, so every difference these suites observe is caused by
    the bytes and by nothing else.
    """

    SAFETY = (
        {"memory.high": 10 * 1024**3, "memory.max": 12 * 1024**3, "memory.swap.max": 0},
        {
            "available_ram_bytes": 32 * 1024**3,
            "required_available_ram_bytes": 1,
            "gpu": None,
            "required_free_vram_mib": None,
        },
    )

    def setUp(self):
        self.enterContext(registered_fixture_compatibility())
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        # Not the recorded bytes: the fixture record names "e" * 64.
        self.model = self.models / "mismatched.gguf"
        self.model.write_bytes(b"bytes-the-reviewed-record-does-not-admit")
        self.library = self.root / "libllama.so"
        self.library.write_bytes(b"fixture-library")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(
            json.dumps(
                _base_config(
                    {"libllama": fixture_engine_config(self.library)},
                    {
                        "mismatched": {
                            "model_path": str(self.model),
                            "engine": "libllama",
                            "display_name": "Mismatched",
                            "compatibility_status": "tested",
                            "compatibility_record": (
                                FIXTURE_COMPATIBILITY_RECORD.record_id
                            ),
                        }
                    },
                )
                | {"model_roots": [str(self.models)]}
            ),
            encoding="utf-8",
        )
        self.config = load_config(json.loads(self.config_path.read_text()))

    @property
    def observed(self) -> str:
        return hashlib.sha256(self.model.read_bytes()).hexdigest()

    def _embedded_session(self, config=None, state_name: str = "state"):
        """Register a backend with the native library replaced by a double."""
        from plag_in.cli import _start_embedded_session

        config = config or self.config
        with patch("plag_in.cli.EmbeddedLibLlama", _FakeEmbedded):
            session = _start_embedded_session(
                config,
                "mismatched",
                config.profiles["mismatched"],
                self.root / state_name,
            )
        self.addCleanup(session.close)
        return session

    def _one_receipt(self, session, request_id: str) -> dict:
        session.context._append_receipt(
            request_id=request_id,
            alias="mismatched",
            backend=session.context.backends["mismatched"],
            start=time.monotonic(),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            status="completed",
        )
        return session.receipts.get(request_id)


class MismatchedBytesUnderAValidRecordTests(_MismatchedBytesFixture):
    """F1-R1: every route that resolves a model reports the derived state."""

    def _assert_mismatch(self, status: str, reason: str) -> None:
        self.assertEqual(status, "unverified")
        self.assertEqual(reason, "model_digest_does_not_match_reviewed_record")

    def test_1_inventory_selection_reports_the_derived_state(self):
        from plag_in.cli import _inventory_rows

        rows = _inventory_rows(self.config, None)
        row = next(row for row in rows if row.alias == "mismatched")
        self._assert_mismatch(row.compatibility_status, row.reason)
        self.assertIn("model_digest_does_not_match_reviewed_record", row.label)
        self.assertNotIn("compatibility: tested", row.label)

    def _drive_route(self, entry_point) -> str:
        output = io.StringIO()
        with (
            patch("plag_in.cli._model_load_safety", return_value=self.SAFETY),
            patch("plag_in.cli._prepare_receipt_state", return_value=False),
        ):
            entry_point(
                self.config_path,
                lambda _: "n",
                output,
                state_dir=self.root / "state",
            )
        return output.getvalue()

    def test_2_private_chat_confirmation_reports_the_derived_state(self):
        from plag_in.cli import run_private_chat

        text = self._drive_route(run_private_chat)
        self.assertIn(
            "Compatibility: unverified (model_digest_does_not_match_reviewed_record)",
            text,
        )
        self.assertNotIn("Compatibility: tested", text)

    def test_3_harness_gateway_confirmation_reports_the_derived_state(self):
        from plag_in.cli import run_profile_gateway

        text = self._drive_route(run_profile_gateway)
        self.assertIn(
            "Compatibility: unverified (model_digest_does_not_match_reviewed_record)",
            text,
        )
        self.assertNotIn("Compatibility: tested", text)

    def test_4_embedded_registration_and_its_consequences(self):
        from plag_in.cli import _start_embedded_session

        with patch("plag_in.cli.EmbeddedLibLlama", _FakeEmbedded):
            session = _start_embedded_session(
                self.config,
                "mismatched",
                self.config.profiles["mismatched"],
                self.root / "state",
            )
        self.addCleanup(session.close)
        backend = session.context.backends["mismatched"]
        self.assertEqual(backend.compatibility_status, "unverified")

        key = self.config.security.api_keys[0]
        status = session.context.status_document(key)
        self.assertEqual(status["model_compatibility"]["mismatched"], "unverified")
        self.assertEqual(
            status["service_class"]["mismatched"], "unverified_local_trial"
        )
        self.assertEqual(
            status["runtime_profiles"]["mismatched"]["compatibility_status"],
            "unverified",
        )

    def test_5_configured_profile_inspection_reports_both(self):
        from plag_in.cli import cmd_inspect

        report = cmd_inspect(
            argparse.Namespace(config=str(self.config_path), all_models=False)
        )
        entry = next(
            item for item in report["profiles"] if item["alias"] == "mismatched"
        )
        self._assert_mismatch(
            entry["compatibility_status"], entry["compatibility_reason"]
        )
        self.assertEqual(entry["configured_compatibility_status"], "tested")
        self.assertEqual(entry["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())

    def test_the_same_configuration_with_the_recorded_bytes_is_tested(self):
        """The control: only the bytes differ between this and the probes above."""
        from plag_in.cli import _inventory_rows

        with registered_fixture_compatibility(
            model_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest()
        ):
            rows = _inventory_rows(self.config, None)
        row = next(row for row in rows if row.alias == "mismatched")
        self.assertEqual(row.compatibility_status, "tested")
        self.assertIn("compatibility: tested", row.label)


class RegistrationSinkTests(_MismatchedBytesFixture):
    """F1-R2: the sink derives its own state and takes none from a caller."""

    def test_a_conflicting_tested_state_cannot_be_supplied(self):
        from plag_in.cli import _start_embedded_session

        with patch("plag_in.cli.EmbeddedLibLlama", _FakeEmbedded):
            with self.assertRaises(TypeError):
                _start_embedded_session(
                    self.config,
                    "mismatched",
                    self.config.profiles["mismatched"],
                    self.root / "state",
                    compatibility="tested",
                )

    def test_the_sink_declares_no_compatibility_parameter(self):
        """The structural guard against the argument being reintroduced.

        The behavioural probe above refuses one spelling. This refuses any
        parameter through which a caller could hand the sink a state it did
        not derive.
        """
        import inspect

        from plag_in.cli import _start_embedded_session

        parameters = set(inspect.signature(_start_embedded_session).parameters)
        self.assertEqual(
            parameters & {"compatibility", "compatibility_status", "compatibility_evidence"},
            set(),
            "the registration sink must derive compatibility, never accept it",
        )

    def test_the_sink_derives_unverified_on_every_evidence_surface(self):
        session = self._embedded_session()
        backend = session.context.backends["mismatched"]
        key = self.config.security.api_keys[0]

        self.assertEqual(backend.compatibility_status, "unverified")
        self.assertNotEqual(backend.compatibility_status, "tested")

        status = session.context.status_document(key)
        self.assertEqual(status["model_compatibility"]["mismatched"], "unverified")
        self.assertEqual(status["service_class"]["mismatched"], "unverified_local_trial")

        capabilities = session.context.capabilities_document(key)
        self.assertEqual(capabilities["model_compatibility"]["mismatched"], "unverified")

        receipt = self._one_receipt(session, "fixture-request-0001")
        self.assertEqual(
            receipt["plag_in_adapter_record"]["compatibility_status"], "unverified"
        )


class DisclosureReachesEveryEvidenceSurfaceTests(_MismatchedBytesFixture):
    """F1-R3: the reason and both digests, not only the status string."""

    def _assert_evidence(self, evidence: dict, *, recorded: str, observed: str) -> None:
        self.assertEqual(evidence["status"], "unverified")
        self.assertEqual(
            evidence["reason"], "model_digest_does_not_match_reviewed_record"
        )
        self.assertEqual(
            evidence["compatibility_record"], FIXTURE_COMPATIBILITY_RECORD.record_id
        )
        self.assertEqual(evidence["recorded_model_sha256"], recorded)
        self.assertEqual(evidence["observed_model_sha256"], observed)

    def test_private_chat_output_states_both_digests(self):
        from plag_in.cli import run_private_chat

        output = io.StringIO()
        with (
            patch("plag_in.cli._model_load_safety", return_value=self.SAFETY),
            patch("plag_in.cli._prepare_receipt_state", return_value=False),
        ):
            run_private_chat(
                self.config_path,
                lambda _: "n",
                output,
                state_dir=self.root / "state",
            )
        text = output.getvalue()
        self.assertIn("model_digest_does_not_match_reviewed_record", text)
        self.assertIn(
            f"Recorded model SHA-256: {FIXTURE_COMPATIBILITY_RECORD.model_sha256}", text
        )
        self.assertIn(f"Observed model SHA-256: {self.observed}", text)
        self.assertIn(
            f"Compatibility record: {FIXTURE_COMPATIBILITY_RECORD.record_id}", text
        )

    def test_inspection_states_both_digests(self):
        from plag_in.cli import cmd_inspect

        report = cmd_inspect(
            argparse.Namespace(config=str(self.config_path), all_models=False)
        )
        entry = next(
            item for item in report["profiles"] if item["alias"] == "mismatched"
        )
        self.assertEqual(
            entry["recorded_model_sha256"], FIXTURE_COMPATIBILITY_RECORD.model_sha256
        )
        self._assert_evidence(
            entry["compatibility_evidence"],
            recorded=FIXTURE_COMPATIBILITY_RECORD.model_sha256,
            observed=self.observed,
        )

    def test_status_capabilities_and_receipt_state_both_digests(self):
        session = self._embedded_session()
        key = self.config.security.api_keys[0]
        expected = {
            "recorded": FIXTURE_COMPATIBILITY_RECORD.model_sha256,
            "observed": self.observed,
        }

        status = session.context.status_document(key)
        self._assert_evidence(
            status["model_compatibility_evidence"]["mismatched"], **expected
        )

        capabilities = session.context.capabilities_document(key)
        self._assert_evidence(
            capabilities["model_compatibility_evidence"]["mismatched"], **expected
        )

        receipt = self._one_receipt(session, "fixture-request-0002")
        self._assert_evidence(
            receipt["plag_in_adapter_record"]["compatibility_evidence"], **expected
        )

    def test_the_matching_control_carries_the_same_digest_twice(self):
        """The control: a tested state still states what it was compared against."""
        with registered_fixture_compatibility(model_sha256=self.observed):
            config = load_config(json.loads(self.config_path.read_text()))
            session = self._embedded_session(config, state_name="state-control")
            key = config.security.api_keys[0]
            evidence = session.context.status_document(key)[
                "model_compatibility_evidence"
            ]["mismatched"]
        self.assertEqual(evidence["status"], "tested")
        self.assertEqual(evidence["reason"], "")
        self.assertEqual(evidence["recorded_model_sha256"], self.observed)
        self.assertEqual(evidence["observed_model_sha256"], self.observed)
        self.assertEqual(
            evidence["trial_report_sha256"],
            FIXTURE_COMPATIBILITY_RECORD.trial_report_sha256,
        )


if __name__ == "__main__":
    unittest.main()
