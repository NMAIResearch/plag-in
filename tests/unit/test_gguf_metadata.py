"""Unit tests for the bounded GGUF header, metadata and descriptor reader.

Every fixture here is built byte by byte in a scratch directory. No held
model, no native library and no real weight file is opened.

`build_gguf` is the shared fixture builder for the whole suite: other
modules import it from here rather than writing a second encoder that
could drift from this one.
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from plag_in.errors import ConfigurationError
from plag_in.gguf_metadata import (
    DEFAULT_LIMITS,
    GgufLimits,
    _read,
    read_gguf_metadata,
)

TYPE_UINT32 = 4
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10

CHAT_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value_type: int, encoded: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + encoded


def build_gguf(
    path: Path,
    *,
    architecture: str | None = "testarch",
    context_length: int | None = 4096,
    chat_template: str | None = CHAT_TEMPLATE,
    tensors: int = 2,
    version: int = 3,
    magic: bytes = b"GGUF",
    extra_metadata: bytes = b"",
    extra_metadata_count: int = 0,
    declared_tensor_count: int | None = None,
    declared_metadata_count: int | None = None,
    tensor_offsets: list[int] | None = None,
    payload_bytes: int = 512,
    alignment: int | None = None,
    truncate_to: int | None = None,
) -> Path:
    """Write one minimal GGUF file, or a deliberately malformed one."""
    entries: list[bytes] = []
    if architecture is not None:
        entries.append(_kv("general.architecture", TYPE_STRING, _string(architecture)))
    if context_length is not None and architecture is not None:
        entries.append(
            _kv(f"{architecture}.context_length", TYPE_UINT32, struct.pack("<I", context_length))
        )
    if chat_template is not None:
        entries.append(_kv("tokenizer.chat_template", TYPE_STRING, _string(chat_template)))
    if alignment is not None:
        entries.append(_kv("general.alignment", TYPE_UINT32, struct.pack("<I", alignment)))
    entries.append(_kv("general.file_type", TYPE_UINT32, struct.pack("<I", 15)))
    # A tokenizer-shaped array: walked for its shape, never retained.
    entries.append(
        _kv("tokenizer.ggml.tokens", TYPE_ARRAY, struct.pack("<IQ", TYPE_STRING, 3))
        + _string("a")
        + _string("b")
        + _string("c")
    )
    metadata = b"".join(entries) + extra_metadata
    metadata_count = len(entries) + extra_metadata_count

    offsets = tensor_offsets if tensor_offsets is not None else [i * 16 for i in range(tensors)]
    descriptors = b"".join(
        _string(f"tensor.{index}")
        + struct.pack("<I", 1)
        + struct.pack("<Q", 4)
        + struct.pack("<I", 0)
        + struct.pack("<Q", offsets[index])
        for index in range(tensors)
    )

    header = (
        magic
        + struct.pack("<I", version)
        + struct.pack("<Q", tensors if declared_tensor_count is None else declared_tensor_count)
        + struct.pack(
            "<Q", metadata_count if declared_metadata_count is None else declared_metadata_count
        )
    )
    body = header + metadata + descriptors
    effective_alignment = alignment or 32
    remainder = len(body) % effective_alignment
    if remainder:
        body += b"\x00" * (effective_alignment - remainder)
    # A recognisable payload: a reader that crosses into it can be caught.
    body += b"\xab" * payload_bytes
    if truncate_to is not None:
        body = body[:truncate_to]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


class _ScratchCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def model(self, name: str = "model.gguf", **kwargs) -> Path:
        return build_gguf(self.root / name, **kwargs)


class WellFormedReadTests(_ScratchCase):
    def test_reports_locally_derived_evidence(self):
        result = read_gguf_metadata(self.model()).as_dict()
        self.assertEqual(result["gguf_version"], 3)
        self.assertEqual(result["architecture"], "testarch")
        self.assertEqual(result["context_length"], 4096)
        self.assertTrue(result["chat_template_present"])
        self.assertEqual(result["chat_template_bytes"], len(CHAT_TEMPLATE))
        self.assertEqual(result["file_type_id"], 15)
        self.assertEqual(result["tensor_count"], 2)

    def test_quantisation_label_is_never_inferred_from_the_identifier(self):
        result = read_gguf_metadata(self.model()).as_dict()
        self.assertEqual(result["file_type_label"], "unassessed")
        self.assertIn("file_type_label", result["unassessed_fields"])

    def test_absent_values_stay_unassessed_rather_than_guessed(self):
        result = read_gguf_metadata(
            self.model(architecture=None, context_length=None, chat_template=None)
        ).as_dict()
        self.assertEqual(result["architecture"], "unassessed")
        self.assertEqual(result["context_length"], "unassessed")
        self.assertFalse(result["chat_template_present"])
        self.assertIsNone(result["chat_template_sha256"])
        for field in ("architecture", "context_length", "chat_template"):
            self.assertIn(field, result["unassessed_fields"])

    def test_declared_alignment_is_honoured(self):
        result = read_gguf_metadata(self.model(alignment=64)).as_dict()
        self.assertEqual(result["alignment"], 64)
        self.assertEqual(result["tensor_data_offset"] % 64, 0)


class NoTensorPayloadReadTests(_ScratchCase):
    """Probe 7: metadata inspection does not read tensor payload bytes."""

    def test_no_byte_at_or_beyond_the_tensor_data_offset_is_read(self):
        result = read_gguf_metadata(self.model(payload_bytes=4096)).as_dict()
        self.assertFalse(result["tensor_payload_read"])
        self.assertLessEqual(result["max_read_offset"], result["tensor_data_offset"])

    def test_observed_read_range_never_enters_the_payload(self):
        """Watch every read the reader performs, not only its own report."""
        path = self.model(payload_bytes=4096)
        data_offset = read_gguf_metadata(path).tensor_data_offset
        observed: list[int] = []

        class _WatchedHandle:
            def __init__(self, handle):
                self._handle = handle

            def read(self, count):
                start = self._handle.tell()
                chunk = self._handle.read(count)
                observed.append(start + len(chunk))
                return chunk

            def seek(self, offset):
                return self._handle.seek(offset)

        with open(path, "rb") as handle:
            _read(_WatchedHandle(handle), path, path.stat().st_size, DEFAULT_LIMITS)
        self.assertTrue(observed)
        self.assertLessEqual(max(observed), data_offset)


class RefusalTests(_ScratchCase):
    """Probe 6: malformed and oversized structures refuse within fixed bounds."""

    def _reason(self, **kwargs) -> str:
        path = self.model(**kwargs)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        return caught.exception.fields["reason"]

    def test_non_gguf_file_is_refused(self):
        self.assertEqual(self._reason(magic=b"NOPE"), "not_gguf")

    def test_unsupported_version_is_refused(self):
        self.assertEqual(self._reason(version=9), "unsupported_gguf_version")

    def test_truncated_file_is_refused(self):
        self.assertEqual(self._reason(truncate_to=24), "truncated")

    def test_excessive_metadata_count_is_refused(self):
        path = self.model(declared_metadata_count=10_000_000)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        self.assertEqual(caught.exception.fields["reason"], "limit_exceeded")
        self.assertEqual(caught.exception.fields["limit"], "max_metadata_count")

    def test_excessive_tensor_count_is_refused(self):
        path = self.model(declared_tensor_count=10_000_000)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        self.assertEqual(caught.exception.fields["limit"], "max_tensor_count")

    def test_excessive_string_length_is_refused(self):
        oversized = _string("general.name") + struct.pack("<I", TYPE_STRING) + struct.pack(
            "<Q", 1 << 40
        )
        path = self.model(extra_metadata=oversized, extra_metadata_count=1)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        self.assertEqual(caught.exception.fields["reason"], "limit_exceeded")
        self.assertEqual(caught.exception.fields["limit"], "max_string_bytes")

    def test_excessive_array_length_is_refused(self):
        oversized = _string("big.array") + struct.pack("<I", TYPE_ARRAY) + struct.pack(
            "<IQ", TYPE_UINT32, 1 << 40
        )
        path = self.model(extra_metadata=oversized, extra_metadata_count=1)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        self.assertEqual(caught.exception.fields["limit"], "max_array_length")

    def test_nested_array_is_refused(self):
        nested = _string("nested") + struct.pack("<I", TYPE_ARRAY) + struct.pack(
            "<IQ", TYPE_ARRAY, 1
        )
        path = self.model(extra_metadata=nested, extra_metadata_count=1)
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path)
        self.assertEqual(caught.exception.fields["limit"], "max_array_nesting")

    def test_unknown_value_type_is_refused(self):
        unknown = _string("odd") + struct.pack("<I", 99)
        self.assertEqual(
            self._reason(extra_metadata=unknown, extra_metadata_count=1), "unknown_value_type"
        )

    def test_unknown_array_element_type_is_refused(self):
        unknown = _string("odd.array") + struct.pack("<I", TYPE_ARRAY) + struct.pack(
            "<IQ", 99, 1
        )
        self.assertEqual(
            self._reason(extra_metadata=unknown, extra_metadata_count=1), "unknown_value_type"
        )

    def test_integer_beyond_the_signed_range_is_refused(self):
        overflowing = _string("huge") + struct.pack("<I", TYPE_STRING) + struct.pack(
            "<Q", (1 << 63) + 1
        )
        self.assertEqual(
            self._reason(extra_metadata=overflowing, extra_metadata_count=1), "integer_overflow"
        )

    def test_impossible_tensor_offset_is_refused(self):
        self.assertEqual(
            self._reason(tensors=1, tensor_offsets=[1 << 40]), "impossible_offset"
        )

    def test_unsupported_alignment_is_refused(self):
        self.assertEqual(self._reason(alignment=3), "impossible_offset")

    def test_absent_file_is_refused(self):
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(self.root / "absent.gguf")
        self.assertEqual(caught.exception.fields["reason"], "unreadable")


class FixedBoundTests(_ScratchCase):
    """The bounds are the reader's own, not the file's."""

    def test_metadata_byte_budget_is_enforced(self):
        path = self.model()
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path, limits=GgufLimits(max_total_metadata_bytes=8))
        self.assertEqual(caught.exception.fields["limit"], "max_total_metadata_bytes")

    def test_key_length_limit_is_enforced(self):
        path = self.model()
        with self.assertRaises(ConfigurationError) as caught:
            read_gguf_metadata(path, limits=GgufLimits(max_key_bytes=4))
        self.assertEqual(caught.exception.fields["limit"], "max_key_bytes")

    def test_default_limits_are_declared_explicitly(self):
        for field in (
            "max_metadata_count",
            "max_tensor_count",
            "max_key_bytes",
            "max_string_bytes",
            "max_array_length",
            "max_array_nesting",
            "max_tensor_dimensions",
            "max_tensor_descriptor_bytes",
            "max_total_metadata_bytes",
        ):
            self.assertIsInstance(getattr(DEFAULT_LIMITS, field), int)
            self.assertGreater(getattr(DEFAULT_LIMITS, field), 0)


if __name__ == "__main__":
    unittest.main()
