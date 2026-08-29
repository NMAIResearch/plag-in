"""Bounded GGUF header, metadata and tensor-descriptor reader.

This reader identifies a local model file. It is not a loader and not a
parser for tensor content: it reads the header, the metadata key-value
block and the tensor descriptors, and it stops at the aligned tensor-data
offset. `max_read_offset` in the result is the furthest byte the reader
actually read, so a caller can check that no tensor payload byte was
touched.

Every count and length is validated against an explicit limit and against
the file size before any seek or allocation. Truncation, an impossible
offset, an unknown value type, an out-of-range integer and a limit breach
each fail closed with a stable `reason` on a `ConfigurationError`, and no
partial result is returned.

Array values are never retained. An array records only its element type
and length, so a tokenizer vocabulary does not enter memory as data. Only
an explicit allowlist of scalar keys and the chat template are kept.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from plag_in.errors import ConfigurationError

UNASSESSED = "unassessed"

GGUF_MAGIC = b"GGUF"
SUPPORTED_GGUF_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
MAX_ALIGNMENT = 65536

# A uint64 at or above this bound cannot be represented as a signed 64-bit
# integer, which is what a conforming reader uses for an offset or a count.
_INT64_BOUND = 1 << 63

_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<B",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}
_SCALAR_WIDTHS = {type_id: struct.calcsize(fmt) for type_id, fmt in _SCALAR_FORMATS.items()}

# Scalar keys kept from the metadata block. Everything else is walked to
# advance the offset and discarded, so metadata size does not become
# resident memory.
_RETAINED_KEYS = frozenset(
    {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "general.alignment",
    }
)
_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_CONTEXT_LENGTH_SUFFIX = ".context_length"


@dataclass(frozen=True)
class GgufLimits:
    """Fixed resource bounds for one inspection."""

    max_metadata_count: int = 4096
    max_tensor_count: int = 65536
    max_key_bytes: int = 512
    max_tensor_name_bytes: int = 512
    max_string_bytes: int = 1_048_576
    max_array_length: int = 4_194_304
    max_array_nesting: int = 1
    max_tensor_dimensions: int = 8
    max_tensor_descriptor_bytes: int = 8 * 1024 * 1024
    max_total_metadata_bytes: int = 64 * 1024 * 1024


DEFAULT_LIMITS = GgufLimits()


@dataclass(frozen=True)
class GgufMetadata:
    """Locally derived identification evidence for one GGUF file."""

    path: str
    file_size_bytes: int
    gguf_version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    architecture: str
    model_name: str
    context_length: int | str
    chat_template_present: bool
    chat_template_sha256: str | None
    chat_template_bytes: int | None
    file_type_id: int | str
    quantization_version: int | str
    tensor_type_counts: dict[str, int]
    tensor_data_offset: int
    max_read_offset: int
    unassessed_fields: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "gguf_version": self.gguf_version,
            "tensor_count": self.tensor_count,
            "metadata_count": self.metadata_count,
            "alignment": self.alignment,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "context_length": self.context_length,
            "chat_template_present": self.chat_template_present,
            "chat_template_sha256": self.chat_template_sha256,
            "chat_template_bytes": self.chat_template_bytes,
            # The identifier as read. A human-readable quantisation label is
            # not derived from it, because no locally held table binds the
            # identifier to a label for the build that wrote this file.
            "file_type_id": self.file_type_id,
            "file_type_label": UNASSESSED,
            "quantization_version": self.quantization_version,
            "tensor_type_counts": dict(self.tensor_type_counts),
            "tensor_data_offset": self.tensor_data_offset,
            "max_read_offset": self.max_read_offset,
            "tensor_payload_read": self.max_read_offset > self.tensor_data_offset,
            "unassessed_fields": list(self.unassessed_fields),
        }


def _refuse(reason: str, message: str, **fields) -> ConfigurationError:
    return ConfigurationError(message, reason=reason, **fields)


class _BoundedReader:
    """Sequential reader with validated arithmetic and fixed byte bounds."""

    def __init__(self, handle, file_size: int, limits: GgufLimits):
        self._handle = handle
        self._file_size = file_size
        self._limits = limits
        self.offset = 0
        self.max_read_offset = 0

    def _end_of(self, count: int) -> int:
        if count < 0 or count >= _INT64_BOUND:
            raise _refuse(
                "integer_overflow",
                "GGUF declared a length outside the signed 64-bit range",
                declared_length=count,
            )
        end = self.offset + count
        if end > self._file_size:
            raise _refuse(
                "truncated",
                "GGUF declared a structure that extends past the end of the file",
                required_offset=end,
                file_size_bytes=self._file_size,
            )
        if end > self._limits.max_total_metadata_bytes:
            raise _refuse(
                "limit_exceeded",
                "GGUF metadata exceeds the inspection byte budget",
                limit="max_total_metadata_bytes",
                limit_value=self._limits.max_total_metadata_bytes,
            )
        return end

    def read(self, count: int) -> bytes:
        end = self._end_of(count)
        data = self._handle.read(count)
        if len(data) != count:
            raise _refuse(
                "truncated",
                "GGUF file ended before a declared structure was complete",
                required_offset=end,
            )
        self.offset = end
        if end > self.max_read_offset:
            self.max_read_offset = end
        return data

    def skip(self, count: int) -> None:
        """Advance past bytes without reading them into memory."""
        end = self._end_of(count)
        self._handle.seek(end)
        self.offset = end

    def scalar(self, type_id: int):
        raw = self.read(_SCALAR_WIDTHS[type_id])
        value = struct.unpack(_SCALAR_FORMATS[type_id], raw)[0]
        if type_id == _TYPE_BOOL:
            return value != 0
        return value

    def length(self) -> int:
        value = struct.unpack("<Q", self.read(8))[0]
        if value >= _INT64_BOUND:
            raise _refuse(
                "integer_overflow",
                "GGUF declared a count outside the signed 64-bit range",
                declared_length=value,
            )
        return value

    def text(self, limit: int, limit_name: str) -> bytes:
        size = self.length()
        if size > limit:
            raise _refuse(
                "limit_exceeded",
                "GGUF declared a string longer than the inspection limit",
                limit=limit_name,
                limit_value=limit,
                declared_length=size,
            )
        return self.read(size)

    def skip_text(self, limit: int, limit_name: str) -> None:
        size = self.length()
        if size > limit:
            raise _refuse(
                "limit_exceeded",
                "GGUF declared a string longer than the inspection limit",
                limit=limit_name,
                limit_value=limit,
                declared_length=size,
            )
        self.skip(size)


def _read_value(reader: _BoundedReader, limits: GgufLimits, *, retain: bool, depth: int = 0):
    type_id = reader.scalar(_TYPE_UINT32)
    if type_id in _SCALAR_FORMATS:
        return reader.scalar(type_id)
    if type_id == _TYPE_STRING:
        if retain:
            return reader.text(limits.max_string_bytes, "max_string_bytes")
        reader.skip_text(limits.max_string_bytes, "max_string_bytes")
        return None
    if type_id == _TYPE_ARRAY:
        return _read_array(reader, limits, depth=depth)
    raise _refuse(
        "unknown_value_type",
        "GGUF metadata declared an unknown value type",
        value_type=type_id,
    )


def _read_array(reader: _BoundedReader, limits: GgufLimits, *, depth: int) -> dict:
    """Walk one array, retaining its shape and none of its elements."""
    if depth >= limits.max_array_nesting:
        raise _refuse(
            "limit_exceeded",
            "GGUF metadata nests arrays more deeply than the inspection limit",
            limit="max_array_nesting",
            limit_value=limits.max_array_nesting,
        )
    element_type = reader.scalar(_TYPE_UINT32)
    count = reader.length()
    if count > limits.max_array_length:
        raise _refuse(
            "limit_exceeded",
            "GGUF metadata declared an array longer than the inspection limit",
            limit="max_array_length",
            limit_value=limits.max_array_length,
            declared_length=count,
        )
    if element_type in _SCALAR_FORMATS:
        # Validated as one product before the seek, so a large count and a
        # large width cannot combine into an offset the reader then trusts.
        reader.skip(count * _SCALAR_WIDTHS[element_type])
    elif element_type == _TYPE_STRING:
        for _ in range(count):
            reader.skip_text(limits.max_string_bytes, "max_string_bytes")
    elif element_type == _TYPE_ARRAY:
        raise _refuse(
            "limit_exceeded",
            "GGUF metadata nests arrays more deeply than the inspection limit",
            limit="max_array_nesting",
            limit_value=limits.max_array_nesting,
        )
    else:
        raise _refuse(
            "unknown_value_type",
            "GGUF metadata declared an unknown array element type",
            value_type=element_type,
        )
    return {"kind": "array", "element_type": element_type, "length": count}


def _decode_text(raw: bytes) -> str | None:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _resolve_alignment(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_ALIGNMENT
    if value < 1 or value > MAX_ALIGNMENT or value & (value - 1):
        raise _refuse(
            "impossible_offset",
            "GGUF declared an alignment that is not a supported power of two",
            declared_alignment=value,
        )
    return value


def read_gguf_metadata(
    path: Path | str,
    *,
    limits: GgufLimits = DEFAULT_LIMITS,
) -> GgufMetadata:
    """Identify one GGUF file from its header, metadata and descriptors."""
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        raise _refuse(
            "unreadable", "the selected model file could not be inspected", path=str(file_path)
        ) from exc

    try:
        with open(file_path, "rb") as handle:
            return _read(handle, file_path, file_size, limits)
    except ConfigurationError:
        raise
    except OSError as exc:
        raise _refuse(
            "unreadable", "the selected model file could not be inspected", path=str(file_path)
        ) from exc


def _read(handle, file_path: Path, file_size: int, limits: GgufLimits) -> GgufMetadata:
    reader = _BoundedReader(handle, file_size, limits)

    if reader.read(4) != GGUF_MAGIC:
        raise _refuse(
            "not_gguf", "the selected file does not start with the GGUF magic", path=str(file_path)
        )
    gguf_version = reader.scalar(_TYPE_UINT32)
    if gguf_version not in SUPPORTED_GGUF_VERSIONS:
        raise _refuse(
            "unsupported_gguf_version",
            "the selected file declares a GGUF version this reader does not parse",
            gguf_version=gguf_version,
            supported=list(SUPPORTED_GGUF_VERSIONS),
        )

    tensor_count = reader.length()
    if tensor_count > limits.max_tensor_count:
        raise _refuse(
            "limit_exceeded",
            "GGUF declared more tensors than the inspection limit",
            limit="max_tensor_count",
            limit_value=limits.max_tensor_count,
            declared_length=tensor_count,
        )
    metadata_count = reader.length()
    if metadata_count > limits.max_metadata_count:
        raise _refuse(
            "limit_exceeded",
            "GGUF declared more metadata entries than the inspection limit",
            limit="max_metadata_count",
            limit_value=limits.max_metadata_count,
            declared_length=metadata_count,
        )

    retained: dict[str, object] = {}
    context_lengths: dict[str, int] = {}
    chat_template: bytes | None = None

    for _ in range(metadata_count):
        key = _decode_text(reader.text(limits.max_key_bytes, "max_key_bytes"))
        if key is None:
            # An undecodable key is walked past, not guessed at.
            _read_value(reader, limits, retain=False)
            continue
        wanted = (
            key in _RETAINED_KEYS
            or key == _CHAT_TEMPLATE_KEY
            or key.endswith(_CONTEXT_LENGTH_SUFFIX)
        )
        value = _read_value(reader, limits, retain=wanted)
        if not wanted:
            continue
        if key == _CHAT_TEMPLATE_KEY:
            if isinstance(value, bytes):
                chat_template = value
            continue
        if key.endswith(_CONTEXT_LENGTH_SUFFIX):
            if isinstance(value, int) and not isinstance(value, bool):
                context_lengths[key[: -len(_CONTEXT_LENGTH_SUFFIX)]] = value
            continue
        retained[key] = value

    descriptor_start = reader.offset
    tensor_type_counts: dict[str, int] = {}
    tensor_offsets: list[int] = []
    for _ in range(tensor_count):
        reader.skip_text(limits.max_tensor_name_bytes, "max_tensor_name_bytes")
        dimensions = reader.scalar(_TYPE_UINT32)
        if dimensions > limits.max_tensor_dimensions:
            raise _refuse(
                "limit_exceeded",
                "GGUF declared a tensor with more dimensions than the inspection limit",
                limit="max_tensor_dimensions",
                limit_value=limits.max_tensor_dimensions,
                declared_length=dimensions,
            )
        for _dimension in range(dimensions):
            reader.length()
        tensor_type = reader.scalar(_TYPE_UINT32)
        tensor_offsets.append(reader.length())
        key = str(tensor_type)
        tensor_type_counts[key] = tensor_type_counts.get(key, 0) + 1
        if reader.offset - descriptor_start > limits.max_tensor_descriptor_bytes:
            raise _refuse(
                "limit_exceeded",
                "GGUF tensor descriptors exceed the inspection byte budget",
                limit="max_tensor_descriptor_bytes",
                limit_value=limits.max_tensor_descriptor_bytes,
            )

    alignment = _resolve_alignment(retained.get("general.alignment"))
    remainder = reader.offset % alignment
    data_offset = reader.offset if remainder == 0 else reader.offset + (alignment - remainder)
    if data_offset > file_size:
        raise _refuse(
            "impossible_offset",
            "GGUF tensor data begins past the end of the file",
            tensor_data_offset=data_offset,
            file_size_bytes=file_size,
        )
    for tensor_offset in tensor_offsets:
        if data_offset + tensor_offset > file_size:
            raise _refuse(
                "impossible_offset",
                "a GGUF tensor descriptor points past the end of the file",
                tensor_data_offset=data_offset,
                declared_tensor_offset=tensor_offset,
                file_size_bytes=file_size,
            )

    architecture_raw = retained.get("general.architecture")
    architecture = (
        _decode_text(architecture_raw) or UNASSESSED
        if isinstance(architecture_raw, bytes)
        else UNASSESSED
    )
    name_raw = retained.get("general.name")
    model_name = (
        _decode_text(name_raw) or UNASSESSED if isinstance(name_raw, bytes) else UNASSESSED
    )
    context_length: int | str = context_lengths.get(architecture, UNASSESSED)
    file_type_id = retained.get("general.file_type")
    quantization_version = retained.get("general.quantization_version")

    unassessed = []
    if architecture == UNASSESSED:
        unassessed.append("architecture")
    if model_name == UNASSESSED:
        unassessed.append("model_name")
    if context_length == UNASSESSED:
        unassessed.append("context_length")
    if not isinstance(file_type_id, int) or isinstance(file_type_id, bool):
        file_type_id = UNASSESSED
        unassessed.append("file_type_id")
    if not isinstance(quantization_version, int) or isinstance(quantization_version, bool):
        quantization_version = UNASSESSED
        unassessed.append("quantization_version")
    # The identifier is read; no local table binds it to a label.
    unassessed.append("file_type_label")
    if chat_template is None:
        unassessed.append("chat_template")

    return GgufMetadata(
        path=str(file_path),
        file_size_bytes=file_size,
        gguf_version=gguf_version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        alignment=alignment,
        architecture=architecture,
        model_name=model_name,
        context_length=context_length,
        chat_template_present=chat_template is not None,
        chat_template_sha256=(
            hashlib.sha256(chat_template).hexdigest() if chat_template is not None else None
        ),
        chat_template_bytes=len(chat_template) if chat_template is not None else None,
        tensor_type_counts=tensor_type_counts,
        file_type_id=file_type_id,
        quantization_version=quantization_version,
        tensor_data_offset=data_offset,
        max_read_offset=reader.max_read_offset,
        unassessed_fields=tuple(unassessed),
    )
