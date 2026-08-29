"""Configuration loading and validation.

Unknown fields in a security-sensitive section fail closed rather than
being silently ignored. The `security` and `bind` sections are treated as
security-sensitive because they control authentication and network
exposure (THREAT_MODEL.md trust boundaries 1 and 4).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from plag_in.aliasing import validate_alias
from plag_in.compatibility_records import find_compatibility_record
from plag_in.errors import ConfigurationError, UnknownConfigurationFieldError
from plag_in.native_profiles import identify_native_abi_profile

_VALID_ORIGINS = {"loopback", "any"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "model_roots",
    "ollama_manifest_root",
    "engines",
    "bind",
    "security",
    "recommend_catalogue",
    "runtime",
    "profiles",
}
_BIND_KEYS = {"host", "port"}
_SECURITY_KEYS = {"auth_mode", "api_keys", "content_logging"}
_API_KEY_KEYS = {"id", "secret", "aliases", "endpoints", "origin", "expiry"}
_ENGINE_KEYS = {
    "executable",
    "library",
    "library_sha256",
    "ggml_base_library",
    "ggml_base_library_sha256",
    "ggml_library",
    "ggml_library_sha256",
    "backend_libraries",
    "upstream_identity",
}
_BACKEND_LIBRARY_KEYS = {"path", "sha256"}
_RUNTIME_KEYS = {
    "context_size",
    "gpu_layers",
    "threads",
    "batch_size",
    "ubatch_size",
    "parallel",
    "temperature",
    "top_p",
}
_PROFILE_KEYS = {
    "model_path",
    "engine",
    "display_name",
    "compatibility_status",
    "compatibility_record",
}

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an absolute UTC timestamp. Fails closed on anything ambiguous."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ConfigurationError(f"invalid expiry timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigurationError(
            f"expiry timestamp must be absolute (include a UTC offset or 'Z'): {value!r}"
        )
    return parsed.astimezone(timezone.utc)


def _reject_unknown(section_name: str, data: dict, allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise UnknownConfigurationFieldError(
            f"unknown field(s) in {section_name!r} section: {sorted(unknown)}",
            section=section_name,
            fields=sorted(unknown),
        )


def _require_strict_int(value, field_name: str) -> int:
    """Reject a bool, a float or a numeric string standing in for an int.

    `bool` is a subclass of `int` in Python, so an unguarded `int(value)`
    silently turns `true` into `1`; a bare `isinstance(value, int)` check
    would still accept that `bool`, so it is excluded explicitly.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an integer, not {type(value).__name__}")
    return value


def _require_finite_float(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a finite number, not {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigurationError(f"{field_name} must be a finite number")
    return value


def _require_sha256(value, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ConfigurationError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty_str(value, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field_name} must be a non-empty string, not {value!r}")
    return value


@dataclass(frozen=True)
class ApiKey:
    id: str
    secret: str
    aliases: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = ()
    origin: str = "loopback"
    expiry: str | None = None


@dataclass(frozen=True)
class BindConfig:
    host: str = "127.0.0.1"
    port: int = 0


@dataclass(frozen=True)
class SecurityConfig:
    auth_mode: str = "none"
    api_keys: tuple[ApiKey, ...] = ()
    content_logging: bool = False


@dataclass(frozen=True)
class BackendLibraryConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class EngineConfig:
    executable: str | None = None
    library: str | None = None
    library_sha256: str | None = None
    ggml_base_library: str | None = None
    ggml_base_library_sha256: str | None = None
    ggml_library: str | None = None
    ggml_library_sha256: str | None = None
    backend_libraries: tuple[BackendLibraryConfig, ...] = ()
    upstream_identity: str | None = None

    @property
    def kind(self) -> str:
        return "embedded" if self.library is not None else "executable"

    def identity_record(self) -> dict:
        if self.kind == "executable":
            return {"kind": self.kind, "executable": self.executable}
        return {
            "kind": self.kind,
            "library_sha256": self.library_sha256,
            "ggml_base_library_sha256": self.ggml_base_library_sha256,
            "ggml_library_sha256": self.ggml_library_sha256,
            "backend_library_sha256": [item.sha256 for item in self.backend_libraries],
            "upstream_identity": self.upstream_identity,
        }


@dataclass(frozen=True)
class ModelProfile:
    """One configured alias and the evidence behind its compatibility state.

    `compatibility_record` names a reviewed record in the source, and is set
    only when `compatibility_status` is `tested`. The status field alone is
    an operator assertion and is never treated as evidence (D-021).
    """

    model_path: Path
    engine: str
    display_name: str
    compatibility_status: str
    compatibility_record: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Explicit inference settings confirmed before model loading.

    These values are passed as arguments rather than inherited from a
    runtime's changing defaults. Sampling values are server fallbacks only;
    a client may provide request-specific values.
    """

    context_size: int = 8192
    gpu_layers: str = "auto"
    threads: int = -1
    batch_size: int = 2048
    ubatch_size: int = 512
    parallel: int = 1
    temperature: float = 0.8
    top_p: float = 0.95

    def as_dict(self) -> dict:
        return {
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "parallel": self.parallel,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def as_direct_record(self) -> dict:
        return {
            "engine": "llama_server_direct",
            "ollama_runtime_dependency": False,
            "arguments": self.as_dict(),
            "sampling_scope": "server fallback; explicit client request values take precedence",
            "measurement_status": "not_applicable_to_worker_adapter",
            "native_identity_digest": None,
            "gpu_offload": {
                "requested": self.gpu_layers,
                "observed": None,
                "measurement_method": (
                    "the worker adapter forwards to an external llama-server process; "
                    "no effective-value query is implemented by this adapter"
                ),
                "status": "unassessed",
            },
        }

    def as_embedded_request_record(self) -> dict:
        return {
            "engine": "libllama_embedded",
            "ollama_runtime_dependency": False,
            "requested": self.as_dict(),
            "sampling_scope": "fallback; explicit client request values take precedence",
        }


@dataclass(frozen=True)
class GatewayConfig:
    model_roots: tuple[Path, ...] = ()
    ollama_manifest_root: Path | None = None
    engines: dict[str, EngineConfig] = field(default_factory=dict)
    bind: BindConfig = field(default_factory=BindConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    recommend_catalogue: Path | None = None
    profiles: dict[str, ModelProfile] = field(default_factory=dict)


def _validate_compatibility_evidence(
    *,
    alias: str,
    profile_data: dict,
    compatibility_status: str,
    engine_name: str,
    engine_cfg: "EngineConfig",
) -> str | None:
    """Refuse a `tested` profile that is not backed by a reviewed record.

    Configuration may reference a record, never supply one, so a profile
    cannot assert `tested` for a model no reviewed trial covers. The record
    also names the exact native bundle that loaded the model, which is
    checked here against the engine's verified component digests. The model
    bytes themselves are checked where they are read, at serve.
    """
    declared = profile_data.get("compatibility_record")

    if compatibility_status != "tested":
        if declared is not None:
            raise ConfigurationError(
                f"profiles.{alias}.compatibility_record is only valid with "
                "compatibility_status 'tested'",
                alias=alias,
                compatibility_status=compatibility_status,
            )
        return None

    if declared is None:
        raise ConfigurationError(
            f"profiles.{alias}.compatibility_status 'tested' requires "
            "compatibility_record naming a reviewed compatibility record",
            alias=alias,
        )
    record_id = _require_nonempty_str(declared, f"profiles.{alias}.compatibility_record")
    record = find_compatibility_record(record_id)
    if record is None:
        raise ConfigurationError(
            f"profiles.{alias}.compatibility_record names no reviewed compatibility record",
            alias=alias,
            compatibility_record=record_id,
        )

    if engine_cfg.kind != "embedded":
        raise ConfigurationError(
            f"profiles.{alias} claims a compatibility record, but engine "
            f"{engine_name!r} declares no native component digests to match it",
            alias=alias,
            engine=engine_name,
            compatibility_record=record_id,
            required_native_abi_profile=record.native_abi_profile_id,
        )

    observed = identify_native_abi_profile(
        upstream_identity=engine_cfg.upstream_identity,
        library_sha256=engine_cfg.library_sha256,
        ggml_base_library_sha256=engine_cfg.ggml_base_library_sha256,
        ggml_library_sha256=engine_cfg.ggml_library_sha256,
        backend_library_sha256=tuple(item.sha256 for item in engine_cfg.backend_libraries),
    )
    if observed is None or observed.profile_id != record.native_abi_profile_id:
        raise ConfigurationError(
            f"profiles.{alias} engine {engine_name!r} is not the native bundle "
            "the reviewed compatibility record was completed against",
            alias=alias,
            engine=engine_name,
            compatibility_record=record_id,
            required_native_abi_profile=record.native_abi_profile_id,
            observed_native_abi_profile=observed.profile_id if observed else None,
        )
    return record_id


def with_trial_profile(
    config: GatewayConfig,
    alias: str,
    model_path: Path,
    engine: str,
    display_name: str,
) -> GatewayConfig:
    """Return a copy carrying one ephemeral unverified profile.

    A trial profile exists for the life of one process and is never written
    to a configuration file: trying a model must not change what the
    computer is configured to serve (D-021). The returned profile is always
    `unverified`, so no trial can promote a model, and an alias already
    declared in the configuration is refused rather than shadowed.
    """
    validate_alias(alias)
    if alias in config.profiles:
        raise ConfigurationError(
            f"trial alias is already a configured profile: {alias!r}", alias=alias
        )
    if engine not in config.engines:
        raise ConfigurationError(
            f"trial profile refers to an unconfigured engine: {engine!r}", engine=engine
        )
    profiles = dict(config.profiles)
    profiles[alias] = ModelProfile(
        model_path=Path(model_path),
        engine=engine,
        display_name=display_name,
        compatibility_status="unverified",
    )
    return replace(config, profiles=profiles)


def load_config(data: dict) -> GatewayConfig:
    _reject_unknown("root", data, _TOP_LEVEL_KEYS)

    bind_data = data.get("bind", {})
    if not isinstance(bind_data, dict):
        raise ConfigurationError("'bind' must be an object")
    _reject_unknown("bind", bind_data, _BIND_KEYS)
    port = _require_strict_int(bind_data.get("port", 0), "bind.port")
    if not 0 <= port <= 65535:
        raise ConfigurationError(f"bind.port must be between 0 and 65535, got {port}")
    bind = BindConfig(host=bind_data.get("host", "127.0.0.1"), port=port)

    security_data = data.get("security", {})
    if not isinstance(security_data, dict):
        raise ConfigurationError("'security' must be an object")
    _reject_unknown("security", security_data, _SECURITY_KEYS)

    auth_mode = security_data.get("auth_mode", "none")
    if auth_mode not in ("none", "api_key"):
        raise ConfigurationError(f"unsupported auth_mode: {auth_mode!r}")

    api_keys = []
    for entry in security_data.get("api_keys", []):
        if not isinstance(entry, dict):
            raise ConfigurationError("each api_keys entry must be an object")
        _reject_unknown("security.api_keys[]", entry, _API_KEY_KEYS)

        origin = entry.get("origin", "loopback")
        if origin not in _VALID_ORIGINS:
            raise ConfigurationError(
                f"unsupported api key origin: {origin!r}, expected one of {sorted(_VALID_ORIGINS)}"
            )

        expiry = entry.get("expiry")
        if expiry is not None:
            parse_utc_timestamp(expiry)  # validated eagerly; raises ConfigurationError on failure

        aliases = tuple(entry.get("aliases", []))
        for alias in aliases:
            validate_alias(alias)

        api_keys.append(
            ApiKey(
                id=entry["id"],
                secret=entry["secret"],
                aliases=aliases,
                endpoints=tuple(entry.get("endpoints", [])),
                origin=origin,
                expiry=expiry,
            )
        )

    security = SecurityConfig(
        auth_mode=auth_mode,
        api_keys=tuple(api_keys),
        content_logging=bool(security_data.get("content_logging", False)),
    )

    if not is_loopback(bind.host) and auth_mode == "none":
        raise ConfigurationError(
            "non-loopback bind requires security.auth_mode=api_key",
            host=bind.host,
        )

    engines = {}
    for name, engine_data in data.get("engines", {}).items():
        if not isinstance(engine_data, dict):
            raise ConfigurationError(f"engine {name!r} configuration must be an object")
        _reject_unknown(f"engines.{name}", engine_data, _ENGINE_KEYS)

        executable = engine_data.get("executable")
        if executable is not None:
            executable = _require_nonempty_str(executable, f"engines.{name}.executable")
        library = engine_data.get("library")
        if library is not None:
            library = _require_nonempty_str(library, f"engines.{name}.library")
        if bool(executable) == bool(library):
            raise ConfigurationError(
                f"engine {name!r} must declare exactly one of 'executable' or 'library'"
            )

        backend_libraries = []
        for entry in engine_data.get("backend_libraries", []):
            if not isinstance(entry, dict):
                raise ConfigurationError(
                    f"each engines.{name}.backend_libraries entry must be an object"
                )
            _reject_unknown(f"engines.{name}.backend_libraries[]", entry, _BACKEND_LIBRARY_KEYS)
            if "path" not in entry or "sha256" not in entry:
                raise ConfigurationError(
                    f"engines.{name}.backend_libraries entries require path and sha256"
                )
            backend_libraries.append(
                BackendLibraryConfig(
                    path=_require_nonempty_str(
                        entry["path"], f"engines.{name}.backend_libraries[].path"
                    ),
                    sha256=_require_sha256(
                        entry["sha256"], f"engines.{name}.backend_libraries[].sha256"
                    ),
                )
            )

        library_sha256 = ggml_base_library = ggml_base_library_sha256 = None
        ggml_library = ggml_library_sha256 = upstream_identity = None
        if library:
            required = {
                "library_sha256": engine_data.get("library_sha256"),
                "ggml_base_library": engine_data.get("ggml_base_library"),
                "ggml_base_library_sha256": engine_data.get("ggml_base_library_sha256"),
                "ggml_library": engine_data.get("ggml_library"),
                "ggml_library_sha256": engine_data.get("ggml_library_sha256"),
                "upstream_identity": engine_data.get("upstream_identity"),
            }
            missing = sorted(key for key, value in required.items() if not value)
            if missing:
                raise ConfigurationError(
                    f"embedded engine {name!r} is missing required identity fields: {missing}"
                )
            if not backend_libraries:
                raise ConfigurationError(
                    f"embedded engine {name!r} requires at least one explicit backend library"
                )
            library_sha256 = _require_sha256(
                required["library_sha256"], f"engines.{name}.library_sha256"
            )
            ggml_base_library = _require_nonempty_str(
                required["ggml_base_library"], f"engines.{name}.ggml_base_library"
            )
            ggml_base_library_sha256 = _require_sha256(
                required["ggml_base_library_sha256"], f"engines.{name}.ggml_base_library_sha256"
            )
            ggml_library = _require_nonempty_str(
                required["ggml_library"], f"engines.{name}.ggml_library"
            )
            ggml_library_sha256 = _require_sha256(
                required["ggml_library_sha256"], f"engines.{name}.ggml_library_sha256"
            )
            upstream_identity = _require_nonempty_str(
                required["upstream_identity"], f"engines.{name}.upstream_identity"
            )

        engines[name] = EngineConfig(
            executable=executable,
            library=library,
            library_sha256=library_sha256,
            ggml_base_library=ggml_base_library,
            ggml_base_library_sha256=ggml_base_library_sha256,
            ggml_library=ggml_library,
            ggml_library_sha256=ggml_library_sha256,
            backend_libraries=tuple(backend_libraries),
            upstream_identity=upstream_identity,
        )

    runtime_data = data.get("runtime", {})
    if not isinstance(runtime_data, dict):
        raise ConfigurationError("'runtime' must be an object")
    _reject_unknown("runtime", runtime_data, _RUNTIME_KEYS)

    gpu_layers = runtime_data.get("gpu_layers", "auto")
    if not isinstance(gpu_layers, str):
        raise ConfigurationError(
            f"runtime.gpu_layers must be a string, not {type(gpu_layers).__name__}"
        )
    runtime = RuntimeConfig(
        context_size=_require_strict_int(runtime_data.get("context_size", 8192), "runtime.context_size"),
        gpu_layers=gpu_layers,
        threads=_require_strict_int(runtime_data.get("threads", -1), "runtime.threads"),
        batch_size=_require_strict_int(runtime_data.get("batch_size", 2048), "runtime.batch_size"),
        ubatch_size=_require_strict_int(runtime_data.get("ubatch_size", 512), "runtime.ubatch_size"),
        parallel=_require_strict_int(runtime_data.get("parallel", 1), "runtime.parallel"),
        temperature=_require_finite_float(runtime_data.get("temperature", 0.8), "runtime.temperature"),
        top_p=_require_finite_float(runtime_data.get("top_p", 0.95), "runtime.top_p"),
    )

    if runtime.context_size <= 0:
        raise ConfigurationError("runtime.context_size must be a positive token count")
    if runtime.gpu_layers not in {"auto", "all"} and not runtime.gpu_layers.isdigit():
        raise ConfigurationError("runtime.gpu_layers must be 'auto', 'all' or a non-negative integer")
    if runtime.threads != -1 and runtime.threads <= 0:
        raise ConfigurationError("runtime.threads must be -1 for automatic selection or a positive integer")
    if runtime.batch_size <= 0 or runtime.ubatch_size <= 0:
        raise ConfigurationError("runtime batch sizes must be positive")
    if runtime.ubatch_size > runtime.batch_size:
        raise ConfigurationError("runtime.ubatch_size must not exceed runtime.batch_size")
    if runtime.parallel <= 0:
        raise ConfigurationError("runtime.parallel must be positive")
    if runtime.temperature < 0:
        raise ConfigurationError("runtime.temperature must not be negative")
    if not 0 <= runtime.top_p <= 1:
        raise ConfigurationError("runtime.top_p must be between 0 and 1")

    model_roots = tuple(Path(p) for p in data.get("model_roots", []))
    ollama_root = data.get("ollama_manifest_root")
    recommend_catalogue = data.get("recommend_catalogue")

    profiles = {}
    profiles_data = data.get("profiles", {})
    if not isinstance(profiles_data, dict):
        raise ConfigurationError("'profiles' must be an object")
    for alias, profile_data in profiles_data.items():
        validate_alias(alias)
        if not isinstance(profile_data, dict):
            raise ConfigurationError(f"profile {alias!r} must be an object")
        _reject_unknown(f"profiles.{alias}", profile_data, _PROFILE_KEYS)
        missing = sorted({"model_path", "engine", "display_name", "compatibility_status"} - set(profile_data))
        if missing:
            raise ConfigurationError(f"profile {alias!r} is missing required fields: {missing}")
        engine_name = _require_nonempty_str(profile_data["engine"], f"profiles.{alias}.engine")
        if engine_name not in engines:
            raise ConfigurationError(
                f"profile {alias!r} refers to an unconfigured engine: {engine_name!r}"
            )
        compatibility_status = _require_nonempty_str(
            profile_data["compatibility_status"], f"profiles.{alias}.compatibility_status"
        )
        if compatibility_status not in {"tested", "unverified"}:
            raise ConfigurationError(
                f"profiles.{alias}.compatibility_status must be 'tested' or 'unverified'"
            )
        compatibility_record = _validate_compatibility_evidence(
            alias=alias,
            profile_data=profile_data,
            compatibility_status=compatibility_status,
            engine_name=engine_name,
            engine_cfg=engines[engine_name],
        )
        profiles[alias] = ModelProfile(
            model_path=Path(
                _require_nonempty_str(profile_data["model_path"], f"profiles.{alias}.model_path")
            ),
            engine=engine_name,
            display_name=_require_nonempty_str(
                profile_data["display_name"], f"profiles.{alias}.display_name"
            ),
            compatibility_status=compatibility_status,
            compatibility_record=compatibility_record,
        )

    return GatewayConfig(
        model_roots=model_roots,
        ollama_manifest_root=Path(ollama_root) if ollama_root else None,
        engines=engines,
        bind=bind,
        security=security,
        runtime=runtime,
        recommend_catalogue=Path(recommend_catalogue) if recommend_catalogue else None,
        profiles=profiles,
    )


def load_config_file(path: Path) -> GatewayConfig:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"configuration file is not valid JSON: {exc}") from exc
    return load_config(data)
