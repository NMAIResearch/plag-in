"""Typed errors for PLAG IN.

Every error carries a stable machine-readable `error_type` so a client can
branch on failure without parsing prose. Unsupported capabilities and
security refusals must surface through these types rather than a generic
exception or a silently rewritten response.
"""
from __future__ import annotations


class PlagInError(Exception):
    error_type = "plag_in_error"
    http_status = 500

    def __init__(self, message: str, **fields):
        super().__init__(message)
        self.message = message
        self.fields = fields

    def to_dict(self) -> dict:
        return {"error": {"type": self.error_type, "message": self.message, **self.fields}}


class ConfigurationError(PlagInError):
    error_type = "configuration_error"
    http_status = 400


class UnknownConfigurationFieldError(ConfigurationError):
    error_type = "unknown_configuration_field"


class ExecutableNotAllowedError(PlagInError):
    error_type = "executable_not_allowed"
    http_status = 403


class PathContainmentError(PlagInError):
    error_type = "path_containment_violation"
    http_status = 403


class ModelIdentityMismatchError(PlagInError):
    error_type = "model_identity_mismatch"
    http_status = 409


class UnsupportedCapabilityError(PlagInError):
    error_type = "unsupported_capability"
    http_status = 501


class AuthenticationError(PlagInError):
    error_type = "authentication_error"
    http_status = 401


class LocalityPolicyError(PlagInError):
    error_type = "locality_policy_violation"
    http_status = 400


class PortConflictError(PlagInError):
    error_type = "port_conflict"
    http_status = 409


class ProcessIdentityError(PlagInError):
    error_type = "process_identity_mismatch"
    http_status = 409


class BackendUnavailableError(PlagInError):
    error_type = "backend_unavailable"
    http_status = 502


class NativeRuntimeError(BackendUnavailableError):
    error_type = "native_runtime_error"


class NativeInitializationError(NativeRuntimeError):
    """An unanticipated failure during native load (e.g. a missing symbol).

    The message is always a fixed string with no interpolated exception
    detail, so a native loader error (which on some platforms embeds the
    library's local file path) can never reach the public error body.
    """

    error_type = "native_initialization_failed"


class ReceiptChainError(PlagInError):
    error_type = "receipt_chain_violation"
    http_status = 409


class ReceiptNotFoundError(PlagInError):
    error_type = "receipt_not_found"
    http_status = 404


class InvalidRequestError(PlagInError):
    error_type = "invalid_request"
    http_status = 400


class InvalidAliasError(PlagInError):
    error_type = "invalid_alias"
    http_status = 400


class PayloadTooLargeError(PlagInError):
    error_type = "payload_too_large"
    http_status = 413


class OverloadedError(PlagInError):
    error_type = "overloaded"
    http_status = 429


class BackendResponseError(PlagInError):
    error_type = "backend_response_invalid"
    http_status = 502


class ReceiptPersistenceError(PlagInError):
    error_type = "receipt_persistence_failed"
    http_status = 500


class ReceiptCheckpointError(ReceiptPersistenceError):
    error_type = "receipt_checkpoint_failed"


class RequestTimeoutError(PlagInError):
    error_type = "request_timeout"
    http_status = 408


class RequestCancelledError(BackendUnavailableError):
    """Native generation stopped before a completed response existed."""

    error_type = "request_cancelled"
    http_status = 499


class AliasAlreadyRunningError(PlagInError):
    error_type = "alias_already_running"
    http_status = 409
