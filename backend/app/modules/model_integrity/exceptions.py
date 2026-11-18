"""Structured exception hierarchy for the Model Integrity Engine.

Provides distinct, machine-readable error codes and forensic details
for model ingestion, checkpoint inspection, architecture validation,
and worker isolation failures.
"""

from typing import Any


class ModelIntegrityError(Exception):
    """Base exception for all Model Integrity Engine failures."""

    def __init__(
        self,
        message: str,
        code: str = "MODEL_INTEGRITY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ModelFileTooLargeError(ModelIntegrityError):
    """Raised when an uploaded model artifact exceeds maximum size limit."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODEL_FILE_TOO_LARGE", details=details)


class EmptyModelFileError(ModelIntegrityError):
    """Raised when an uploaded model file is empty (0 bytes)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="EMPTY_MODEL_FILE", details=details)


class InvalidModelExtensionError(ModelIntegrityError):
    """Raised when a model file has an unsupported extension (not .pt or .pth)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="INVALID_MODEL_EXTENSION", details=details)


class ModelDigestMismatchError(ModelIntegrityError):
    """Raised when candidate model artifact digest does not match expected reference digest."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODEL_DIGEST_MISMATCH", details=details)


class UnsafeCheckpointError(ModelIntegrityError):
    """Raised when checkpoint loading detects untrusted bytecode, exploit payloads, or unpickler errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="UNSAFE_CHECKPOINT", details=details)


class UnsupportedCheckpointFormatError(ModelIntegrityError):
    """Raised when checkpoint format is unsupported (e.g. non-dictionary root, TorchScript, ONNX)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="UNSUPPORTED_CHECKPOINT_FORMAT", details=details)


class MalformedCheckpointError(ModelIntegrityError):
    """Raised when checkpoint file is corrupted, truncated, or unparseable."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MALFORMED_CHECKPOINT", details=details)


class ModelIsolationUnavailableError(ModelIntegrityError):
    """Raised when required sandbox or process isolation cannot be guaranteed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODEL_ISOLATION_UNAVAILABLE", details=details)


class ModelWorkerExecutionError(ModelIntegrityError):
    """Raised when isolated model worker crashes or exits with non-zero status."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODEL_WORKER_EXECUTION_ERROR", details=details)


class ModelWorkerTimeoutError(ModelIntegrityError):
    """Raised when model inspection worker exceeds execution timeout."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODEL_WORKER_TIMEOUT", details=details)


class ArchitectureCompatibilityError(ModelIntegrityError):
    """Raised when state dict parameters do not conform to expected architecture specification."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="ARCHITECTURE_COMPATIBILITY_ERROR", details=details)


class UnknownArchitectureError(ModelIntegrityError):
    """Raised when requested architecture identifier is not registered in TrustedArchitectureRegistry."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="UNKNOWN_ARCHITECTURE", details=details)


class NonFiniteWeightsError(ModelIntegrityError):
    """Raised when state dict tensors contain NaN or Infinite values."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="NON_FINITE_WEIGHTS", details=details)


class PathTraversalError(ModelIntegrityError):
    """Raised when model filename or path attempts directory traversal."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PATH_TRAVERSAL_DETECTED", details=details)


class InvalidModelIdError(ModelIntegrityError):
    """Raised when model ID does not match strict validation policy."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="INVALID_MODEL_ID", details=details)


class InvalidFilenameError(ModelIntegrityError):
    """Raised when candidate model filename contains illegal characters or formats."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="INVALID_FILENAME", details=details)


class SymlinkNotPermittedError(ModelIntegrityError):
    """Raised when candidate model path or target directory involves a symlink."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="SYMLINK_NOT_PERMITTED", details=details)


class UploadConcurrencyLimitExceededError(ModelIntegrityError):
    """Raised when concurrent upload quota across API processes is exhausted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="CONCURRENCY_LIMIT_EXCEEDED", details=details)


class QuarantineQuotaExceededError(ModelIntegrityError):
    """Raised when global quarantine storage capacity or active reservations exceed quota."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="QUOTA_EXCEEDED", details=details)


class DiskSpaceExhaustedError(ModelIntegrityError):
    """Raised when host filesystem partition lacks adequate disk space."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="DISK_SPACE_EXHAUSTED", details=details)


class InvalidModelStateTransitionError(ModelIntegrityError):
    """Raised when an illegal or stale model lifecycle state transition is attempted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="INVALID_MODEL_STATE_TRANSITION", details=details)


class ReservationExpiredError(ModelIntegrityError):
    """Raised when an upload reservation has expired or lapsed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="RESERVATION_EXPIRED", details=details)
