"""Structured exception hierarchy for the secure dataset ingestion pipeline.

Ensures every ingestion rejection, quarantine error, decompression bomb,
or path traversal provides explicit, non-swallowed error telemetry.
"""

from typing import Any


class IngestionError(Exception):
    """Base exception for all dataset ingestion and validation failures."""

    def __init__(
        self, message: str, code: str = "INGESTION_ERROR", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class QuarantineError(IngestionError):
    """Raised when an error occurs during upload receipt or quarantine staging."""

    def __init__(
        self, message: str, code: str = "QUARANTINE_ERROR", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code=code, details=details)


class FileTooLargeError(QuarantineError):
    """Raised when an uploaded archive exceeds the configured upload size limit."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="FILE_SIZE_LIMIT_EXCEEDED", details=details)


class EmptyFileError(QuarantineError):
    """Raised when an uploaded file is 0 bytes."""

    def __init__(
        self, message: str = "Uploaded file is 0 bytes.", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="EMPTY_FILE_REJECTED", details=details)


class ArchiveInspectionError(IngestionError):
    """Raised when an archive fails defensive security pre-inspection."""

    def __init__(
        self,
        message: str,
        code: str = "ARCHIVE_INSPECTION_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class PathTraversalError(ArchiveInspectionError):
    """Raised when an archive entry attempts directory traversal (ZIP Slip)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PATH_TRAVERSAL_DETECTED", details=details)


class ReservedDeviceNameError(ArchiveInspectionError):
    """Raised when an archive entry uses a reserved Windows device name (e.g., CON, PRN)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="WINDOWS_RESERVED_NAME_DETECTED", details=details)


class AlternateDataStreamError(ArchiveInspectionError):
    """Raised when an archive entry attempts NTFS Alternate Data Stream injection (:)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="ALTERNATE_DATA_STREAM_DETECTED", details=details)


class SymlinkNotPermittedError(ArchiveInspectionError):
    """Raised when an archive entry contains a symlink or hardlink."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="LINK_NOT_PERMITTED", details=details)


class DecompressionBombError(ArchiveInspectionError):
    """Raised when an archive exceeds total uncompressed size or compression ratio limits."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="SUSPECTED_DECOMPRESSION_BOMB", details=details)


class CompressionRatioExceededError(DecompressionBombError):
    """Raised when an archive compression ratio exceeds the configured safety threshold."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)


class EntryCountLimitExceededError(ArchiveInspectionError):
    """Raised when an archive entry count exceeds the configured maximum."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="ENTRY_COUNT_EXCEEDED", details=details)


class DuplicateEntryError(ArchiveInspectionError):
    """Raised when an archive contains duplicate paths or case-insensitive collisions."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="DUPLICATE_OR_COLLIDING_ENTRY", details=details)


class MalformedArchiveError(ArchiveInspectionError):
    """Raised when an archive is corrupted or not a valid ZIP container."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MALFORMED_ARCHIVE", details=details)


class WorkerExecutionError(IngestionError):
    """Raised when the isolated extraction worker subprocess fails."""

    def __init__(
        self,
        message: str,
        code: str = "WORKER_EXECUTION_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class WorkerTimeoutError(WorkerExecutionError):
    """Raised when the worker subprocess exceeds its configured execution timeout."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="WORKER_TIMEOUT_EXCEEDED", details=details)


class WorkerIsolationUnavailableError(WorkerExecutionError):
    """Raised when mandatory security isolation controls cannot be established on host."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="ISOLATION_UNAVAILABLE", details=details)


class InvalidImageError(IngestionError):
    """Raised when an extracted image file fails structural or decoding validation."""

    def __init__(
        self, message: str, code: str = "INVALID_IMAGE_FILE", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code=code, details=details)


class ImageDecompressionBombError(InvalidImageError):
    """Raised when an image pixel count exceeds the configured maximum."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="IMAGE_DECOMPRESSION_BOMB_DETECTED", details=details)


class UnsupportedFormatError(IngestionError):
    """Raised when an extracted dataset does not match any approved format adapter."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="UNSUPPORTED_DATASET_FORMAT", details=details)


class MalformedAnnotationError(IngestionError):
    """Raised when annotations in an otherwise supported dataset format are malformed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MALFORMED_ANNOTATION_STRUCTURE", details=details)


class SnapshotIntegrityError(IngestionError):
    """Raised when snapshot manifest generation or verification fails."""

    def __init__(
        self,
        message: str,
        code: str = "SNAPSHOT_INTEGRITY_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class SnapshotAlreadyExistsError(SnapshotIntegrityError):
    """Raised when attempting to overwrite an already existing immutable snapshot directory."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="SNAPSHOT_ALREADY_EXISTS", details=details)


class SnapshotInUseError(SnapshotIntegrityError):
    """Raised when an operation is attempted on a snapshot that is currently locked or in use by active readers."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="SNAPSHOT_IN_USE", details=details)


class QuarantineCollisionError(QuarantineError):
    """Raised when a quarantine operation encounters an existing destination and refuses to overwrite it."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="QUARANTINE_COLLISION", details=details)


class QuarantineMetadataError(QuarantineError):
    """Raised when writing the quarantine_record.json audit file fails.

    The quarantined snapshot directory and its contents are preserved.
    This exception indicates that the audit record is incomplete and
    must be detected and completed during recovery.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="QUARANTINE_METADATA_WRITE_FAILED", details=details)
