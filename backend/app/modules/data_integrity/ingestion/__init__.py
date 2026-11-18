"""VisionGuard Secure Dataset Ingestion, Quarantine & Snapshot Subsystem."""

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    AlternateDataStreamError,
    ArchiveInspectionError,
    CompressionRatioExceededError,
    DecompressionBombError,
    DuplicateEntryError,
    EmptyFileError,
    EntryCountLimitExceededError,
    FileTooLargeError,
    ImageDecompressionBombError,
    IngestionError,
    InvalidImageError,
    MalformedAnnotationError,
    MalformedArchiveError,
    PathTraversalError,
    QuarantineError,
    ReservedDeviceNameError,
    SnapshotAlreadyExistsError,
    SnapshotIntegrityError,
    SymlinkNotPermittedError,
    UnsupportedFormatError,
    WorkerExecutionError,
    WorkerIsolationUnavailableError,
    WorkerTimeoutError,
)
from backend.app.modules.data_integrity.ingestion.formats import (
    BaseFormatAdapter,
    COCODatasetAdapter,
    DatasetFormatDetector,
    FolderPerClassDatasetAdapter,
    NormalizedBBox,
    NormalizedSample,
    ParsedDataset,
    YOLODatasetAdapter,
)
from backend.app.modules.data_integrity.ingestion.image_validator import (
    ImageValidationResult,
    SafeImageValidator,
)
from backend.app.modules.data_integrity.ingestion.inspector import (
    ArchiveInspectionResult,
    DefensiveArchiveInspector,
)
from backend.app.modules.data_integrity.ingestion.normalizer import (
    DatasetNormalizer,
    NormalizedDataset,
)
from backend.app.modules.data_integrity.ingestion.quarantine import (
    QuarantineManager,
    QuarantineRecord,
    sanitize_filename,
)
from backend.app.modules.data_integrity.ingestion.service import (
    IngestionResult,
    IngestionService,
)
from backend.app.modules.data_integrity.ingestion.snapshot import (
    SnapshotManager,
    SnapshotVerificationResult,
    canonical_json_bytes,
)
from backend.app.modules.data_integrity.ingestion.worker import (
    SubprocessExtractionWorker,
    WorkerExtractionResult,
)

__all__ = [
    "AlternateDataStreamError",
    "ArchiveInspectionError",
    "ArchiveInspectionResult",
    "BaseFormatAdapter",
    "COCODatasetAdapter",
    "CompressionRatioExceededError",
    "DatasetFormatDetector",
    "DatasetNormalizer",
    "DecompressionBombError",
    "DefensiveArchiveInspector",
    "DuplicateEntryError",
    "EmptyFileError",
    "EntryCountLimitExceededError",
    "FileTooLargeError",
    "FolderPerClassDatasetAdapter",
    "ImageDecompressionBombError",
    "ImageValidationResult",
    "IngestionConfig",
    "IngestionError",
    "IngestionResult",
    "IngestionService",
    "InvalidImageError",
    "MalformedAnnotationError",
    "MalformedArchiveError",
    "NormalizedBBox",
    "NormalizedDataset",
    "NormalizedSample",
    "ParsedDataset",
    "PathTraversalError",
    "QuarantineError",
    "QuarantineManager",
    "QuarantineRecord",
    "ReservedDeviceNameError",
    "SafeImageValidator",
    "SnapshotAlreadyExistsError",
    "SnapshotIntegrityError",
    "SnapshotManager",
    "SnapshotVerificationResult",
    "SubprocessExtractionWorker",
    "SymlinkNotPermittedError",
    "UnsupportedFormatError",
    "WorkerExecutionError",
    "WorkerExtractionResult",
    "WorkerIsolationUnavailableError",
    "WorkerTimeoutError",
    "YOLODatasetAdapter",
    "canonical_json_bytes",
    "sanitize_filename",
]
