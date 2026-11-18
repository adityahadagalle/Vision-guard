"""VisionGuard Shared Domain Layer.

Contains pure domain entities, value objects, ports (abstract interfaces),
and the in-memory detector registry. Completely independent of frameworks,
ORMs, databases, and deep learning execution runtimes.
"""

from backend.app.domain.entities import (
    AffectedAssetReference,
    AssessmentStatus,
    AssetSnapshot,
    AssetType,
    CapabilityStatus,
    EvidenceRecord,
    EvidenceType,
    FindingCategory,
    IngestionStatus,
    JobStatus,
    ManifestEntry,
    ReproducibilityMetadata,
    SeverityLevel,
    SuggestedDisposition,
    UniversalFinding,
)
from backend.app.domain.ports import (
    ArtifactStoragePort,
    AssessmentContext,
    AssessmentRepositoryPort,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
    JobQueuePort,
)
from backend.app.domain.registry import (
    DetectorNotFoundError,
    DetectorRegistry,
    DetectorRegistryError,
    DuplicateDetectorError,
)

__all__ = [
    "AffectedAssetReference",
    "ArtifactStoragePort",
    "AssessmentContext",
    "AssessmentRepositoryPort",
    "AssessmentStatus",
    "AssetSnapshot",
    "AssetType",
    "BaseDetector",
    "CapabilityStatus",
    "DetectorNotFoundError",
    "DetectorRegistry",
    "DetectorRegistryError",
    "DetectorRequirements",
    "DetectorResult",
    "DuplicateDetectorError",
    "EvidenceRecord",
    "EvidenceType",
    "FindingCategory",
    "IngestionStatus",
    "JobQueuePort",
    "JobStatus",
    "ManifestEntry",
    "ReproducibilityMetadata",
    "SeverityLevel",
    "SuggestedDisposition",
    "UniversalFinding",
]
