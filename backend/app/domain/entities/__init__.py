"""Domain entities and value objects for VisionGuard."""

from backend.app.domain.entities.asset import (
    AssetSnapshot,
    ManifestEntry,
    validate_sha256_hex,
    validate_timezone_aware,
)
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    AssetType,
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    IngestionStatus,
    JobStatus,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord, validate_finite_number
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)

__all__ = [
    "AffectedAssetReference",
    "AssessmentStatus",
    "AssetSnapshot",
    "AssetType",
    "CapabilityStatus",
    "EvidenceRecord",
    "EvidenceType",
    "FindingCategory",
    "IngestionStatus",
    "JobStatus",
    "ManifestEntry",
    "ReproducibilityMetadata",
    "SeverityLevel",
    "SuggestedDisposition",
    "UniversalFinding",
    "validate_finite_number",
    "validate_sha256_hex",
    "validate_timezone_aware",
]
