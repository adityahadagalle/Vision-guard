"""Domain-level enumerations for VisionGuard.

Defines controlled vocabularies for assets, findings, severities,
evidence types, dispositions, and execution capability states.
"""

from enum import StrEnum


class AssetType(StrEnum):
    """Supported artifact and asset types in the content-addressed store."""

    DATASET_ARCHIVE = "DATASET_ARCHIVE"
    DATASET_SNAPSHOT = "DATASET_SNAPSHOT"
    MODEL_CHECKPOINT = "MODEL_CHECKPOINT"
    REFERENCE_SET = "REFERENCE_SET"
    PROVENANCE_MANIFEST = "PROVENANCE_MANIFEST"


class FindingCategory(StrEnum):
    """Categorization of integrity issues identified by detectors."""

    CORRUPTED_FILE = "CORRUPTED_FILE"
    STRUCTURAL_DEFECT = "STRUCTURAL_DEFECT"
    DUPLICATE_SAMPLE = "DUPLICATE_SAMPLE"
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    LABEL_ANOMALY = "LABEL_ANOMALY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    SUSPECTED_POISON = "SUSPECTED_POISON"
    UNAUTHORIZED_MODEL_CHANGE = "UNAUTHORIZED_MODEL_CHANGE"
    INFERENCE_TAMPERING = "INFERENCE_TAMPERING"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    CONFLICT = "CONFLICT"


class SeverityLevel(StrEnum):
    """Qualitative severity rating representing consequence or operational risk."""

    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SuggestedDisposition(StrEnum):
    """Automated workflow recommendation requiring human-in-the-loop governance."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    QUARANTINE = "QUARANTINE"


class CapabilityStatus(StrEnum):
    """Eight-state formal capability reporting model.

    Enforces radical epistemic honesty: missing prerequisites or unexecuted
    detectors are NEVER reported as clean or passing.
    """

    COMPLETED = "COMPLETED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONFLICTED = "CONFLICTED"


class EvidenceType(StrEnum):
    """Classification of forensic evidence supporting a detector finding."""

    IMAGE_CORRUPTION = "IMAGE_CORRUPTION"
    HEX_DUMP = "HEX_DUMP"
    METRIC_DISTRIBUTION = "METRIC_DISTRIBUTION"
    STATISTICAL_TEST = "STATISTICAL_TEST"
    EMBEDDING_DISTANCE = "EMBEDDING_DISTANCE"
    HASH_COLLISION = "HASH_COLLISION"
    LABEL_CONFIDENCE_MARGIN = "LABEL_CONFIDENCE_MARGIN"
    ACTIVATION_CLUSTER = "ACTIVATION_CLUSTER"
    RECEIPT_CHAIN_BREAK = "RECEIPT_CHAIN_BREAK"
    METADATA_MISMATCH = "METADATA_MISMATCH"
    CONFLICT_CONTRADICTION = "CONFLICT_CONTRADICTION"


class AssessmentStatus(StrEnum):
    """Lifecycle states of an end-to-end assessment execution."""

    PENDING = "PENDING"
    INITIALIZING = "INITIALIZING"
    INGESTING = "INGESTING"
    ANALYZING = "ANALYZING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(StrEnum):
    """States for SQLite-backed task orchestration."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class IngestionStatus(StrEnum):
    """Lifecycle states of the secure dataset ingestion and quarantine pipeline."""

    RECEIVED = "RECEIVED"
    QUARANTINED = "QUARANTINED"
    INSPECTING = "INSPECTING"
    EXTRACTING = "EXTRACTING"
    NORMALIZING = "NORMALIZING"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
