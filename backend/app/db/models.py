"""SQLAlchemy 2.0 relational models for VisionGuard persistence.

Defines schemas for datasets, snapshots, assessments, detector executions,
findings, evidence records, and SQLite-backed jobs.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.app.domain.entities.enums import (
    AssetType,
    FindingCategory,
    IngestionStatus,
    JobStatus,
    SeverityLevel,
    SuggestedDisposition,
)


class UTCDateTime(TypeDecorator):
    """Stores timezone-aware UTC datetimes, ensuring naive SQLite datetimes get tzinfo=UTC on load."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None:
            value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    """Base declarative class for all VisionGuard database models."""

    pass


class DatasetModel(Base):
    """Stores dataset metadata, format, upload hashes, and ingestion lifecycle states."""

    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(64), nullable=False, default=IngestionStatus.RECEIVED
    )
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    # Relationships
    snapshots: Mapped[list["AssetSnapshotModel"]] = relationship(
        "AssetSnapshotModel", back_populates="dataset", cascade="all, delete-orphan"
    )
    assessments: Mapped[list["AssessmentModel"]] = relationship(
        "AssessmentModel", back_populates="dataset", cascade="all, delete-orphan"
    )


class ModelArtifactModel(Base):
    """Stores model checkpoint metadata, formats, upload hashes, and lifecycle states."""

    __tablename__ = "models"

    model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(64), nullable=False, default="pytorch")
    status: Mapped[str] = mapped_column(
        String(64), nullable=False, default=IngestionStatus.RECEIVED
    )
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    # Relationships
    snapshots: Mapped[list["AssetSnapshotModel"]] = relationship(
        "AssetSnapshotModel", back_populates="model", cascade="all, delete-orphan"
    )
    assessments: Mapped[list["AssessmentModel"]] = relationship(
        "AssessmentModel", back_populates="model", cascade="all, delete-orphan"
    )


class AssetSnapshotModel(Base):
    """Stores immutable asset snapshot references and authoritative manifest digests."""

    __tablename__ = "asset_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=True
    )
    model_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("models.model_id", ondelete="CASCADE"), nullable=True
    )
    asset_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default=AssetType.DATASET_SNAPSHOT
    )
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    # Relationships
    dataset: Mapped["DatasetModel | None"] = relationship(
        "DatasetModel", back_populates="snapshots"
    )
    model: Mapped["ModelArtifactModel | None"] = relationship(
        "ModelArtifactModel", back_populates="snapshots"
    )
    assessments: Mapped[list["AssessmentModel"]] = relationship(
        "AssessmentModel", back_populates="snapshot", cascade="all, delete-orphan"
    )


class AssessmentModel(Base):
    """Stores assessment evaluations, run configurations, and audit status."""

    __tablename__ = "assessments"

    assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=True
    )
    model_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("models.model_id", ondelete="CASCADE"), nullable=True
    )
    snapshot_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("asset_snapshots.snapshot_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    # Relationships
    dataset: Mapped["DatasetModel | None"] = relationship(
        "DatasetModel", back_populates="assessments"
    )
    model: Mapped["ModelArtifactModel | None"] = relationship(
        "ModelArtifactModel", back_populates="assessments"
    )
    snapshot: Mapped["AssetSnapshotModel"] = relationship(
        "AssetSnapshotModel", back_populates="assessments"
    )
    detector_runs: Mapped[list["DetectorRunModel"]] = relationship(
        "DetectorRunModel", back_populates="assessment", cascade="all, delete-orphan"
    )
    findings: Mapped[list["FindingModel"]] = relationship(
        "FindingModel", back_populates="assessment", cascade="all, delete-orphan"
    )


class DetectorRunModel(Base):
    """Tracks per-detector execution status, capability reporting, duration, and metrics."""

    __tablename__ = "detector_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    assessment_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False
    )
    detector_id: Mapped[str] = mapped_column(String(128), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    # Relationships
    assessment: Mapped["AssessmentModel"] = relationship(
        "AssessmentModel", back_populates="detector_runs"
    )
    findings: Mapped[list["FindingModel"]] = relationship(
        "FindingModel", back_populates="detector_run"
    )


class FindingModel(Base):
    """Stores universal integrity findings produced by detectors."""

    __tablename__ = "findings"

    finding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    assessment_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("detector_runs.run_id", ondelete="SET NULL"), nullable=True
    )
    detector_id: Mapped[str] = mapped_column(String(128), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(
        String(64), nullable=False, default=FindingCategory.STRUCTURAL_DEFECT
    )
    severity: Mapped[str] = mapped_column(
        String(32), nullable=False, default=SeverityLevel.INFORMATIONAL
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_disposition: Mapped[str] = mapped_column(
        String(32), nullable=False, default=SuggestedDisposition.REVIEW
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    detector_config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    reproducibility_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    # Relationships
    assessment: Mapped["AssessmentModel"] = relationship(
        "AssessmentModel", back_populates="findings"
    )
    detector_run: Mapped["DetectorRunModel | None"] = relationship(
        "DetectorRunModel", back_populates="findings"
    )
    evidence_records: Mapped[list["EvidenceRecordModel"]] = relationship(
        "EvidenceRecordModel", back_populates="finding", cascade="all, delete-orphan"
    )
    affected_assets: Mapped[list["FindingAffectedAssetModel"]] = relationship(
        "FindingAffectedAssetModel", back_populates="finding", cascade="all, delete-orphan"
    )


class EvidenceRecordModel(Base):
    """Stores quantitative forensic evidence records supporting findings."""

    __tablename__ = "evidence_records"

    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    finding_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("findings.finding_id", ondelete="CASCADE"), nullable=False
    )
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_name: Mapped[str] = mapped_column(String(128), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    sample_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    observed_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    # Relationships
    finding: Mapped["FindingModel"] = relationship(
        "FindingModel", back_populates="evidence_records"
    )


class FindingAffectedAssetModel(Base):
    """Stores affected asset references for an integrity finding."""

    __tablename__ = "finding_affected_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    finding_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("findings.finding_id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    sample_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    finding: Mapped["FindingModel"] = relationship("FindingModel", back_populates="affected_assets")


class JobModel(Base):
    """SQLite-backed task queue model for asynchronous background execution."""

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=JobStatus.QUEUED)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    __table_args__ = (
        Index("idx_jobs_status_created_at", "status", "created_at"),
        Index("idx_jobs_locked_at", "locked_at"),
        Index("idx_jobs_status_expires", "status", "lease_expires_at"),
        Index("idx_jobs_lease_token", "lease_token"),
    )


class UploadReservationModel(Base):
    """Tracks cross-process durable quota reservations and concurrency slots for uploads."""

    __tablename__ = "upload_reservations"

    reservation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False, default="model")
    reserved_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarantine_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="ACTIVE")
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    __table_args__ = (
        Index("idx_upload_reservations_status_expires", "status", "expires_at"),
        Index("idx_upload_reservations_created_at", "created_at"),
    )


class InferenceReceiptModel(Base):
    """Stores immutable cryptographic receipts and hash-chain linkage for inference events."""

    __tablename__ = "inference_receipts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    edge_device_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    timestamp_iso: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    inference_output: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    previous_receipt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    signature_b64: Mapped[str] = mapped_column(Text, nullable=False)
    public_key_pem: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("edge_device_id", "sequence_number", name="uq_inference_device_sequence"),
        Index("idx_inference_receipts_device_seq", "edge_device_id", "sequence_number"),
    )
