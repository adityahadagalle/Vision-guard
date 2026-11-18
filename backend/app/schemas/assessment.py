"""API request and response schemas for assessment and finding endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateAssessmentRequest(BaseModel):
    """Payload to trigger an assessment on an approved dataset snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Automated Assessment", min_length=1, max_length=255)
    detectors: list[str] | None = Field(
        default=None,
        description="Optional list of detector IDs to execute. Defaults to all registered Tier-1 detectors.",
    )
    config: dict[str, Any] = Field(default_factory=dict)


class DetectorRunSummary(BaseModel):
    """Summary of a detector execution within an assessment."""

    run_id: str
    detector_id: str
    detector_version: str
    status: str
    status_reason: str | None = None
    duration_ms: int
    started_at: datetime
    completed_at: datetime | None = None


class AssessmentResponse(BaseModel):
    """Assessment metadata and execution summary."""

    assessment_id: str
    name: str
    dataset_id: str | None = None
    model_id: str | None = None
    snapshot_id: str
    status: str
    status_reason: str | None = None
    config_digest: str
    started_at: datetime
    completed_at: datetime | None = None
    runs: list[DetectorRunSummary] = Field(default_factory=list)
    job_id: str | None = None


class FindingAffectedAssetSchema(BaseModel):
    """Reference to an asset or sample affected by an integrity defect."""

    asset_id: str
    sample_path: str | None = None
    sha256: str | None = None


class FindingResponse(BaseModel):
    """Schema representing an individual integrity finding."""

    finding_id: str
    assessment_id: str
    detector_id: str
    detector_version: str
    category: str
    severity: str
    title: str
    description: str
    anomaly_score: float | None = None
    confidence: float
    suggested_disposition: str
    status: str
    affected_assets: list[FindingAffectedAssetSchema] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class PaginatedFindingsResponse(BaseModel):
    """Paginated collection of findings for an assessment."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    items: list[FindingResponse] = Field(default_factory=list)
