"""API request and response schemas for model artifact endpoints.

Enforces strict Pydantic v2 validation, frozen immutability, and prohibition of
unrecognized extra fields across all model integrity ingestion and query boundaries.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelUploadResponse(BaseModel):
    """Schema representing an uploaded model artifact."""

    model_config = ConfigDict(from_attributes=True, extra="forbid", frozen=True)

    model_id: str
    name: str
    format: str
    status: str
    sha256: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    snapshot_id: str | None = None
    job_id: str | None = None


class ModelDetailResponse(ModelUploadResponse):
    """Detailed model representation including snapshot and assessment references."""

    snapshot_manifest_sha256: str | None = None
    snapshot_file_count: int | None = None
    assessments_count: int = 0


class ModelListResponse(BaseModel):
    """Paginated collection of model artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int = Field(ge=1, description="1-indexed page number")
    page_size: int = Field(ge=1, le=100, description="Items per page")
    total_items: int = Field(ge=0, description="Total models matching filter")
    total_pages: int = Field(ge=0, description="Total available pages")
    items: list[ModelUploadResponse] = Field(default_factory=list)


class CreateModelAssessmentRequest(BaseModel):
    """Payload to trigger an integrity assessment on an approved model checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        default="Automated Model Assessment",
        min_length=1,
        max_length=255,
        description="Friendly human-readable label for this assessment run",
    )
    detectors: list[str] | None = Field(
        default=None,
        description="Optional list of specific detector IDs to execute. Defaults to all registered Model Integrity detectors.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution parameters, probe configurations, or reference artifact digests",
    )
