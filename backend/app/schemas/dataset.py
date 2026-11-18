"""API request and response schemas for dataset endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetResponse(BaseModel):
    """Basic dataset representation in API responses."""

    model_config = ConfigDict(from_attributes=True)

    dataset_id: str
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


class DatasetDetailResponse(DatasetResponse):
    """Detailed dataset representation including snapshot information."""

    snapshot_file_count: int | None = None
    snapshot_manifest_sha256: str | None = None
