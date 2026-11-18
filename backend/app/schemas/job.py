"""API response schema for background job tracking."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    """Schema representing background job status and progress."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str
    task_type: str
    status: str
    attempts: int
    max_attempts: int
    error_message: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
