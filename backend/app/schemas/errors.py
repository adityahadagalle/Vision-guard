"""Standardized structured error response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Structured error payload returned by all API endpoints on failure."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
