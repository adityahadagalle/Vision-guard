"""Domain model for forensic evidence records supporting integrity findings.

Evidence records provide transparent, reproducible mathematical or
observational proof anchoring detector findings to specific samples or signals.
"""

import math
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.domain.entities.asset import validate_sha256_hex, validate_timezone_aware
from backend.app.domain.entities.enums import EvidenceType

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{3,128}$")


def validate_finite_number(value: float | None, field_name: str) -> float | None:
    """Validates that a floating-point value is finite (neither NaN nor infinite)."""
    if value is not None:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Field '{field_name}' must be numeric, got {type(value)}.")
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Field '{field_name}' must be a finite numerical value, got {value}.")
        return float(value)
    return None


class EvidenceRecord(BaseModel):
    """Represents forensic evidence supporting a detector finding.

    Anchors findings to concrete signals, metrics, or sample references without
    exposing arbitrary host filesystem paths or unvalidated binary blobs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    evidence_type: EvidenceType
    finding_id: str
    signal_name: str
    rationale: str
    created_at: datetime
    sample_reference: str | None = None
    evidence_digest: str | None = None
    observed_value: float | None = None
    threshold_value: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "finding_id")
    @classmethod
    def _validate_identifier(cls, v: str, info) -> str:
        if not isinstance(v, str) or not _IDENTIFIER_PATTERN.match(v):
            raise ValueError(
                f"Field '{info.field_name}' must be an alphanumeric identifier (3-128 chars, "
                f"allowing hyphens, underscores, and dots), got '{v}'."
            )
        return v

    @field_validator("signal_name", "rationale")
    @classmethod
    def _validate_non_empty_string(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        return validate_timezone_aware(v)

    @field_validator("evidence_digest")
    @classmethod
    def _validate_evidence_digest(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_sha256_hex(v)
        return None

    @field_validator("observed_value", "threshold_value")
    @classmethod
    def _validate_numerical_signals(cls, v: float | None, info) -> float | None:
        return validate_finite_number(v, info.field_name)

    @field_validator("sample_reference")
    @classmethod
    def _validate_sample_reference(cls, v: str | None) -> str | None:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("sample_reference cannot be empty string.")
        if clean.startswith("/") or clean.startswith("\\") or ":" in clean:
            raise ValueError(
                f"sample_reference must be a relative sample path or digest, got absolute reference '{v}'."
            )
        if ".." in clean:
            raise ValueError(f"sample_reference cannot contain directory traversal ('..'): '{v}'.")
        return clean.replace("\\", "/")
