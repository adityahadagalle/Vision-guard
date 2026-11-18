"""Domain models for universal integrity findings.

Provides the standardized schema for all anomalies, corruptions,
Trojan alerts, and distribution shift signals detected across VisionGuard.
"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.domain.entities.asset import validate_sha256_hex, validate_timezone_aware
from backend.app.domain.entities.enums import (
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import validate_finite_number

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{3,128}$")
_DETECTOR_ID_PATTERN = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")


class AffectedAssetReference(BaseModel):
    """References an affected asset or individual sample within an asset snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    sample_path: str | None = None
    sha256: str | None = None

    @field_validator("asset_id")
    @classmethod
    def _validate_asset_id(cls, v: str) -> str:
        if not isinstance(v, str) or not _IDENTIFIER_PATTERN.match(v):
            raise ValueError(f"asset_id must be a valid alphanumeric identifier, got '{v}'.")
        return v

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_sha256_hex(v)
        return None

    @field_validator("sample_path")
    @classmethod
    def _validate_sample_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise ValueError("sample_path cannot be empty.")
        if clean.startswith("/") or clean.startswith("\\") or ":" in clean or ".." in clean:
            raise ValueError(f"sample_path must be a clean relative path, got '{v}'.")
        return clean.replace("\\", "/")


class ReproducibilityMetadata(BaseModel):
    """Metadata enabling deterministic re-execution and audit verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_timestamp: datetime
    seed: int | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)
    detector_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("execution_timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        return validate_timezone_aware(v)


class UniversalFinding(BaseModel):
    """Standardized finding schema shared by all VisionGuard detector modules.

    Enforces strict decoupling of three distinct evaluation concepts:
    1. anomaly_score: Magnitude of statistical or mathematical deviation.
    2. confidence: Calibrated probability [0.0, 1.0] that the finding is reliable.
    3. severity: Qualitative consequence rating (INFORMATIONAL through CRITICAL).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str
    assessment_id: str
    detector_id: str
    detector_version: str
    category: FindingCategory
    severity: SeverityLevel
    title: str
    description: str
    confidence: float
    reproducibility: ReproducibilityMetadata
    created_at: datetime
    run_id: str | None = None
    status: CapabilityStatus = CapabilityStatus.COMPLETED
    suggested_disposition: SuggestedDisposition = SuggestedDisposition.REVIEW
    anomaly_score: float | None = None
    affected_assets: list[AffectedAssetReference] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("finding_id", "assessment_id")
    @classmethod
    def _validate_identifiers(cls, v: str, info) -> str:
        if not isinstance(v, str) or not _IDENTIFIER_PATTERN.match(v):
            raise ValueError(
                f"Field '{info.field_name}' must be an alphanumeric identifier (3-128 chars, "
                f"allowing hyphens, underscores, and dots), got '{v}'."
            )
        return v

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: str | None) -> str | None:
        if v is not None and (not isinstance(v, str) or not _IDENTIFIER_PATTERN.match(v)):
            raise ValueError(f"Field 'run_id' must be a valid identifier, got '{v}'.")
        return v

    @field_validator("detector_id")
    @classmethod
    def _validate_detector_id(cls, v: str) -> str:
        if not isinstance(v, str) or not _DETECTOR_ID_PATTERN.match(v):
            raise ValueError(
                f"Field 'detector_id' must be a dot-delimited lowercase identifier "
                f"(e.g., 'data_integrity.corrupt_image'), got '{v}'."
            )
        return v

    @field_validator("detector_version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field 'detector_version' cannot be empty.")
        return v.strip()

    @field_validator("title", "description")
    @classmethod
    def _validate_text_fields(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        val = validate_finite_number(v, "confidence")
        if val is None or val < 0.0 or val > 1.0:
            raise ValueError(f"Field 'confidence' must be within range [0.0, 1.0], got {v}.")
        return val

    @field_validator("anomaly_score")
    @classmethod
    def _validate_anomaly_score(cls, v: float | None) -> float | None:
        return validate_finite_number(v, "anomaly_score")

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        return validate_timezone_aware(v)

    @field_validator("evidence_ids")
    @classmethod
    def _validate_evidence_ids(cls, v: list[str]) -> list[str]:
        for eid in v:
            if not isinstance(eid, str) or not _IDENTIFIER_PATTERN.match(eid):
                raise ValueError(
                    f"Evidence reference ID must be a valid alphanumeric identifier, got '{eid}'."
                )
        return v
