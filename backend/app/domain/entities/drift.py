"""Domain models for distribution shift and covariate drift detection.

Defines structured schemas for individual statistical drift metrics and
aggregated multi-metric drift reports. Enforces ADR-012 epistemic honesty,
distinguishing between statistical distances, p-values, decision thresholds,
and capability states.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.enums import CapabilityStatus


class DriftMetric(BaseModel):
    """Statistical evaluation metric representing covariate or concept drift."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    metric_name: str = Field(
        ...,
        description="Standard identifier of the statistical drift metric (e.g. kolmogorov_smirnov, wasserstein_distance).",
    )
    drift_score: float | None = Field(
        default=None,
        description="Computed statistical distance or test statistic (None if inconclusive).",
    )
    threshold: float = Field(
        ...,
        description="Configured decision threshold or alpha significance level for flagging drift.",
    )
    p_value: float | None = Field(
        default=None,
        description="Statistical p-value if applicable (e.g., from two-sample hypothesis tests).",
    )
    is_drifting: bool = Field(
        default=False,
        description="Whether this metric indicates significant distribution shift.",
    )
    capability_status: CapabilityStatus = Field(
        default=CapabilityStatus.COMPLETED,
        description="Execution capability status (COMPLETED, INCONCLUSIVE, UNAVAILABLE, etc.).",
    )
    status_reason: str | None = Field(
        default=None,
        description="Epistemic diagnostic reason when status is not COMPLETED.",
    )
    reference_sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of clean reference samples evaluated in baseline window.",
    )
    production_sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of production samples evaluated in candidate window.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting distributional moments, quantiles, and diagnostics.",
    )


class DriftReport(BaseModel):
    """Aggregated distribution shift evaluation report across one or more metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    report_id: str = Field(
        default_factory=lambda: f"drf_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this drift evaluation report.",
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Evaluation timestamp in UTC.",
    )
    dataset_or_model_id: str | None = Field(
        default=None,
        description="Identifier of the dataset, snapshot, or model evaluated.",
    )
    metrics: list[DriftMetric] = Field(
        default_factory=list,
        description="Collection of individual statistical drift metrics.",
    )
    is_drifting: bool = Field(
        default=False,
        description="Overall drift verdict: True if any capable metric flagged distribution shift.",
    )
    overall_status: CapabilityStatus = Field(
        default=CapabilityStatus.COMPLETED,
        description="Aggregated execution capability state across all metrics.",
    )
    status_reason: str | None = Field(
        default=None,
        description="Diagnostic explanation if overall status is INCONCLUSIVE, UNAVAILABLE, or FAILED.",
    )
    reference_sample_count: int = Field(
        default=0,
        ge=0,
        description="Total reference samples across evaluation windows.",
    )
    production_sample_count: int = Field(
        default=0,
        ge=0,
        description="Total production samples across evaluation windows.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary execution context, hyperparameters, and environment metadata.",
    )
