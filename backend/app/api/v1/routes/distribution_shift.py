"""API endpoints and database persistence for distribution shift and covariate drift detection.

Provides REST endpoints for:
1. Evaluating continuous 1D distribution drift across reference and production samples.
2. Querying historical drift assessment reports for a given model.
Enforces strict Pydantic validation (extra="forbid") and ADR-012 epistemic honesty.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, Boolean, Integer, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.app.db.models import Base, UTCDateTime
from backend.app.db.session import get_db
from backend.app.domain.entities.drift import DriftMetric, DriftReport
from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.distribution_shift.detectors.ks_drift import (
    KolmogorovSmirnovDriftDetector,
)
from backend.app.modules.distribution_shift.detectors.wasserstein_drift import (
    WassersteinDistanceDriftDetector,
)

# =============================================================================
# Database Model & Repository
# =============================================================================


class DriftReportModel(Base):
    """Relational model storing evaluated distribution drift reports."""

    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    target_feature: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    is_drifting: Mapped[bool] = mapped_column(Boolean, nullable=False)
    overall_status: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    production_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class SqliteDriftRepository:
    """SQLite repository for storing and querying distribution shift reports."""

    def __init__(self, session: Session) -> None:
        self._session = session
        # Ensure drift_reports table exists
        DriftReportModel.__table__.create(bind=self._session.get_bind(), checkfirst=True)

    def save_report(self, report: DriftReport, target_feature: str) -> DriftReportModel:
        """Persists an evaluated drift report to the SQLite database."""
        model = DriftReportModel(
            id=report.report_id,
            model_id=report.dataset_or_model_id or "unknown",
            target_feature=target_feature,
            timestamp_utc=report.timestamp_utc,
            is_drifting=report.is_drifting,
            overall_status=report.overall_status.value,
            reference_sample_count=report.reference_sample_count,
            production_sample_count=report.production_sample_count,
            report_json=report.model_dump(mode="json"),
        )
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return model

    def get_reports_by_model(self, model_id: str) -> list[DriftReport]:
        """Queries historical drift reports for a given model in descending timestamp order."""
        stmt = (
            select(DriftReportModel)
            .where(DriftReportModel.model_id == model_id)
            .order_by(DriftReportModel.timestamp_utc.desc())
        )
        rows = self._session.scalars(stmt).all()
        return [DriftReport.model_validate(row.report_json) for row in rows]


# =============================================================================
# Distribution Shift Orchestrator
# =============================================================================


class DistributionShiftOrchestrator:
    """Orchestrates multi-detector distribution drift evaluations."""

    @classmethod
    def evaluate_drift(
        cls,
        model_id: str,
        target_feature: str,
        reference_data: Sequence[float],
        production_data: Sequence[float],
        *,
        ks_alpha: float = 0.05,
        wasserstein_threshold: float = 0.10,
        min_samples: int = 30,
    ) -> DriftReport:
        """Executes KS and Wasserstein drift evaluations and aggregates them into a DriftReport."""
        ks_metric: DriftMetric = KolmogorovSmirnovDriftDetector.evaluate(
            reference_data=reference_data,
            production_data=production_data,
            alpha=ks_alpha,
            min_samples=min_samples,
        )
        w_metric: DriftMetric = WassersteinDistanceDriftDetector.evaluate(
            reference_data=reference_data,
            production_data=production_data,
            threshold=wasserstein_threshold,
            min_samples=min_samples,
        )

        metrics = [ks_metric, w_metric]
        is_drifting = any(m.is_drifting for m in metrics)

        # Epistemic aggregation of capability status
        if any(m.capability_status == CapabilityStatus.FAILED for m in metrics):
            overall_status = CapabilityStatus.FAILED
            status_reason = "One or more drift metrics failed execution."
        elif all(m.capability_status == CapabilityStatus.INCONCLUSIVE for m in metrics):
            overall_status = CapabilityStatus.INCONCLUSIVE
            status_reason = ks_metric.status_reason or w_metric.status_reason
        elif any(m.capability_status == CapabilityStatus.INCONCLUSIVE for m in metrics):
            overall_status = CapabilityStatus.COMPLETED
            status_reason = "Some metrics were inconclusive due to sample density constraints."
        else:
            overall_status = CapabilityStatus.COMPLETED
            status_reason = None

        return DriftReport(
            dataset_or_model_id=model_id,
            metrics=metrics,
            is_drifting=is_drifting,
            overall_status=overall_status,
            status_reason=status_reason,
            reference_sample_count=len(reference_data),
            production_sample_count=len(production_data),
            details={
                "target_feature": target_feature,
                "ks_alpha": ks_alpha,
                "wasserstein_threshold": wasserstein_threshold,
                "min_samples": min_samples,
            },
        )


# =============================================================================
# API Request / Response Schemas
# =============================================================================


class DriftEvaluateRequest(BaseModel):
    """Request payload for distribution drift evaluation."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        ...,
        min_length=1,
        description="UUID or unique identifier of the target model.",
    )
    target_feature: str = Field(
        ...,
        min_length=1,
        description="Name of the evaluated scalar feature (e.g., 'confidence_score', 'logit_norm').",
    )
    reference_data: list[float] = Field(
        ...,
        description="1D array-like sequence of baseline/reference samples.",
    )
    production_data: list[float] = Field(
        ...,
        description="1D array-like sequence of production/runtime samples.",
    )
    ks_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level alpha for the Kolmogorov-Smirnov test.",
    )
    wasserstein_threshold: float = Field(
        default=0.10,
        gt=0.0,
        description="Distance threshold for the Wasserstein metric.",
    )
    min_samples: int = Field(
        default=30,
        ge=1,
        description="Minimum sample density threshold per window.",
    )


# =============================================================================
# FastAPI Router Definition
# =============================================================================

router = APIRouter(prefix="/distribution-shift", tags=["Distribution Shift"])


@router.post(
    "/evaluate",
    response_model=DriftReport,
    status_code=status.HTTP_201_CREATED,
    summary="Evaluate distribution drift across reference and production samples",
)
def evaluate_distribution_shift(
    request: DriftEvaluateRequest,
    db: Session = Depends(get_db),
) -> DriftReport:
    """Evaluates covariate shift using Kolmogorov-Smirnov and Wasserstein distance detectors.

    Persists the resulting DriftReport to SQLite and returns the report.
    """
    try:
        report = DistributionShiftOrchestrator.evaluate_drift(
            model_id=request.model_id,
            target_feature=request.target_feature,
            reference_data=request.reference_data,
            production_data=request.production_data,
            ks_alpha=request.ks_alpha,
            wasserstein_threshold=request.wasserstein_threshold,
            min_samples=request.min_samples,
        )

        repo = SqliteDriftRepository(db)
        repo.save_report(report, target_feature=request.target_feature)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "DRIFT_EVALUATION_ERROR",
                "message": f"Failed to evaluate distribution shift: {exc}",
                "details": {"model_id": request.model_id},
            },
        ) from exc


@router.get(
    "/reports/{model_id}",
    response_model=list[DriftReport],
    status_code=status.HTTP_200_OK,
    summary="Retrieve historical distribution shift reports for a model",
)
def get_model_drift_reports(
    model_id: str,
    db: Session = Depends(get_db),
) -> list[DriftReport]:
    """Retrieves all historical drift assessment reports for the specified model ID."""
    if not model_id or not model_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_MODEL_ID",
                "message": "model_id parameter must be a non-empty string.",
                "details": {},
            },
        )

    repo = SqliteDriftRepository(db)
    return repo.get_reports_by_model(model_id.strip())
