"""API endpoints for querying assessments and paginated findings."""

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.session import get_db
from backend.app.domain.entities.enums import FindingCategory, SeverityLevel
from backend.app.schemas.assessment import (
    AssessmentResponse,
    DetectorRunSummary,
    FindingAffectedAssetSchema,
    FindingResponse,
    PaginatedFindingsResponse,
)

router = APIRouter(prefix="/assessments", tags=["Assessments"])


@router.get(
    "/{assessment_id}",
    response_model=AssessmentResponse,
    summary="Retrieve assessment status and detector execution summaries",
)
def get_assessment(
    assessment_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    """Retrieves assessment metadata, current lifecycle state, and detector runs."""
    repo = SqliteAssessmentRepository(db)
    ass = repo.get_assessment(assessment_id)
    if not ass:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ASSESSMENT_NOT_FOUND",
                "message": f"Assessment '{assessment_id}' does not exist.",
            },
        )

    runs = [
        DetectorRunSummary(
            run_id=r["run_id"],
            detector_id=r["detector_id"],
            detector_version=r["detector_version"],
            status=r["status"],
            status_reason=r["status_reason"],
            duration_ms=r["duration_ms"],
            started_at=r["started_at"],
            completed_at=r["completed_at"],
        )
        for r in ass.get("runs", [])
    ]

    return AssessmentResponse(
        assessment_id=ass["assessment_id"],
        name=ass["name"],
        dataset_id=ass.get("dataset_id"),
        model_id=ass.get("model_id"),
        snapshot_id=ass["snapshot_id"],
        status=ass["status"],
        status_reason=ass["status_reason"],
        config_digest=ass["config_digest"],
        started_at=ass["started_at"],
        completed_at=ass["completed_at"],
        runs=runs,
    )


@router.get(
    "/{assessment_id}/findings",
    response_model=PaginatedFindingsResponse,
    summary="Retrieve paginated findings produced by an assessment",
)
def get_assessment_findings(
    assessment_id: str,
    page: Annotated[int, Query(ge=1, description="Page number, 1-indexed")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    category: Annotated[
        FindingCategory | None, Query(description="Filter by finding category")
    ] = None,
    severity: Annotated[SeverityLevel | None, Query(description="Filter by severity level")] = None,
    db: Session = Depends(get_db),
) -> PaginatedFindingsResponse:
    """Retrieves paginated findings for an assessment with optional category and severity filtering."""
    repo = SqliteAssessmentRepository(db)
    ass = repo.get_assessment(assessment_id)
    if not ass:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ASSESSMENT_NOT_FOUND",
                "message": f"Assessment '{assessment_id}' does not exist.",
            },
        )

    offset = (page - 1) * page_size
    findings, total_items = repo.get_findings_paginated(
        assessment_id=assessment_id,
        offset=offset,
        limit=page_size,
        category=category.value if category else None,
        severity=severity.value if severity else None,
    )

    items = [
        FindingResponse(
            finding_id=f.finding_id,
            assessment_id=f.assessment_id,
            detector_id=f.detector_id,
            detector_version=f.detector_version,
            category=f.category.value,
            severity=f.severity.value,
            title=f.title,
            description=f.description,
            anomaly_score=f.anomaly_score,
            confidence=f.confidence,
            suggested_disposition=f.suggested_disposition.value,
            status=f.status.value,
            affected_assets=[
                FindingAffectedAssetSchema(
                    asset_id=a.asset_id,
                    sample_path=a.sample_path,
                    sha256=a.sha256,
                )
                for a in f.affected_assets
            ],
            evidence_ids=f.evidence_ids,
            created_at=f.created_at,
        )
        for f in findings
    ]

    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 0

    return PaginatedFindingsResponse(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        items=items,
    )
