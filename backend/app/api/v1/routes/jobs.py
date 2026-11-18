"""API endpoints for asynchronous job status queries."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.session import get_db
from backend.app.schemas.job import JobResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Retrieve execution status and result of a background job",
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobResponse:
    """Queries background job status, attempts, error message, and result payload."""
    queue = SqliteJobQueue(db)
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' does not exist.",
            },
        )

    return JobResponse(
        job_id=job["job_id"],
        task_type=job["task_type"],
        status=job["status"],
        attempts=job["attempts"],
        max_attempts=job["max_attempts"],
        error_message=job["error_message"],
        result=job["result"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
