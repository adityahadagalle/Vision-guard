"""API endpoints for dataset upload, inspection, and assessment triggering."""

import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import AssetSnapshotModel
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.session import get_db
from backend.app.domain.entities.enums import (
    IngestionStatus,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    FileTooLargeError,
    QuarantineError,
)
from backend.app.modules.data_integrity.ingestion.quarantine import QuarantineManager
from backend.app.schemas.assessment import (
    AssessmentResponse,
    CreateAssessmentRequest,
)
from backend.app.schemas.dataset import DatasetDetailResponse, DatasetResponse

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload dataset archive to quarantine",
)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="Compressed dataset archive (.zip)")],
    name: Annotated[str | None, Form(description="Optional friendly dataset name")] = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetResponse:
    """Streams uploaded dataset directly to quarantine and enqueues asynchronous ingestion."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "EMPTY_FILENAME",
                "message": "Uploaded file must have a filename.",
            },
        )

    # Use application IngestionConfig pointing to app data_dir
    ingestion_config = IngestionConfig(
        quarantine_dir=settings.data_dir / "quarantine",
        staging_dir=settings.data_dir / "staging",
        snapshots_dir=settings.data_dir / "snapshots",
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    quarantine_mgr = QuarantineManager(config=ingestion_config)

    dataset_id = f"ds_{uuid.uuid4().hex[:16]}"
    dataset_name = name.strip() if name and name.strip() else file.filename

    try:
        # Stream chunks directly from upload file into quarantine
        quarantine_rec = quarantine_mgr.quarantine_stream(
            stream=file.file,
            filename=file.filename,
        )
    except FileTooLargeError as err:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error_code": "FILE_TOO_LARGE",
                "message": f"Upload exceeds maximum allowable size of {settings.max_upload_mb} MB.",
                "details": {"max_bytes": err.details.get("limit_bytes")},
            },
        ) from err
    except QuarantineError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "QUARANTINE_REJECTED",
                "message": f"Upload failed quarantine inspection: {err}",
            },
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "UPLOAD_STREAM_FAILED",
                "message": f"An unexpected error occurred while staging upload: {err}",
            },
        ) from err

    # Create dataset record in database
    dataset_repo = SqliteDatasetRepository(db)
    dataset = dataset_repo.create_dataset(
        dataset_id=dataset_id,
        name=dataset_name,
        dataset_format="AUTODETECT",
        storage_path=str(quarantine_rec.quarantine_path),
        sha256=quarantine_rec.sha256,
        size_bytes=quarantine_rec.size_bytes,
        status=IngestionStatus.QUARANTINED,
        metadata={"original_filename": file.filename},
    )

    # Enqueue background ingestion task
    job_queue = SqliteJobQueue(db)
    job_id = job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": dataset_id,
            "quarantine_path": str(quarantine_rec.quarantine_path),
            "filename": file.filename,
            "sha256": quarantine_rec.sha256,
        },
    )

    return DatasetResponse(
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        format=dataset.format,
        status=dataset.status,
        sha256=dataset.sha256,
        size_bytes=dataset.size_bytes,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        job_id=job_id,
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetailResponse,
    summary="Retrieve dataset metadata and snapshot status",
)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> DatasetDetailResponse:
    """Retrieves dataset metadata and associated snapshot details."""
    dataset_repo = SqliteDatasetRepository(db)
    dataset = dataset_repo.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "DATASET_NOT_FOUND",
                "message": f"Dataset '{dataset_id}' does not exist.",
            },
        )

    # Check for associated snapshot
    stmt = (
        select(AssetSnapshotModel)
        .where(AssetSnapshotModel.dataset_id == dataset_id)
        .order_by(AssetSnapshotModel.created_at.desc())
        .limit(1)
    )
    snapshot = db.execute(stmt).scalar_one_or_none()

    return DatasetDetailResponse(
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        format=dataset.format,
        status=dataset.status,
        sha256=dataset.sha256,
        size_bytes=dataset.size_bytes,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        snapshot_file_count=snapshot.file_count if snapshot else None,
        snapshot_manifest_sha256=snapshot.manifest_sha256 if snapshot else None,
    )


_DEFAULT_ASSESSMENT_REQUEST = CreateAssessmentRequest()


@router.post(
    "/{dataset_id}/assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger an integrity assessment on a dataset snapshot",
)
def create_dataset_assessment(
    dataset_id: str,
    payload: CreateAssessmentRequest = _DEFAULT_ASSESSMENT_REQUEST,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AssessmentResponse:
    """Creates a new assessment for the dataset's validated snapshot and queues analysis."""
    dataset_repo = SqliteDatasetRepository(db)
    dataset = dataset_repo.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "DATASET_NOT_FOUND",
                "message": f"Dataset '{dataset_id}' does not exist.",
            },
        )

    # Resolve latest snapshot for dataset
    stmt = (
        select(AssetSnapshotModel)
        .where(AssetSnapshotModel.dataset_id == dataset_id)
        .order_by(AssetSnapshotModel.created_at.desc())
        .limit(1)
    )
    snapshot = db.execute(stmt).scalar_one_or_none()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "NO_SNAPSHOT_AVAILABLE",
                "message": (
                    f"Dataset '{dataset_id}' has no sealed snapshot (status is '{dataset.status}'). "
                    "Assessments can only be executed against approved, immutable snapshots."
                ),
            },
        )

    assessment_id = f"ass_{uuid.uuid4().hex[:16]}"
    assessment_repo = SqliteAssessmentRepository(db)

    config = {
        **payload.config,
        "detectors": payload.detectors,
    }

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name=payload.name,
        snapshot_id=snapshot.snapshot_id,
        config=config,
        dataset_id=dataset_id,
    )

    # Enqueue background assessment job
    job_queue = SqliteJobQueue(db)
    job_id = job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snapshot.snapshot_id,
            "dataset_id": dataset_id,
        },
    )

    created_ass = assessment_repo.get_assessment(assessment_id)
    return AssessmentResponse(
        assessment_id=created_ass["assessment_id"],
        name=created_ass["name"],
        dataset_id=created_ass["dataset_id"],
        snapshot_id=created_ass["snapshot_id"],
        status=created_ass["status"],
        status_reason=created_ass["status_reason"],
        config_digest=created_ass["config_digest"],
        started_at=created_ass["started_at"],
        completed_at=created_ass["completed_at"],
        runs=[],
        job_id=job_id,
    )
