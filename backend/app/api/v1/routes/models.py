"""API endpoints for secure model artifact upload, metadata queries, and assessment initiation.

Enforces cross-process atomic upload quotas, concurrency bounds, safe streaming to quarantine,
conditional lifecycle transitions, and zero unpickling/deserialization within API worker processes.
"""

import hashlib
import math
import time
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import AssessmentModel, AssetSnapshotModel
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import (
    SqliteModelRepository,
)
from backend.app.db.repositories.upload_reservation_repository import (
    DiskSpaceExhaustedError,
    QuarantineQuotaExceededError,
    ReservationExpiredError,
    SqliteUploadReservationRepository,
    UploadConcurrencyLimitExceededError,
)
from backend.app.db.session import get_db
from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.model_integrity.artifact import validate_model_filename
from backend.app.modules.model_integrity.exceptions import (
    InvalidFilenameError,
    InvalidModelExtensionError,
    PathTraversalError,
)
from backend.app.schemas.assessment import AssessmentResponse
from backend.app.schemas.model import (
    CreateModelAssessmentRequest,
    ModelDetailResponse,
    ModelListResponse,
    ModelUploadResponse,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.post(
    "",
    response_model=ModelUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload candidate PyTorch model checkpoint to quarantine",
)
async def upload_model(
    request: Request,
    file: Annotated[UploadFile, File(description="PyTorch checkpoint artifact (.pt or .pth)")],
    name: Annotated[str | None, Form(description="Optional friendly model name")] = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ModelUploadResponse:
    """Streams candidate model checkpoint directly into quarantine with durable quota reservation."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "EMPTY_FILENAME",
                "message": "Uploaded file must have a filename.",
            },
        )

    # 1. Filename & Path Traversal Validation
    try:
        safe_filename = validate_model_filename(file.filename)
    except (InvalidModelExtensionError, InvalidFilenameError, PathTraversalError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": getattr(err, "code", "INVALID_FILENAME"),
                "message": str(err),
                "details": getattr(err, "details", {}),
            },
        ) from err

    # 2. Acquire Atomic Quota & Concurrency Reservation
    client_ip = request.client.host if request.client else None
    quarantine_base_dir = settings.data_dir / "quarantine" / "models"
    quarantine_base_dir.mkdir(parents=True, exist_ok=True)

    reservation_repo = SqliteUploadReservationRepository(db)
    try:
        reservation = reservation_repo.acquire_reservation(
            asset_type="model",
            requested_bytes=settings.max_model_upload_bytes,
            quarantine_dir=quarantine_base_dir,
            max_concurrent_uploads=settings.max_concurrent_uploads,
            global_quarantine_quota_bytes=settings.global_quarantine_quota_bytes,
            timeout_seconds=settings.upload_reservation_timeout_seconds,
            client_ip=client_ip,
        )
    except UploadConcurrencyLimitExceededError as err:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": "CONCURRENCY_LIMIT_EXCEEDED",
                "message": str(err),
            },
        ) from err
    except (QuarantineQuotaExceededError, DiskSpaceExhaustedError) as err:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail={
                "error_code": getattr(err, "code", "QUOTA_EXCEEDED"),
                "message": str(err),
            },
        ) from err

    res_id = reservation.reservation_id
    model_id = f"model_{uuid.uuid4().hex[:16]}"
    upload_target_dir = quarantine_base_dir / f"upload_{model_id}"
    upload_target_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_target_dir / safe_filename

    reservation_repo.update_reservation_path(res_id, str(target_path))

    # 3. Stream Upload into Quarantine & Hash On-the-Fly
    hasher = hashlib.sha256()
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB
    last_heartbeat_time = time.monotonic()
    heartbeat_interval_seconds = 60.0
    bytes_since_renewal = 0
    bytes_renewal_threshold = 32 * 1024 * 1024

    try:
        with open(target_path, "wb") as out_f:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)
                bytes_since_renewal += len(chunk)
                if total_bytes > settings.max_model_upload_bytes:
                    out_f.close()
                    reservation_repo.release_reservation(
                        res_id,
                        reason="FILE_TOO_LARGE",
                        cleanup_file=True,
                        quarantine_dir=quarantine_base_dir,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "error_code": "MODEL_FILE_TOO_LARGE",
                            "message": f"Upload exceeds maximum allowable model size of {settings.max_model_upload_bytes} bytes (500 MB).",
                            "details": {
                                "max_bytes": settings.max_model_upload_bytes,
                                "received_bytes": total_bytes,
                            },
                        },
                    )

                # Periodic lease heartbeat renewal during streaming
                now_mono = time.monotonic()
                if (now_mono - last_heartbeat_time >= heartbeat_interval_seconds) or (
                    bytes_since_renewal >= bytes_renewal_threshold
                ):
                    try:
                        reservation_repo.renew_reservation_lease(
                            res_id, extension_seconds=settings.upload_reservation_timeout_seconds
                        )
                        last_heartbeat_time = now_mono
                        bytes_since_renewal = 0
                    except ReservationExpiredError as exp_err:
                        out_f.close()
                        reservation_repo.release_reservation(
                            res_id,
                            reason="LEASE_EXPIRED_MID_STREAM",
                            cleanup_file=True,
                            quarantine_dir=quarantine_base_dir,
                        )
                        raise HTTPException(
                            status_code=status.HTTP_408_REQUEST_TIMEOUT,
                            detail={
                                "error_code": "RESERVATION_LEASE_LOST",
                                "message": "Upload reservation lease expired during streaming transfer.",
                                "details": exp_err.details,
                            },
                        ) from exp_err

                hasher.update(chunk)
                out_f.write(chunk)

        if total_bytes == 0:
            reservation_repo.release_reservation(
                res_id,
                reason="EMPTY_FILE",
                cleanup_file=True,
                quarantine_dir=quarantine_base_dir,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "EMPTY_MODEL_FILE",
                    "message": f"Uploaded model file '{safe_filename}' contains 0 bytes.",
                },
            )

        file_sha256 = hasher.hexdigest()

        # 4. Atomic Persistence: Complete Reservation + Persist Model + Enqueue Job
        model_name = name.strip() if name and name.strip() else safe_filename
        model_repo = SqliteModelRepository(db)
        job_queue = SqliteJobQueue(db)

        if db.in_transaction():
            db.commit()

        with db.begin():
            reservation_repo.complete_reservation(
                res_id,
                actual_bytes=total_bytes,
                quarantine_path=str(target_path),
                auto_commit=False,
            )
            created_model = model_repo.create_model(
                model_id=model_id,
                name=model_name,
                storage_path=str(target_path),
                sha256=file_sha256,
                size_bytes=total_bytes,
                model_format="pytorch",
                status=IngestionStatus.RECEIVED,
                metadata={"original_filename": safe_filename, "reservation_id": res_id},
                auto_commit=False,
            )
            job_id = job_queue.submit_job(
                task_type="model_ingestion",
                payload={
                    "model_id": model_id,
                    "quarantine_path": str(target_path),
                    "filename": safe_filename,
                    "sha256": file_sha256,
                    "size_bytes": total_bytes,
                },
                auto_commit=False,
            )

        return ModelUploadResponse(
            model_id=created_model.model_id,
            name=created_model.name,
            format=created_model.format,
            status=created_model.status,
            sha256=created_model.sha256,
            size_bytes=created_model.size_bytes,
            created_at=created_model.created_at,
            updated_at=created_model.updated_at,
            metadata={"original_filename": safe_filename, "reservation_id": res_id},
            snapshot_id=None,
            job_id=job_id,
        )

    except HTTPException:
        raise
    except ReservationExpiredError as err:
        reservation_repo.release_reservation(
            res_id,
            reason="RESERVATION_EXPIRED",
            cleanup_file=True,
            quarantine_dir=quarantine_base_dir,
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "error_code": "RESERVATION_EXPIRED",
                "message": str(err),
                "details": err.details,
            },
        ) from err
    except Exception as err:
        reservation_repo.release_reservation(
            res_id,
            reason="STREAM_ERROR",
            cleanup_file=True,
            quarantine_dir=quarantine_base_dir,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "UPLOAD_STREAMING_FAILED",
                "message": f"Failed to stream model upload to quarantine: {err}",
            },
        ) from err


@router.get(
    "/{model_id}",
    response_model=ModelDetailResponse,
    summary="Retrieve model artifact metadata and snapshot status",
)
def get_model_details(
    model_id: str,
    db: Session = Depends(get_db),
) -> ModelDetailResponse:
    """Retrieves full model metadata, snapshot integrity reference, and assessment count."""
    model_repo = SqliteModelRepository(db)
    model = model_repo.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "MODEL_NOT_FOUND",
                "message": f"Model '{model_id}' does not exist.",
            },
        )

    # Resolve latest snapshot if ingested
    stmt = (
        select(AssetSnapshotModel)
        .where(AssetSnapshotModel.model_id == model_id)
        .order_by(AssetSnapshotModel.created_at.desc())
        .limit(1)
    )
    snapshot = db.execute(stmt).scalar_one_or_none()

    # Count assessments associated with this model
    count_stmt = select(func.count(AssessmentModel.assessment_id)).where(
        AssessmentModel.model_id == model_id
    )
    assessments_count = db.execute(count_stmt).scalar() or 0

    return ModelDetailResponse(
        model_id=model.model_id,
        name=model.name,
        format=model.format,
        status=model.status,
        sha256=model.sha256,
        size_bytes=model.size_bytes,
        created_at=model.created_at,
        updated_at=model.updated_at,
        metadata={},
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        snapshot_manifest_sha256=snapshot.manifest_sha256 if snapshot else None,
        snapshot_file_count=snapshot.file_count if snapshot else None,
        assessments_count=assessments_count,
        job_id=None,
    )


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List registered model artifacts with pagination and status filtering",
)
def list_models(
    page: Annotated[int, Query(ge=1, description="Page number, 1-indexed")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    status_filter: Annotated[
        IngestionStatus | None,
        Query(alias="status", description="Filter by ingestion lifecycle state"),
    ] = None,
    db: Session = Depends(get_db),
) -> ModelListResponse:
    """Returns paginated list of candidate and accepted models with optional status filter."""
    model_repo = SqliteModelRepository(db)
    offset = (page - 1) * page_size
    models, total_count = model_repo.list_models(
        offset=offset, limit=page_size, status_filter=status_filter
    )
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0

    items = [
        ModelUploadResponse(
            model_id=m.model_id,
            name=m.name,
            format=m.format,
            status=m.status,
            sha256=m.sha256,
            size_bytes=m.size_bytes,
            created_at=m.created_at,
            updated_at=m.updated_at,
            metadata={},
            snapshot_id=m.snapshots[0].snapshot_id if m.snapshots else None,
            job_id=None,
        )
        for m in models
    ]

    return ModelListResponse(
        page=page,
        page_size=page_size,
        total_items=total_count,
        total_pages=total_pages,
        items=items,
    )


@router.post(
    "/{model_id}/assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger model integrity assessment on an approved model snapshot",
)
def create_model_assessment(
    model_id: str,
    payload: CreateModelAssessmentRequest | None = None,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    """Enqueues an integrity assessment for an approved model artifact snapshot."""
    actual_payload = payload or CreateModelAssessmentRequest()
    model_repo = SqliteModelRepository(db)
    model = model_repo.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "MODEL_NOT_FOUND",
                "message": f"Model '{model_id}' does not exist.",
            },
        )

    if model.status != IngestionStatus.ACCEPTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "MODEL_NOT_ACCEPTED",
                "message": (
                    f"Model '{model_id}' is in lifecycle state '{model.status}'. "
                    "Assessments may only be scheduled for models with ACCEPTED status."
                ),
            },
        )

    stmt = (
        select(AssetSnapshotModel)
        .where(AssetSnapshotModel.model_id == model_id)
        .order_by(AssetSnapshotModel.created_at.desc())
        .limit(1)
    )
    snapshot = db.execute(stmt).scalar_one_or_none()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "NO_SNAPSHOT_AVAILABLE",
                "message": f"Model '{model_id}' has no sealed snapshot to evaluate.",
            },
        )

    assessment_id = f"ass_{uuid.uuid4().hex[:16]}"
    assessment_repo = SqliteAssessmentRepository(db)
    job_queue = SqliteJobQueue(db)

    config = {
        **actual_payload.config,
        "detectors": actual_payload.detectors,
    }

    if db.in_transaction():
        db.commit()

    with db.begin():
        assessment_repo.create_assessment(
            assessment_id=assessment_id,
            name=actual_payload.name,
            snapshot_id=snapshot.snapshot_id,
            config=config,
            model_id=model_id,
            auto_commit=False,
        )

        job_id = job_queue.submit_job(
            task_type="assessment",
            payload={
                "assessment_id": assessment_id,
                "snapshot_id": snapshot.snapshot_id,
                "model_id": model_id,
            },
            auto_commit=False,
        )

    created_ass = assessment_repo.get_assessment(assessment_id)
    return AssessmentResponse(
        assessment_id=created_ass["assessment_id"],
        name=created_ass["name"],
        dataset_id=None,
        model_id=created_ass.get("model_id"),
        snapshot_id=created_ass["snapshot_id"],
        status=created_ass["status"],
        status_reason=created_ass["status_reason"],
        config_digest=created_ass["config_digest"],
        started_at=created_ass["started_at"],
        completed_at=created_ass["completed_at"],
        runs=[],
        job_id=job_id,
    )
