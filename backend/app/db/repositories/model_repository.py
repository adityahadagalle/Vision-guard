"""SQLite repository adapter for model artifacts and immutable snapshots."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.app.db.models import AssetSnapshotModel, JobModel, ModelArtifactModel
from backend.app.db.repositories.job_repository import (
    InvalidJobStateTransitionError,
    JobLeaseExpiredError,
    JobNotClaimedError,
)
from backend.app.db.session import now_utc
from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType, IngestionStatus, JobStatus


class InvalidModelStateTransitionError(ValueError):
    """Raised when an illegal or stale model lifecycle state transition is attempted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = "INVALID_MODEL_STATE_TRANSITION"
        self.details = details or {}


VALID_MODEL_TRANSITIONS: dict[str, set[str]] = {
    IngestionStatus.RECEIVED: {IngestionStatus.INSPECTING, IngestionStatus.REJECTED},
    IngestionStatus.INSPECTING: {IngestionStatus.ACCEPTED, IngestionStatus.REJECTED},
    IngestionStatus.ACCEPTED: set(),  # Terminal state
    IngestionStatus.REJECTED: set(),  # Terminal state
}


class SqliteModelRepository:
    """Repository managing model artifact records and content-addressed asset snapshots in SQLite."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_model(
        self,
        model_id: str,
        name: str,
        storage_path: str,
        sha256: str,
        size_bytes: int,
        model_format: str = "pytorch",
        status: IngestionStatus = IngestionStatus.RECEIVED,
        metadata: dict[str, Any] | None = None,
        auto_commit: bool = True,
    ) -> ModelArtifactModel:
        """Persists a new model artifact record."""
        now = now_utc()
        model = ModelArtifactModel(
            model_id=model_id,
            name=name,
            format=model_format,
            status=status,
            storage_path=storage_path,
            sha256=sha256,
            size_bytes=size_bytes,
            metadata_json=json.dumps(metadata or {}, sort_keys=True),
            created_at=now,
            updated_at=now,
        )
        self._session.add(model)
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()
        return model

    def reconcile_stranded_received_models(
        self,
        job_queue: Any | None = None,
    ) -> int:
        """Startup reconciliation for stranded RECEIVED or INSPECTING models lacking active ingestion jobs.

        Finds models stuck in RECEIVED or interrupted in INSPECTING status after crashes,
        and safely re-enqueues an ingestion job if the quarantined file exists,
        or marks the model REJECTED if the file is missing.
        """
        stmt = select(ModelArtifactModel).where(
            ModelArtifactModel.status.in_([IngestionStatus.RECEIVED, IngestionStatus.INSPECTING])
        )
        stranded_models = list(self._session.execute(stmt).scalars().all())
        reconciled = 0

        for model in stranded_models:
            # Check for existing QUEUED or RUNNING ingestion job
            job_stmt = (
                select(JobModel)
                .where(JobModel.task_type == "model_ingestion")
                .where(JobModel.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            )
            active_jobs = list(self._session.execute(job_stmt).scalars().all())
            has_active_job = any(model.model_id in (j.payload_json or "") for j in active_jobs)
            if has_active_job:
                continue

            storage_file = Path(model.storage_path) if model.storage_path else None
            if storage_file and storage_file.is_file():
                if job_queue is not None:
                    meta = json.loads(model.metadata_json) if model.metadata_json else {}
                    filename = meta.get("original_filename", storage_file.name)
                    job_queue.submit_job(
                        task_type="model_ingestion",
                        payload={
                            "model_id": model.model_id,
                            "quarantine_path": str(storage_file),
                            "filename": filename,
                            "sha256": model.sha256,
                            "size_bytes": model.size_bytes,
                        },
                    )
                    reconciled += 1
            else:
                model.status = str(IngestionStatus.REJECTED)
                meta = json.loads(model.metadata_json) if model.metadata_json else {}
                meta["rejection_reason"] = "Quarantine artifact missing during startup recovery."
                model.metadata_json = json.dumps(meta, sort_keys=True)
                model.updated_at = now_utc()
                reconciled += 1

        if reconciled > 0:
            self._session.commit()
        return reconciled

    def get_model(self, model_id: str) -> ModelArtifactModel | None:
        """Retrieves a model artifact record by its identifier."""
        stmt = select(ModelArtifactModel).where(ModelArtifactModel.model_id == model_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_model_by_storage_path(self, storage_path: str) -> ModelArtifactModel | None:
        """Retrieves an authoritative model artifact record by its filesystem storage path."""
        stmt = select(ModelArtifactModel).where(ModelArtifactModel.storage_path == storage_path)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_model_by_sha256(self, sha256: str) -> ModelArtifactModel | None:
        """Retrieves an authoritative model artifact record by its content SHA-256 digest."""
        stmt = select(ModelArtifactModel).where(ModelArtifactModel.sha256 == sha256)
        return self._session.execute(stmt).scalar_one_or_none()

    def update_model_status(
        self,
        model_id: str,
        status: IngestionStatus,
        expected_current_status: IngestionStatus | None = None,
        auto_commit: bool = True,
    ) -> None:
        """Conditionally updates model ingestion status with state machine invariant enforcement."""
        model = self.get_model(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' not found.")

        current = model.status
        target = str(status)

        if expected_current_status is not None and current != str(expected_current_status):
            raise InvalidModelStateTransitionError(
                f"Model '{model_id}' status is '{current}', expected '{expected_current_status}'.",
                details={
                    "model_id": model_id,
                    "current": current,
                    "expected": str(expected_current_status),
                },
            )

        valid_next = VALID_MODEL_TRANSITIONS.get(current, set())
        if target not in valid_next:
            raise InvalidModelStateTransitionError(
                f"Illegal model state transition for '{model_id}': cannot transition from '{current}' to '{target}'.",
                details={"model_id": model_id, "current": current, "attempted": target},
            )

        model.status = target
        model.updated_at = now_utc()
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()

    def transition_model_to_inspecting(
        self,
        model_id: str,
        job_id: str,
        worker_id: str,
        lease_token: str,
        lease_generation: int,
    ) -> None:
        """Atomically validates worker lease fencing and transitions model from RECEIVED to INSPECTING.

        Fails closed if the worker does not hold a valid, active lease with matching generation.
        Idempotently permits retried execution if model is already INSPECTING under this job.
        """
        if job_id is None or worker_id is None or lease_token is None or lease_generation is None:
            raise JobNotClaimedError(
                "Model state transition requires all lease credentials: job_id, worker_id, lease_token, and lease_generation."
            )

        now = now_utc()
        stmt = text("""
            UPDATE jobs
            SET updated_at = :now
            WHERE job_id = :job_id
              AND locked_by = :worker_id
              AND lease_token = :lease_token
              AND lease_generation = :lease_generation
              AND status = 'RUNNING'
              AND lease_expires_at >= :now
              AND json_valid(payload_json) = 1
              AND json_extract(payload_json, '$.model_id') = :model_id
            RETURNING job_id;
        """)
        res = self._session.execute(
            stmt,
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "lease_token": lease_token,
                "lease_generation": lease_generation,
                "model_id": model_id,
                "now": now,
            },
        ).fetchone()

        if not res:
            self._session.rollback()
            self._raise_lease_fencing_error(
                job_id, worker_id, lease_token, lease_generation, model_id, now
            )

        model = self.get_model(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' not found.")

        if model.status == str(IngestionStatus.INSPECTING):
            # Already inspecting (e.g. retried worker claim for same active job)
            return

        if model.status != str(IngestionStatus.RECEIVED):
            raise InvalidModelStateTransitionError(
                f"Cannot transition model '{model_id}' to INSPECTING: current status is '{model.status}'.",
                details={
                    "model_id": model_id,
                    "current": model.status,
                    "target": str(IngestionStatus.INSPECTING),
                },
            )

        model.status = str(IngestionStatus.INSPECTING)
        model.updated_at = now
        self._session.commit()

    def publish_model_ingestion_success(
        self,
        model_id: str,
        snapshot: AssetSnapshot,
        storage_path: str,
        sha256: str,
        size_bytes: int,
        job_id: str,
        worker_id: str,
        lease_token: str,
        lease_generation: int,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Atomically seals model snapshot, transitions model to ACCEPTED, and completes the background job.

        Enforces atomic worker lease fencing: if the worker's lease expired or generation changed,
        the entire transaction rolls back, leaving database state completely untouched.
        """
        if job_id is None or worker_id is None or lease_token is None or lease_generation is None:
            raise JobNotClaimedError(
                "Publication requires all lease credentials: job_id, worker_id, lease_token, and lease_generation."
            )

        now = now_utc()
        if self._session.in_transaction():
            self._session.commit()

        with self._session.begin():
            # 1. Atomic lease fence check & job completion in single UPDATE
            lease_stmt = text("""
                UPDATE jobs
                SET status = 'COMPLETED',
                    result_json = :result_json,
                    lease_token = NULL,
                    updated_at = :now
                WHERE job_id = :job_id
                  AND locked_by = :worker_id
                  AND lease_token = :lease_token
                  AND lease_generation = :lease_generation
                  AND status = 'RUNNING'
                  AND lease_expires_at >= :now
                  AND json_valid(payload_json) = 1
                  AND json_extract(payload_json, '$.model_id') = :model_id
                RETURNING job_id;
            """)
            res = self._session.execute(
                lease_stmt,
                {
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "lease_generation": lease_generation,
                    "model_id": model_id,
                    "result_json": json.dumps(result or {}, sort_keys=True),
                    "now": now,
                },
            ).fetchone()

            if not res:
                self._raise_lease_fencing_error(
                    job_id, worker_id, lease_token, lease_generation, model_id, now
                )

            # 2. Model state verification and transition to ACCEPTED
            model = self.get_model(model_id)
            if model is None:
                raise KeyError(f"Model '{model_id}' not found.")

            if model.status != str(IngestionStatus.INSPECTING):
                raise InvalidModelStateTransitionError(
                    f"Cannot complete model ingestion for '{model_id}': expected state INSPECTING, found '{model.status}'.",
                    details={
                        "model_id": model_id,
                        "current": model.status,
                        "expected": str(IngestionStatus.INSPECTING),
                    },
                )

            model.status = str(IngestionStatus.ACCEPTED)
            model.storage_path = storage_path
            model.sha256 = sha256
            model.size_bytes = size_bytes
            model.updated_at = now

            # 3. Snapshot record insertion (idempotent within transaction)
            existing_snap = self.get_snapshot(snapshot.snapshot_id)
            if not existing_snap:
                snap_model = AssetSnapshotModel(
                    snapshot_id=snapshot.snapshot_id,
                    dataset_id=None,
                    model_id=model_id,
                    asset_type=snapshot.asset_type,
                    manifest_sha256=snapshot.manifest_sha256,
                    storage_path=snapshot.metadata.get("snapshot_path", storage_path),
                    file_count=snapshot.file_count,
                    total_size_bytes=snapshot.total_size_bytes,
                    schema_version=snapshot.schema_version,
                    metadata_json=json.dumps(snapshot.metadata, sort_keys=True),
                    created_at=now,
                )
                self._session.add(snap_model)
                self._session.flush()

    def publish_model_ingestion_failure(
        self,
        model_id: str,
        error_message: str,
        job_id: str,
        worker_id: str,
        lease_token: str,
        lease_generation: int,
    ) -> None:
        """Atomically marks job FAILED and transitions model to REJECTED under lease fencing.

        If the worker's lease has expired, the operation fails closed without modifying newer model state.
        """
        if job_id is None or worker_id is None or lease_token is None or lease_generation is None:
            raise JobNotClaimedError(
                "Failure publication requires all lease credentials: job_id, worker_id, lease_token, and lease_generation."
            )

        now = now_utc()
        if self._session.in_transaction():
            self._session.commit()

        with self._session.begin():
            lease_stmt = text("""
                UPDATE jobs
                SET status = 'FAILED',
                    error_message = :error_message,
                    lease_token = NULL,
                    updated_at = :now
                WHERE job_id = :job_id
                  AND locked_by = :worker_id
                  AND lease_token = :lease_token
                  AND lease_generation = :lease_generation
                  AND status = 'RUNNING'
                  AND lease_expires_at >= :now
                  AND json_valid(payload_json) = 1
                  AND json_extract(payload_json, '$.model_id') = :model_id
                RETURNING job_id;
            """)
            res = self._session.execute(
                lease_stmt,
                {
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "lease_generation": lease_generation,
                    "model_id": model_id,
                    "error_message": error_message,
                    "now": now,
                },
            ).fetchone()

            if not res:
                self._raise_lease_fencing_error(
                    job_id, worker_id, lease_token, lease_generation, model_id, now
                )

            model = self.get_model(model_id)
            if model and model.status not in (
                str(IngestionStatus.ACCEPTED),
                str(IngestionStatus.REJECTED),
            ):
                model.status = str(IngestionStatus.REJECTED)
                meta = json.loads(model.metadata_json) if model.metadata_json else {}
                meta["rejection_reason"] = error_message
                model.metadata_json = json.dumps(meta, sort_keys=True)
                model.updated_at = now

    def _raise_lease_fencing_error(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        lease_generation: int,
        model_id: str,
        now: datetime,
    ) -> None:
        """Inspects database state to raise structured exception explaining lease fencing rejection."""
        stmt = select(JobModel).where(JobModel.job_id == job_id)
        job_model = self._session.execute(stmt).scalar_one_or_none()
        if job_model is None:
            raise KeyError(f"Job '{job_id}' not found.")

        if job_model.status != JobStatus.RUNNING.value and job_model.status != JobStatus.RUNNING:
            raise InvalidJobStateTransitionError(
                f"Cannot operate on job '{job_id}' in non-running state '{job_model.status}'."
            )

        if job_model.locked_by != worker_id or job_model.lease_token != lease_token:
            raise JobNotClaimedError(
                f"Operation rejected: worker '{worker_id}' with token '{lease_token}' does not hold active "
                f"lease for job '{job_id}' (held by '{job_model.locked_by}' with token '{job_model.lease_token}')."
            )

        if job_model.lease_generation != lease_generation:
            raise JobNotClaimedError(
                f"Operation rejected: lease generation {lease_generation} does not match active "
                f"generation {job_model.lease_generation} for job '{job_id}'."
            )

        if job_model.lease_expires_at is not None and job_model.lease_expires_at < now:
            raise JobLeaseExpiredError(
                f"Operation rejected: lease for job '{job_id}' expired at {job_model.lease_expires_at} (current time: {now})."
            )

        payload = json.loads(job_model.payload_json) if job_model.payload_json else {}
        if payload.get("model_id") != model_id:
            raise JobNotClaimedError(
                f"Operation rejected: job '{job_id}' payload model_id '{payload.get('model_id')}' does not match '{model_id}'."
            )

        raise JobNotClaimedError(f"Lease verification failed for job '{job_id}'.")

    def list_models(
        self,
        offset: int = 0,
        limit: int = 50,
        status_filter: IngestionStatus | None = None,
    ) -> tuple[list[ModelArtifactModel], int]:
        """Returns paginated model artifacts and total count matching optional status filter."""
        stmt = select(ModelArtifactModel)
        count_stmt = select(func.count(ModelArtifactModel.model_id))
        if status_filter is not None:
            filter_val = str(status_filter)
            stmt = stmt.where(ModelArtifactModel.status == filter_val)
            count_stmt = count_stmt.where(ModelArtifactModel.status == filter_val)

        total = self._session.execute(count_stmt).scalar() or 0
        stmt = stmt.order_by(ModelArtifactModel.created_at.desc()).offset(offset).limit(limit)
        models = list(self._session.execute(stmt).scalars().all())
        return models, total

    def create_snapshot(
        self,
        snapshot_id: str,
        model_id: str,
        manifest_sha256: str,
        storage_path: str,
        file_count: int,
        total_size_bytes: int,
        asset_type: AssetType = AssetType.MODEL_CHECKPOINT,
        schema_version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
        auto_commit: bool = True,
    ) -> AssetSnapshot:
        """Persists an immutable asset snapshot record for a model or returns existing."""
        existing = self.get_snapshot(snapshot_id)
        if existing is not None:
            return existing

        now = now_utc()
        meta = metadata or {}
        model = AssetSnapshotModel(
            snapshot_id=snapshot_id,
            dataset_id=None,
            model_id=model_id,
            asset_type=asset_type,
            manifest_sha256=manifest_sha256,
            storage_path=storage_path,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            schema_version=schema_version,
            metadata_json=json.dumps(meta, sort_keys=True),
            created_at=now,
        )
        self._session.add(model)
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()

        return AssetSnapshot(
            snapshot_id=snapshot_id,
            asset_id=model_id,
            asset_type=asset_type,
            manifest_sha256=manifest_sha256,
            created_at=now,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            schema_version=schema_version,
            metadata=meta,
        )

    def get_snapshot(self, snapshot_id: str) -> AssetSnapshot | None:
        """Retrieves an immutable asset snapshot record by its identifier."""
        stmt = select(AssetSnapshotModel).where(AssetSnapshotModel.snapshot_id == snapshot_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is None:
            return None

        meta = json.loads(model.metadata_json) if model.metadata_json else {}
        meta["storage_path"] = model.storage_path
        asset_id = model.model_id or model.dataset_id or ""

        return AssetSnapshot(
            snapshot_id=model.snapshot_id,
            asset_id=asset_id,
            asset_type=AssetType(model.asset_type),
            manifest_sha256=model.manifest_sha256,
            created_at=model.created_at,
            file_count=model.file_count,
            total_size_bytes=model.total_size_bytes,
            schema_version=model.schema_version,
            metadata=meta,
        )

    def get_authoritative_manifest_sha256(self, snapshot_id: str) -> str | None:
        """Retrieves authoritative manifest SHA-256 for a snapshot."""
        stmt = select(AssetSnapshotModel.manifest_sha256).where(
            AssetSnapshotModel.snapshot_id == snapshot_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_snapshot_storage_path(self, snapshot_id: str) -> str | None:
        """Retrieves filesystem storage path for a snapshot."""
        stmt = select(AssetSnapshotModel.storage_path).where(
            AssetSnapshotModel.snapshot_id == snapshot_id
        )
        return self._session.execute(stmt).scalar_one_or_none()
