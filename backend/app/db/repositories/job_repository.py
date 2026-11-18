"""SQLite-backed task queue and job lifecycle orchestration.

Implements atomic job claiming, strict state machine validation, retry policies,
and stale execution recovery for offline-first asynchronous task execution.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.db.models import JobModel
from backend.app.db.session import now_utc
from backend.app.domain.entities.enums import JobStatus
from backend.app.domain.ports.job_queue import JobQueuePort


class InvalidJobStateTransitionError(ValueError):
    """Raised when an illegal job state transition is attempted."""

    pass


class JobNotClaimedError(RuntimeError):
    """Raised when an operation is attempted on a job by an unauthorized worker."""

    pass


class JobLeaseExpiredError(RuntimeError):
    """Raised when an operation is attempted on a job whose lease has expired."""

    pass


_VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
        JobStatus.QUEUED,  # Permitted only during recovery/retry
    },
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.FAILED: set(),  # Terminal state
    JobStatus.CANCELLED: set(),  # Terminal state
    JobStatus.TIMED_OUT: set(),  # Terminal state
}


class SqliteJobQueue(JobQueuePort):
    """SQLite-backed task queue with atomic leases and state machine invariants."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def submit_job(
        self,
        task_type: str,
        payload: dict[str, Any],
        job_id: str | None = None,
        max_attempts: int = 3,
        auto_commit: bool = True,
    ) -> str:
        """Enqueues a new background task."""
        jid = job_id or f"job_{uuid.uuid4().hex}"
        now = now_utc()
        model = JobModel(
            job_id=jid,
            task_type=task_type,
            payload_json=json.dumps(payload, sort_keys=True),
            status=JobStatus.QUEUED,
            attempts=0,
            max_attempts=max_attempts,
            locked_by=None,
            locked_at=None,
            result_json=None,
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self._session.add(model)
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()
        return jid

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieves details and status of a specific job."""
        stmt = select(JobModel).where(JobModel.job_id == job_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is None:
            return None

        payload = json.loads(model.payload_json) if model.payload_json else {}
        result = json.loads(model.result_json) if model.result_json else None

        return {
            "job_id": model.job_id,
            "task_type": model.task_type,
            "payload": payload,
            "status": model.status,
            "attempts": model.attempts,
            "max_attempts": model.max_attempts,
            "locked_by": model.locked_by,
            "locked_at": model.locked_at,
            "lease_token": model.lease_token,
            "lease_expires_at": model.lease_expires_at,
            "lease_generation": model.lease_generation,
            "result": result,
            "error_message": model.error_message,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    def claim_next_job(
        self,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Atomically leases the next available task for processing.

        Uses atomic UPDATE ... RETURNING with a subquery to prevent race conditions across workers.
        """
        now = now_utc()
        expires_at = now + timedelta(seconds=lease_duration_seconds)
        lease_token = f"tok_{uuid.uuid4().hex}"

        claim_sql = text("""
            UPDATE jobs
            SET status = 'RUNNING',
                locked_by = :worker_id,
                lease_token = :lease_token,
                lease_generation = lease_generation + 1,
                attempts = attempts + 1,
                locked_at = :now,
                lease_expires_at = :expires_at,
                updated_at = :now
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE status = 'QUEUED'
                ORDER BY created_at ASC
                LIMIT 1
            )
            RETURNING job_id;
        """)

        try:
            res = self._session.execute(
                claim_sql,
                {
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "now": now,
                    "expires_at": expires_at,
                },
            ).scalar_one_or_none()
            self._session.commit()
            if not res:
                return None
            return self.get_job(res)
        except Exception:
            self._session.rollback()
            raise

    def complete_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str | None = None,
        lease_generation: int | None = None,
        result: dict[str, Any] | None = None,
        auto_commit: bool = True,
    ) -> None:
        """Marks a claimed job as successfully completed after validating lease fencing."""
        model = self._session.get(JobModel, job_id)
        if model is None:
            raise KeyError(f"Job '{job_id}' not found.")

        current_status = JobStatus(model.status)
        self._validate_transition(current_status, JobStatus.COMPLETED)

        if model.locked_by != worker_id or (
            model.lease_token is not None and model.lease_token != lease_token
        ):
            raise JobNotClaimedError(
                f"Worker '{worker_id}' with token '{lease_token}' is not the current lease holder for job '{job_id}' "
                f"(held by '{model.locked_by}' with token '{model.lease_token}')."
            )

        if lease_generation is not None and model.lease_generation != lease_generation:
            raise JobNotClaimedError(
                f"Worker '{worker_id}' lease generation {lease_generation} does not match active "
                f"generation {model.lease_generation} for job '{job_id}'."
            )

        now = now_utc()
        if model.lease_expires_at is not None and model.lease_expires_at < now:
            raise JobLeaseExpiredError(
                f"Job '{job_id}' lease expired at {model.lease_expires_at} (current time: {now}). "
                "Operation rejected to prevent stale lease overwrite."
            )

        model.status = JobStatus.COMPLETED
        model.result_json = json.dumps(result or {}, sort_keys=True)
        model.lease_token = None
        model.updated_at = now
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str | None = None,
        lease_generation: int | None = None,
        error_message: str = "",
        auto_commit: bool = True,
    ) -> None:
        """Marks a claimed job as failed with structured error information after validating lease fencing."""
        model = self._session.get(JobModel, job_id)
        if model is None:
            raise KeyError(f"Job '{job_id}' not found.")

        current_status = JobStatus(model.status)
        self._validate_transition(current_status, JobStatus.FAILED)

        if model.locked_by != worker_id or (
            model.lease_token is not None and model.lease_token != lease_token
        ):
            raise JobNotClaimedError(
                f"Worker '{worker_id}' with token '{lease_token}' is not the current lease holder for job '{job_id}' "
                f"(held by '{model.locked_by}' with token '{model.lease_token}')."
            )

        if lease_generation is not None and model.lease_generation != lease_generation:
            raise JobNotClaimedError(
                f"Worker '{worker_id}' lease generation {lease_generation} does not match active "
                f"generation {model.lease_generation} for job '{job_id}'."
            )

        now = now_utc()
        if model.lease_expires_at is not None and model.lease_expires_at < now:
            raise JobLeaseExpiredError(
                f"Job '{job_id}' lease expired at {model.lease_expires_at} (current time: {now}). "
                "Operation rejected to prevent stale lease overwrite."
            )

        model.status = JobStatus.FAILED
        model.error_message = error_message
        model.lease_token = None
        model.updated_at = now
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        additional_seconds: int = 300,
        lease_generation: int | None = None,
    ) -> datetime:
        """Extends the lease duration for an active job holding a valid lease token."""
        model = self._session.get(JobModel, job_id)
        if model is None:
            raise KeyError(f"Job '{job_id}' not found.")

        if model.status != JobStatus.RUNNING.value and model.status != JobStatus.RUNNING:
            raise InvalidJobStateTransitionError(
                f"Cannot renew lease for job '{job_id}' in '{model.status}' state."
            )

        if model.locked_by != worker_id or (
            model.lease_token is not None and model.lease_token != lease_token
        ):
            raise JobNotClaimedError(
                f"Worker '{worker_id}' with token '{lease_token}' is not the current lease holder for job '{job_id}' "
                f"(held by '{model.locked_by}' with token '{model.lease_token}')."
            )

        if lease_generation is not None and model.lease_generation != lease_generation:
            raise JobNotClaimedError(
                f"Worker '{worker_id}' lease generation {lease_generation} does not match active "
                f"generation {model.lease_generation} for job '{job_id}'."
            )

        now = now_utc()
        if model.lease_expires_at is not None and model.lease_expires_at < now:
            raise JobLeaseExpiredError(
                f"Job '{job_id}' lease expired at {model.lease_expires_at} (current time: {now}). "
                "Renewal rejected."
            )

        new_expires_at = now + timedelta(seconds=additional_seconds)
        model.lease_expires_at = new_expires_at
        model.updated_at = now
        self._session.commit()
        return new_expires_at

    def cancel_job(self, job_id: str) -> None:
        """Cancels a job if it is not in a terminal state."""
        model = self._session.get(JobModel, job_id)
        if model is None:
            raise KeyError(f"Job '{job_id}' not found.")

        current_status = JobStatus(model.status)
        self._validate_transition(current_status, JobStatus.CANCELLED)

        now = now_utc()
        model.status = JobStatus.CANCELLED
        model.updated_at = now
        self._session.commit()

    def recover_stale_jobs(self, stale_threshold_seconds: int = 300) -> int:
        """Identifies and recovers jobs stuck in RUNNING state beyond the lease threshold.

        Policy:
        - If attempts < max_attempts: re-queues job for another attempt.
        - If attempts >= max_attempts: transitions job to TIMED_OUT.

        Returns:
            Count of recovered jobs.
        """
        now = now_utc()
        cutoff = now - timedelta(seconds=stale_threshold_seconds)

        stmt = select(JobModel).where(
            JobModel.status == JobStatus.RUNNING,
            (JobModel.lease_expires_at < now)
            | ((JobModel.lease_expires_at.is_(None)) & (JobModel.locked_at < cutoff)),
        )
        stale_jobs = self._session.execute(stmt).scalars().all()
        recovered_count = 0

        for job in stale_jobs:
            recovered_count += 1
            if job.attempts < job.max_attempts:
                # Re-queue for retry with cleared lease
                job.status = JobStatus.QUEUED
                job.locked_by = None
                job.locked_at = None
                job.lease_token = None
                job.lease_expires_at = None
                job.updated_at = now
            else:
                # Max attempts exhausted, mark timed out
                job.status = JobStatus.TIMED_OUT
                job.lease_token = None
                job.lease_expires_at = None
                job.error_message = (
                    f"Execution exceeded lease duration ({stale_threshold_seconds}s) "
                    f"and max retry limit ({job.max_attempts}) was exhausted."
                )
                job.updated_at = now

        if recovered_count > 0:
            self._session.commit()

        return recovered_count

    def recover_strands_on_startup(self) -> int:
        """Reconciles expired stranded RUNNING jobs when application restarts.

        Never resets active, unexpired leases from other running worker instances.
        Only recovers jobs whose lease has genuinely expired.
        """
        now = now_utc()
        stmt = select(JobModel).where(
            JobModel.status == JobStatus.RUNNING,
            (JobModel.lease_expires_at < now) | (JobModel.lease_expires_at.is_(None)),
        )
        stranded_jobs = self._session.execute(stmt).scalars().all()
        count = 0

        for job in stranded_jobs:
            count += 1
            if job.attempts < job.max_attempts:
                job.status = JobStatus.QUEUED
                job.locked_by = None
                job.locked_at = None
                job.lease_token = None
                job.lease_expires_at = None
                job.updated_at = now
            else:
                job.status = JobStatus.FAILED
                job.lease_token = None
                job.lease_expires_at = None
                job.error_message = (
                    "Application process restarted after job lease expired in RUNNING state."
                )
                job.updated_at = now

        if count > 0:
            self._session.commit()

        return count

    def _validate_transition(self, current: JobStatus, target: JobStatus) -> None:
        """Validates that a proposed state transition is legally permissible."""
        allowed = _VALID_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidJobStateTransitionError(
                f"Illegal job state transition from '{current}' to '{target}'. "
                f"Allowed target states: {[s.value for s in allowed]}."
            )
