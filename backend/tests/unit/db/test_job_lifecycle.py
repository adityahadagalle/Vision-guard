"""Unit tests for SQLite job queue state machine and lifecycle orchestration."""

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.job_repository import (
    InvalidJobStateTransitionError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.session import _set_sqlite_pragmas, now_utc
from backend.app.domain.entities.enums import JobStatus


@pytest.fixture
def job_queue(tmp_path: Path):
    """Provides an isolated SqliteJobQueue backed by a temporary SQLite file."""
    db_file = tmp_path / "jobs.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    queue = SqliteJobQueue(session)
    yield queue
    session.close()
    engine.dispose()


def test_job_submission_and_successful_completion(job_queue):
    """Tests submitting, claiming, and completing a job."""
    jid = job_queue.submit_job(task_type="ingestion", payload={"archive_id": "arc_123"})
    job = job_queue.get_job(jid)
    assert job["status"] == JobStatus.QUEUED
    assert job["attempts"] == 0

    # Claim job
    claimed = job_queue.claim_next_job(worker_id="worker_alpha")
    assert claimed is not None
    assert claimed["job_id"] == jid
    assert claimed["status"] == JobStatus.RUNNING
    assert claimed["attempts"] == 1
    assert claimed["locked_by"] == "worker_alpha"
    assert claimed["lease_token"] is not None

    # Complete job
    job_queue.complete_job(
        job_id=jid,
        worker_id="worker_alpha",
        lease_token=claimed["lease_token"],
        result={"snapshot_id": "snap_123"},
    )
    completed = job_queue.get_job(jid)
    assert completed["status"] == JobStatus.COMPLETED
    assert completed["result"]["snapshot_id"] == "snap_123"


def test_job_failure_records_error_info(job_queue):
    """Tests failing a job records error information and terminal status."""
    jid = job_queue.submit_job(task_type="assessment", payload={"dataset_id": "ds_1"})
    claimed = job_queue.claim_next_job(worker_id="worker_beta")
    assert claimed is not None

    job_queue.fail_job(
        job_id=jid,
        worker_id="worker_beta",
        lease_token=claimed["lease_token"],
        error_message="Detector crashed: OOM",
    )
    failed = job_queue.get_job(jid)
    assert failed["status"] == JobStatus.FAILED
    assert failed["error_message"] == "Detector crashed: OOM"


def test_invalid_state_transitions_rejected(job_queue):
    """Tests that illegal state transitions raise InvalidJobStateTransitionError."""
    jid = job_queue.submit_job(task_type="test_task", payload={})

    # Cannot transition directly from QUEUED to COMPLETED
    with pytest.raises(InvalidJobStateTransitionError):
        job_queue.complete_job(job_id=jid, worker_id="worker_1")

    # Claim to RUNNING
    claimed = job_queue.claim_next_job(worker_id="worker_1")
    assert claimed is not None

    # Complete to COMPLETED (terminal)
    job_queue.complete_job(job_id=jid, worker_id="worker_1", lease_token=claimed["lease_token"])

    # Cannot transition from terminal COMPLETED to RUNNING or FAILED
    with pytest.raises(InvalidJobStateTransitionError):
        job_queue.fail_job(
            job_id=jid,
            worker_id="worker_1",
            lease_token=claimed["lease_token"],
            error_message="Late error",
        )


def test_unauthorized_worker_cannot_complete_or_fail_job(job_queue):
    """Tests that a worker cannot complete or fail a job claimed by another worker."""
    jid = job_queue.submit_job(task_type="secure_task", payload={})
    job_queue.claim_next_job(worker_id="worker_owner")

    with pytest.raises(JobNotClaimedError):
        job_queue.complete_job(job_id=jid, worker_id="worker_imposter")

    with pytest.raises(JobNotClaimedError):
        job_queue.fail_job(job_id=jid, worker_id="worker_imposter", error_message="Tamper")


def test_prevent_duplicate_job_execution(job_queue):
    """Tests that claiming an already running job is impossible."""
    jid = job_queue.submit_job(task_type="unique_task", payload={})

    # First worker claims the only queued job
    c1 = job_queue.claim_next_job(worker_id="worker_1")
    assert c1 is not None
    assert c1["job_id"] == jid

    # Second worker attempts to claim; queue should be empty
    c2 = job_queue.claim_next_job(worker_id="worker_2")
    assert c2 is None


def test_stale_job_recovery_requeues_and_times_out(job_queue):
    """Tests recovering stale jobs: retrying within max_attempts and timing out when exhausted."""
    # Job 1: attempts < max_attempts -> should be re-queued
    j1 = job_queue.submit_job(task_type="t1", payload={}, max_attempts=2)
    job_queue.claim_next_job(worker_id="worker_crashed_1")

    # Artificially set locked_at and lease_expires_at in the past
    from backend.app.db.models import JobModel

    m1 = job_queue._session.get(JobModel, j1)
    m1.locked_at = now_utc() - timedelta(seconds=400)
    m1.lease_expires_at = now_utc() - timedelta(seconds=100)
    job_queue._session.commit()

    recovered = job_queue.recover_stale_jobs(stale_threshold_seconds=300)
    assert recovered == 1
    job1_after = job_queue.get_job(j1)
    assert job1_after["status"] == JobStatus.QUEUED
    assert job1_after["locked_by"] is None

    # Claim again -> attempts becomes 2 (equal to max_attempts)
    job_queue.claim_next_job(worker_id="worker_crashed_again")
    m1 = job_queue._session.get(JobModel, j1)
    m1.locked_at = now_utc() - timedelta(seconds=400)
    m1.lease_expires_at = now_utc() - timedelta(seconds=100)
    job_queue._session.commit()

    # Now max_attempts is exhausted; recover_stale_jobs should mark it TIMED_OUT
    recovered2 = job_queue.recover_stale_jobs(stale_threshold_seconds=300)
    assert recovered2 == 1
    job1_timed_out = job_queue.get_job(j1)
    assert job1_timed_out["status"] == JobStatus.TIMED_OUT
    assert "exhausted" in job1_timed_out["error_message"]


def test_application_restart_recovers_stranded_jobs(job_queue):
    """Tests startup recovery of stranded RUNNING jobs after unexpected crash/restart."""
    from backend.app.db.models import JobModel

    j_stranded = job_queue.submit_job(task_type="crash_task", payload={}, max_attempts=3)
    claimed = job_queue.claim_next_job(worker_id="worker_pre_crash")
    assert claimed is not None

    # 1. Unexpired lease MUST NOT be recovered on startup (active concurrent worker preservation)
    assert job_queue.recover_strands_on_startup() == 0
    assert job_queue.get_job(j_stranded)["status"] == JobStatus.RUNNING

    # 2. Simulate expired lease from crashed worker
    m = job_queue._session.get(JobModel, j_stranded)
    m.lease_expires_at = now_utc() - timedelta(seconds=10)
    job_queue._session.commit()

    # App restart now successfully recovers the expired stranded job
    recovered_count = job_queue.recover_strands_on_startup()
    assert recovered_count == 1

    job_state = job_queue.get_job(j_stranded)
    assert job_state["status"] == JobStatus.QUEUED
    assert job_state["locked_by"] is None
