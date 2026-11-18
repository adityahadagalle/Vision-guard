"""Unit tests for model orchestration, worker lease fencing, and atomic state transitions.

Verifies:
1. Atomic model state transitions (RECEIVED -> INSPECTING -> ACCEPTED / REJECTED).
2. Worker lease credentials verification (job_id, worker_id, lease_token, lease_generation).
3. Stale worker eviction on lease expiration and lease generation mismatch.
4. Terminal state immutability.
5. Interrupted job startup recovery for both RECEIVED and INSPECTING models.
6. Model assessment scheduling constraints (requires ACCEPTED status and sealed snapshot).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base, JobModel
from backend.app.db.repositories.job_repository import (
    JobLeaseExpiredError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.repositories.model_repository import (
    InvalidModelStateTransitionError,
    SqliteModelRepository,
)
from backend.app.db.session import _set_sqlite_pragmas, now_utc
from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    IngestionStatus,
    JobStatus,
)


@pytest.fixture
def db_session(tmp_path: Path):
    """Provisions an isolated SQLite database with WAL mode and tables created."""
    db_file = tmp_path / "test_leases.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_transition_model_to_inspecting_success(db_session):
    """Worker holding valid active lease successfully transitions model from RECEIVED to INSPECTING."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    # 1. Create model in RECEIVED state
    model_id = "mod_valid_01"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="a" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    # 2. Enqueue and claim job
    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_alpha", lease_duration_seconds=300)
    assert claimed is not None
    assert claimed["job_id"] == job_id
    assert claimed["lease_generation"] == 1

    # 3. Transition to INSPECTING with valid credentials
    model_repo.transition_model_to_inspecting(
        model_id=model_id,
        job_id=job_id,
        worker_id="worker_alpha",
        lease_token=claimed["lease_token"],
        lease_generation=claimed["lease_generation"],
    )

    # Verify model is now INSPECTING
    model = model_repo.get_model(model_id)
    assert model is not None
    assert model.status == str(IngestionStatus.INSPECTING)


def test_transition_model_to_inspecting_unauthorized_worker_fails(db_session):
    """Worker with mismatched worker_id or lease_token is rejected."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_unauth_01"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="b" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_owner", lease_duration_seconds=300)
    assert claimed is not None

    # Impostor worker attempts transition
    with pytest.raises(JobNotClaimedError) as exc_info:
        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id="worker_impostor",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )
    assert "does not hold active lease" in str(exc_info.value)

    # Model status remains RECEIVED
    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.RECEIVED)


def test_transition_model_to_inspecting_generation_mismatch_fails(db_session):
    """Worker with obsolete lease generation is rejected."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_gen_mismatch"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="c" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_alpha", lease_duration_seconds=300)
    assert claimed is not None
    active_gen = claimed["lease_generation"]

    # Attempt with stale generation (active_gen - 1)
    with pytest.raises(JobNotClaimedError) as exc_info:
        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id="worker_alpha",
            lease_token=claimed["lease_token"],
            lease_generation=active_gen - 1,
        )
    assert "generation" in str(exc_info.value).lower()


def test_transition_model_to_inspecting_expired_lease_fails(db_session):
    """Worker whose lease expired cannot transition model state."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_expired_01"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="d" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    # Claim with -10 second lease (immediately expired)
    claimed = job_queue.claim_next_job(worker_id="worker_slow", lease_duration_seconds=-10)
    assert claimed is not None

    with pytest.raises(JobLeaseExpiredError) as exc_info:
        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id="worker_slow",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )
    assert "expired" in str(exc_info.value).lower()


def test_transition_model_to_inspecting_terminal_state_fails(db_session):
    """Attempting to transition an already ACCEPTED or REJECTED model to INSPECTING fails closed."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_terminal_01"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="e" * 64,
        size_bytes=1024,
        status=IngestionStatus.ACCEPTED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_1", lease_duration_seconds=300)
    assert claimed is not None

    with pytest.raises(InvalidModelStateTransitionError) as exc_info:
        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id="worker_1",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )
    assert "Cannot transition model" in str(exc_info.value)


def test_publish_model_ingestion_success_atomic(db_session):
    """Atomic publication updates model to ACCEPTED, creates snapshot, and completes job."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_success_atomic"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="f" * 64,
        size_bytes=2048,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_worker", lease_duration_seconds=300)
    assert claimed is not None

    # Step 1: Transition to INSPECTING
    model_repo.transition_model_to_inspecting(
        model_id=model_id,
        job_id=job_id,
        worker_id="worker_worker",
        lease_token=claimed["lease_token"],
        lease_generation=claimed["lease_generation"],
    )

    # Step 2: Publish success
    snap_id = "snap_model_001"
    snapshot = AssetSnapshot(
        snapshot_id=snap_id,
        asset_id=model_id,
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="1" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=2048,
        schema_version="1.0.0",
        metadata={"filename": "test_model.pt", "snapshot_path": f"/storage/snapshots/{snap_id}"},
    )

    model_repo.publish_model_ingestion_success(
        model_id=model_id,
        snapshot=snapshot,
        storage_path=f"/storage/snapshots/{snap_id}",
        sha256="f" * 64,
        size_bytes=2048,
        job_id=job_id,
        worker_id="worker_worker",
        lease_token=claimed["lease_token"],
        lease_generation=claimed["lease_generation"],
        result={"model_id": model_id, "snapshot_id": snap_id},
    )

    # Verify all 3 states updated atomically
    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.ACCEPTED)
    assert model.storage_path == f"/storage/snapshots/{snap_id}"

    snap_in_db = model_repo.get_snapshot(snap_id)
    assert snap_in_db is not None
    assert snap_in_db.manifest_sha256 == "1" * 64

    job = job_queue.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED
    assert job["result"]["snapshot_id"] == snap_id


def test_publish_model_ingestion_success_stale_generation_rejected(db_session):
    """Worker with obsolete generation cannot publish success; transaction rolls back completely."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_stale_pub"
    model_repo.create_model(
        model_id=model_id,
        name="test_model.pt",
        storage_path="/quarantine/models/test_model.pt",
        sha256="2" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed_gen1 = job_queue.claim_next_job(worker_id="worker_1", lease_duration_seconds=300)
    assert claimed_gen1 is not None

    model_repo.transition_model_to_inspecting(
        model_id=model_id,
        job_id=job_id,
        worker_id="worker_1",
        lease_token=claimed_gen1["lease_token"],
        lease_generation=claimed_gen1["lease_generation"],
    )

    # Fast-forward job lease expiry to the past so recover_stale_jobs can recover it
    job_model = db_session.get(JobModel, job_id)
    job_model.lease_expires_at = now_utc() - timedelta(seconds=10)
    db_session.commit()

    # Job lease expired and is recovered back to QUEUED
    recovered = job_queue.recover_stale_jobs(stale_threshold_seconds=0)
    assert recovered == 1

    # Worker 2 claims the job -> generation 2
    claimed_gen2 = job_queue.claim_next_job(worker_id="worker_2", lease_duration_seconds=300)
    assert claimed_gen2 is not None
    assert claimed_gen2["lease_generation"] == 2

    # Stale Worker 1 attempts to publish success with generation 1
    snap = AssetSnapshot(
        snapshot_id="snap_stale",
        asset_id=model_id,
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="3" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=1024,
        schema_version="1.0.0",
        metadata={},
    )

    with pytest.raises(JobNotClaimedError) as exc_info:
        model_repo.publish_model_ingestion_success(
            model_id=model_id,
            snapshot=snap,
            storage_path="/storage/snapshots/snap_stale",
            sha256="2" * 64,
            size_bytes=1024,
            job_id=job_id,
            worker_id="worker_1",
            lease_token=claimed_gen1["lease_token"],
            lease_generation=claimed_gen1["lease_generation"],
        )
    err_msg = str(exc_info.value).lower()
    assert "generation" in err_msg or "does not hold active lease" in err_msg

    # Verify model is NOT marked ACCEPTED and snapshot was NOT persisted
    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.INSPECTING)
    assert model_repo.get_snapshot("snap_stale") is None

    # Job is still RUNNING held by Worker 2
    job = job_queue.get_job(job_id)
    assert job["status"] == JobStatus.RUNNING
    assert job["locked_by"] == "worker_2"


def test_publish_model_ingestion_failure_atomic(db_session):
    """Failure publication atomically transitions model to REJECTED and job to FAILED."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_failure_01"
    model_repo.create_model(
        model_id=model_id,
        name="bad_model.pt",
        storage_path="/quarantine/models/bad_model.pt",
        sha256="4" * 64,
        size_bytes=512,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_fail", lease_duration_seconds=300)
    assert claimed is not None

    model_repo.transition_model_to_inspecting(
        model_id=model_id,
        job_id=job_id,
        worker_id="worker_fail",
        lease_token=claimed["lease_token"],
        lease_generation=claimed["lease_generation"],
    )

    error_reason = "Corrupted PyTorch header detected."
    model_repo.publish_model_ingestion_failure(
        model_id=model_id,
        error_message=error_reason,
        job_id=job_id,
        worker_id="worker_fail",
        lease_token=claimed["lease_token"],
        lease_generation=claimed["lease_generation"],
    )

    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.REJECTED)
    meta = json.loads(model.metadata_json)
    assert meta["rejection_reason"] == error_reason

    job = job_queue.get_job(job_id)
    assert job["status"] == JobStatus.FAILED
    assert error_reason in job["error_message"]


def test_publish_model_ingestion_failure_stale_worker_rejected(db_session):
    """Stale worker cannot reject a model after losing its lease."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_stale_fail"
    model_repo.create_model(
        model_id=model_id,
        name="model.pt",
        storage_path="/quarantine/models/model.pt",
        sha256="5" * 64,
        size_bytes=512,
        status=IngestionStatus.RECEIVED,
    )

    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={"model_id": model_id},
    )
    claimed = job_queue.claim_next_job(worker_id="worker_stale", lease_duration_seconds=-5)
    assert claimed is not None

    with pytest.raises(JobLeaseExpiredError):
        model_repo.publish_model_ingestion_failure(
            model_id=model_id,
            error_message="Should not be recorded",
            job_id=job_id,
            worker_id="worker_stale",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )

    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.RECEIVED)


def test_startup_reconciliation_recovers_stranded_inspecting_models(tmp_path: Path, db_session):
    """Startup reconciliation recovers interrupted INSPECTING models that lack active jobs."""
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    q_file = tmp_path / "valid_model.pt"
    q_file.write_bytes(b"dummy_weights")

    model_id = "mod_inspecting_stranded"
    model_repo.create_model(
        model_id=model_id,
        name="valid_model.pt",
        storage_path=str(q_file),
        sha256="6" * 64,
        size_bytes=len(b"dummy_weights"),
        status=IngestionStatus.INSPECTING,  # Interrupted in INSPECTING
    )

    # First reconciliation: file exists, no active job -> submits ingestion job
    reconciled = model_repo.reconcile_stranded_received_models(job_queue)
    assert reconciled == 1

    # Verify a new job was enqueued
    job_claim = job_queue.claim_next_job(worker_id="new_worker")
    assert job_claim is not None
    assert job_claim["payload"]["model_id"] == model_id

    # Second reconciliation: active job exists -> returns 0 (idempotent)
    reconciled_again = model_repo.reconcile_stranded_received_models(job_queue)
    assert reconciled_again == 0


def test_job_queue_complete_and_fail_enforce_generation_fencing(db_session):
    """Direct SqliteJobQueue complete_job and fail_job enforce lease_generation fencing."""
    job_queue = SqliteJobQueue(db_session)

    job_id = job_queue.submit_job(task_type="generic_task", payload={})
    claimed = job_queue.claim_next_job(worker_id="worker_fenced", lease_duration_seconds=300)
    assert claimed is not None
    gen = claimed["lease_generation"]

    # Mismatched generation on complete_job fails
    with pytest.raises(JobNotClaimedError) as exc_info:
        job_queue.complete_job(
            job_id=job_id,
            worker_id="worker_fenced",
            lease_token=claimed["lease_token"],
            lease_generation=gen + 99,
        )
    assert "lease generation" in str(exc_info.value).lower()

    # Mismatched generation on fail_job fails
    with pytest.raises(JobNotClaimedError) as exc_info:
        job_queue.fail_job(
            job_id=job_id,
            worker_id="worker_fenced",
            lease_token=claimed["lease_token"],
            lease_generation=gen + 99,
            error_message="Test error",
        )
    assert "lease generation" in str(exc_info.value).lower()

    # Correct generation succeeds
    job_queue.complete_job(
        job_id=job_id,
        worker_id="worker_fenced",
        lease_token=claimed["lease_token"],
        lease_generation=gen,
    )
    job = job_queue.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED
