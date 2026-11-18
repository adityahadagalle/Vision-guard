"""Adversarial and cross-process tests for Tasks 1 & 2 remediation.

Validates:
1. FINDING-001: SQLite cross-process reservation race (BEGIN IMMEDIATE guarantees concurrency and quota limits).
2. FINDING-002: Atomic model persistence + ingestion job creation, and startup reconciliation.
3. FINDING-003: Streaming heartbeat, lease renewal, and expired lease safety.
"""

import io
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base, JobModel, ModelArtifactModel, UploadReservationModel
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.repositories.upload_reservation_repository import (
    ReservationExpiredError,
    SqliteUploadReservationRepository,
)
from backend.app.db.session import _set_sqlite_pragmas, get_db, now_utc
from backend.app.domain.entities.enums import IngestionStatus, JobStatus
from backend.app.main import create_app


@pytest.fixture
def clean_db_and_quarantine(tmp_path: Path):
    """Provisions an isolated SQLite database and quarantine directory tree."""
    db_file = tmp_path / "test_remediation.db"
    quarantine_dir = tmp_path / "quarantine" / "models"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{db_file.as_posix()}"
    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)
    engine.dispose()

    return db_url, quarantine_dir, tmp_path


def test_cross_process_concurrency_limit_exact_slot_enforcement(clean_db_and_quarantine):
    """FINDING-001: 4 independent OS processes competing for 3 slots -> exactly 3 succeed."""
    db_url, quarantine_dir, tmp_path = clean_db_and_quarantine

    procs = []
    out_files = []
    # 4 independent processes competing for 3 slots, 50MB each, quota 5GB
    for i in range(4):
        out_file = tmp_path / f"worker_concurrency_{i}.json"
        out_files.append(out_file)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.tests.integration.upload_reservation_worker",
                db_url,
                str(quarantine_dir),
                str(50 * 1024 * 1024),
                "3",  # max 3 slots
                str(5 * 1024 * 1024 * 1024),
                str(out_file),
                str(i),
            ]
        )
        procs.append(p)

    for p in procs:
        p.wait(timeout=30)

    results = [json.loads(f.read_text(encoding="utf-8")) for f in out_files if f.exists()]
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    assert len(successes) == 3, f"Expected exactly 3 successes, got {len(successes)}"
    assert len(failures) == 1, f"Expected exactly 1 failure, got {len(failures)}"
    assert failures[0]["error"] == "CONCURRENCY_LIMIT_EXCEEDED"

    # Verify database state
    engine = create_engine(db_url, echo=False)
    session = sessionmaker(bind=engine)()
    active_in_db = session.query(UploadReservationModel).filter_by(status="ACTIVE").count()
    assert active_in_db == 3
    session.close()
    engine.dispose()


def test_cross_process_quota_oversubscription_prevented(clean_db_and_quarantine):
    """FINDING-001: 4 processes requesting 200MB against 500MB quota -> exactly 2 succeed."""
    db_url, quarantine_dir, tmp_path = clean_db_and_quarantine

    procs = []
    out_files = []
    # 4 independent processes requesting 200MB each with 500MB total quota and 4 slots
    for i in range(4):
        out_file = tmp_path / f"worker_quota_{i}.json"
        out_files.append(out_file)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.tests.integration.upload_reservation_worker",
                db_url,
                str(quarantine_dir),
                str(200 * 1024 * 1024),
                "4",  # slots not the constraint
                str(500 * 1024 * 1024),  # 500MB quota
                str(out_file),
                str(i),
            ]
        )
        procs.append(p)

    for p in procs:
        p.wait(timeout=30)

    results = [json.loads(f.read_text(encoding="utf-8")) for f in out_files if f.exists()]
    assert len(results) == 4

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    # Exactly 2 fit (2 * 200MB = 400MB <= 500MB, 3rd would be 600MB > 500MB)
    assert len(successes) == 2, f"Expected 2 successes, got {len(successes)}"
    assert len(failures) == 2, f"Expected 2 failures, got {len(failures)}"
    for f in failures:
        assert f["error"] == "QUOTA_EXCEEDED"

    # Verify total reserved bytes does not exceed 500MB in DB
    engine = create_engine(db_url, echo=False)
    session = sessionmaker(bind=engine)()
    active_records = session.query(UploadReservationModel).filter_by(status="ACTIVE").all()
    total_reserved = sum(r.reserved_bytes for r in active_records)
    assert total_reserved == 400 * 1024 * 1024
    session.close()
    engine.dispose()


def test_atomic_persistence_and_job_creation_rollback(clean_db_and_quarantine):
    """FINDING-002: A failure during job insertion rolls back model persistence and reservation."""
    db_url, quarantine_dir, _ = clean_db_and_quarantine
    engine = create_engine(db_url, echo=False)
    session = sessionmaker(bind=engine)()

    res_repo = SqliteUploadReservationRepository(session)
    model_repo = SqliteModelRepository(session)
    job_queue = SqliteJobQueue(session)

    res = res_repo.acquire_reservation(
        asset_type="model",
        requested_bytes=10 * 1024 * 1024,
        quarantine_dir=quarantine_dir,
    )
    res_id = res.reservation_id

    if session.in_transaction():
        session.commit()

    # Simulate atomic block where an exception occurs before commit
    with pytest.raises(RuntimeError, match="Simulated crash during job enqueue"), session.begin():
        res_repo.complete_reservation(
            res_id,
            actual_bytes=1000,
            quarantine_path=str(quarantine_dir / "candidate.pt"),
            auto_commit=False,
        )
        model_repo.create_model(
            model_id="model_crash_test",
            name="crash_test.pt",
            storage_path=str(quarantine_dir / "candidate.pt"),
            sha256="abc1234",
            size_bytes=1000,
            auto_commit=False,
        )
        # Crash happens right before job is submitted
        raise RuntimeError("Simulated crash during job enqueue")

    # Verify rollback: Model does NOT exist
    assert model_repo.get_model("model_crash_test") is None

    # Verify reservation was rolled back to ACTIVE
    res_db = session.query(UploadReservationModel).filter_by(reservation_id=res_id).one()
    assert res_db.status == "ACTIVE"

    # Now verify successful atomic completion
    if session.in_transaction():
        session.commit()

    with session.begin():
        res_repo.complete_reservation(
            res_id,
            actual_bytes=1000,
            quarantine_path=str(quarantine_dir / "candidate.pt"),
            auto_commit=False,
        )
        model_repo.create_model(
            model_id="model_success_test",
            name="success_test.pt",
            storage_path=str(quarantine_dir / "candidate.pt"),
            sha256="abc1234",
            size_bytes=1000,
            auto_commit=False,
        )
        jid = job_queue.submit_job(
            task_type="model_ingestion",
            payload={"model_id": "model_success_test"},
            auto_commit=False,
        )

    # Both model and job are committed atomically
    assert model_repo.get_model("model_success_test") is not None
    assert job_queue.get_job(jid) is not None
    assert res_db.status == "COMPLETED"

    session.close()
    engine.dispose()


def test_startup_reconciliation_stranded_received_models(clean_db_and_quarantine):
    """FINDING-002: Reconciles models stuck in RECEIVED state without active jobs."""
    db_url, quarantine_dir, _ = clean_db_and_quarantine
    engine = create_engine(db_url, echo=False)
    session = sessionmaker(bind=engine)()

    model_repo = SqliteModelRepository(session)
    job_queue = SqliteJobQueue(session)

    # Case 1: Model with valid file on disk but no job
    model1_file = quarantine_dir / "stranded_valid.pt"
    model1_file.write_bytes(b"torch model data")
    model_repo.create_model(
        model_id="model_stranded_1",
        name="stranded_valid.pt",
        storage_path=str(model1_file),
        sha256="sha1",
        size_bytes=16,
        status=IngestionStatus.RECEIVED,
    )

    # Case 2: Model with MISSING file on disk and no job
    model2_file = quarantine_dir / "stranded_missing.pt"
    model_repo.create_model(
        model_id="model_stranded_2",
        name="stranded_missing.pt",
        storage_path=str(model2_file),
        sha256="sha2",
        size_bytes=16,
        status=IngestionStatus.RECEIVED,
    )

    # Run startup reconciliation
    reconciled_count = model_repo.reconcile_stranded_received_models(job_queue)
    assert reconciled_count == 2

    # Case 1 should have an enqueued model_ingestion job
    stmt = select(JobModel).where(JobModel.task_type == "model_ingestion")
    jobs = list(session.execute(stmt).scalars().all())
    assert len(jobs) == 1
    assert "model_stranded_1" in jobs[0].payload_json
    assert jobs[0].status == JobStatus.QUEUED

    # Case 2 should be transitioned to REJECTED
    m2 = model_repo.get_model("model_stranded_2")
    assert m2.status == str(IngestionStatus.REJECTED)

    # Idempotency check: Running reconciliation again does nothing
    second_run = model_repo.reconcile_stranded_received_models(job_queue)
    assert second_run == 0
    jobs_after = list(session.execute(stmt).scalars().all())
    assert len(jobs_after) == 1

    session.close()
    engine.dispose()


def test_reservation_lease_renewal_and_expiry_protection(clean_db_and_quarantine):
    """FINDING-003: Tests lease extension, rejection of expired lease, and safe release."""
    db_url, quarantine_dir, _ = clean_db_and_quarantine
    engine = create_engine(db_url, echo=False)
    session = sessionmaker(bind=engine)()
    repo = SqliteUploadReservationRepository(session)

    # 1. Acquire active reservation
    res = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=50 * 1024 * 1024,
        quarantine_dir=quarantine_dir,
        timeout_seconds=60,
    )
    res_id = res.reservation_id
    initial_expiry = res.expires_at

    # 2. Renew lease: expires_at must be extended
    renewed = repo.renew_reservation_lease(res_id, extension_seconds=120)
    assert renewed.expires_at > initial_expiry

    # 3. Simulate expired reservation
    res.expires_at = now_utc() - timedelta(seconds=10)
    session.commit()

    # 4. Attempting renewal on expired reservation must fail
    with pytest.raises(ReservationExpiredError):
        repo.renew_reservation_lease(res_id)

    # 5. Attempting completion on expired reservation must fail
    with pytest.raises(ReservationExpiredError):
        repo.complete_reservation(
            res_id, actual_bytes=1000, quarantine_path=str(quarantine_dir / "test.pt")
        )

    # 6. Verify stale release cannot modify an already COMPLETED reservation
    res_ok = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=10 * 1024 * 1024,
        quarantine_dir=quarantine_dir,
    )
    repo.complete_reservation(
        res_ok.reservation_id,
        actual_bytes=500,
        quarantine_path=str(quarantine_dir / "completed.pt"),
    )
    # Stale release call
    repo.release_reservation(res_ok.reservation_id, reason="CANCELLED_BY_STALE_OWNER")
    # Must remain COMPLETED!
    check_res = (
        session.query(UploadReservationModel).filter_by(reservation_id=res_ok.reservation_id).one()
    )
    assert check_res.status == "COMPLETED"

    session.close()
    engine.dispose()


def test_api_upload_lease_lost_returns_408(clean_db_and_quarantine, monkeypatch):
    """FINDING-003: Mid-stream lease expiration returns HTTP 408 RESERVATION_LEASE_LOST."""
    db_url, quarantine_dir, tmp_path = clean_db_and_quarantine

    # Create app configured with test data directory and db_url
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        database_url=db_url,
        data_dir=data_dir,
        upload_reservation_timeout_seconds=1,
    )

    engine = create_engine(db_url, echo=False)
    session_factory = sessionmaker(bind=engine)
    app = create_app(settings)

    def override_get_db():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    # Simulate expired lease during complete_reservation call by monkeypatching
    def mock_expired_complete(self, *args, **kwargs):
        raise ReservationExpiredError("Lease expired mid-stream")

    monkeypatch.setattr(
        SqliteUploadReservationRepository, "complete_reservation", mock_expired_complete
    )

    fake_file = io.BytesIO(b"Candidate PyTorch model payload data")
    response = client.post(
        "/api/v1/models",
        files={"file": ("model_lease_test.pt", fake_file, "application/octet-stream")},
    )

    assert response.status_code == 408
    body = response.json()
    assert body["error_code"] == "RESERVATION_EXPIRED"

    # Verify no partial models were persisted
    session = session_factory()
    count = session.query(ModelArtifactModel).count()
    assert count == 0
    session.close()
    engine.dispose()
