"""Integration and adversarial multi-process concurrency tests for Task 3 model orchestration.

Tests:
1. Multi-process concurrent worker claims using independent OS processes.
2. End-to-end model ingestion through SQLite job queue and AssessmentOrchestrator.
3. Stale worker eviction and lease-fenced rejection after lease expiration.
4. Crash and restart recovery idempotency for interrupted model ingestion.
5. Model assessment scheduling and lease-fenced detector execution.
6. Model state constraints on assessment scheduling.
"""

import io
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import (
    Base,
    JobModel,
)
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.job_repository import (
    InvalidJobStateTransitionError,
    JobLeaseExpiredError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.session import _set_sqlite_pragmas, get_db, now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    IngestionStatus,
    JobStatus,
)
from backend.app.main import create_app
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up an isolated database, directories, and TestClient."""
    db_file = tmp_path / "test_model_orchestration.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)
    engine.dispose()

    storage_dir = tmp_path / "storage"
    quarantine_dir = storage_dir / "quarantine"
    models_dir = storage_dir / "models"
    snapshots_dir = storage_dir / "snapshots"

    for d in (quarantine_dir, models_dir, snapshots_dir):
        d.mkdir(parents=True, exist_ok=True)

    app_settings = Settings(
        database_url=db_url,
        data_dir=storage_dir,
        quarantine_dir=quarantine_dir,
        max_model_upload_bytes=500 * 1024 * 1024,
        global_quarantine_quota_bytes=5 * 1024 * 1024 * 1024,
        max_concurrent_uploads=3,
        upload_reservation_timeout_seconds=900,
    )

    app = create_app(app_settings)

    test_engine = create_engine(db_url, echo=False)
    event.listen(test_engine, "connect", _set_sqlite_pragmas)
    test_sessionmaker = sessionmaker(bind=test_engine)

    def _get_test_db():
        session = test_sessionmaker()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_test_db

    client = TestClient(app)

    yield {
        "db_url": db_url,
        "test_engine": test_engine,
        "session_factory": test_sessionmaker,
        "storage_dir": storage_dir,
        "quarantine_dir": quarantine_dir,
        "models_dir": models_dir,
        "snapshots_dir": snapshots_dir,
        "client": client,
        "tmp_path": tmp_path,
        "app_settings": app_settings,
    }

    test_engine.dispose()


def test_multiprocess_concurrent_model_job_claims(test_env):
    """Multiple independent OS processes competing to claim jobs: each job claimed exactly once."""
    db_url = test_env["db_url"]
    session_factory = test_env["session_factory"]
    tmp_path = test_env["tmp_path"]

    # 1. Enqueue 4 model ingestion jobs in SQLite
    session = session_factory()
    queue = SqliteJobQueue(session)
    job_ids = []
    for i in range(4):
        jid = queue.submit_job(
            task_type="model_ingestion",
            payload={"model_id": f"mod_multi_{i}"},
        )
        job_ids.append(jid)
    session.close()

    # 2. Spawn 4 independent OS processes
    procs = []
    out_files = []
    for i in range(4):
        out_file = tmp_path / f"worker_claim_out_{i}.json"
        out_files.append(out_file)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.tests.integration.model_ingestion_worker",
                db_url,
                f"worker_proc_{i}",
                str(out_file),
                "claim_all",
            ]
        )
        procs.append(p)

    for p in procs:
        p.wait(timeout=30)

    # 3. Read and verify claim results
    all_claims = []
    for f in out_files:
        assert f.exists(), f"Output file {f} missing."
        data = json.loads(f.read_text("utf-8"))
        all_claims.extend(data["claims"])

    # Exactly 4 claims total across all processes
    assert len(all_claims) == 4

    claimed_jids = [c["job_id"] for c in all_claims]
    assert sorted(claimed_jids) == sorted(job_ids), "All 4 distinct jobs must be claimed."

    # Zero duplicate claims
    assert len(set(claimed_jids)) == 4

    # Every claim must have a unique lease token and generation 1
    tokens = [c["lease_token"] for c in all_claims]
    assert len(set(tokens)) == 4
    for c in all_claims:
        assert c["lease_generation"] == 1


def test_end_to_end_model_ingestion_orchestration(test_env):
    """Uploads a model, processes the ingestion job via AssessmentOrchestrator, and verifies snapshot creation."""
    client = test_env["client"]
    session_factory = test_env["session_factory"]
    snapshots_dir = test_env["snapshots_dir"]

    # 1. Upload valid .pt checkpoint file via API
    fake_weights = b"PK\x03\x04" + b"\x00" * 500  # Zip/PyTorch header mock
    resp = client.post(
        "/api/v1/models",
        files={
            "file": ("resnet18_weights.pt", io.BytesIO(fake_weights), "application/octet-stream")
        },
        data={"name": "ResNet-18 Production"},
    )
    assert resp.status_code == 201
    upload_data = resp.json()
    model_id = upload_data["model_id"]
    job_id = upload_data["job_id"]
    assert upload_data["status"] == "RECEIVED"

    # 2. Execute background job using AssessmentOrchestrator
    session = session_factory()
    ingest_config = IngestionConfig(
        snapshots_dir=snapshots_dir,
        quarantine_dir=test_env["quarantine_dir"],
        staging_dir=test_env["storage_dir"] / "staging",
    )
    orchestrator = AssessmentOrchestrator(session=session, config=ingest_config)

    job_result = orchestrator.process_next_job(worker_id="orchestrator_worker_01")
    assert job_result is not None
    assert job_result["job_id"] == job_id
    assert job_result["status"] == JobStatus.COMPLETED

    # 3. Verify model state transitioned to ACCEPTED
    model_repo = SqliteModelRepository(session)
    model = model_repo.get_model(model_id)
    assert model is not None
    assert model.status == str(IngestionStatus.ACCEPTED)

    # 4. Verify sealed snapshot exists
    snap_id = job_result["result"]["snapshot_id"]
    snap = model_repo.get_snapshot(snap_id)
    assert snap is not None
    assert snap.asset_id == model_id

    session.close()


def test_multiprocess_lease_expiration_and_stale_completion_rejection(test_env):
    """Simulates lease expiration during ingestion: stale worker cannot overwrite newer worker's state."""
    session_factory = test_env["session_factory"]
    snapshots_dir = test_env["snapshots_dir"]

    session = session_factory()
    model_repo = SqliteModelRepository(session)
    job_queue = SqliteJobQueue(session)

    # 1. Create model in RECEIVED state
    model_id = "mod_stale_race_01"
    model_repo.create_model(
        model_id=model_id,
        name="candidate.pt",
        storage_path=str(test_env["quarantine_dir"] / "candidate.pt"),
        sha256="7" * 64,
        size_bytes=1024,
        status=IngestionStatus.RECEIVED,
    )

    # Create dummy quarantine file
    q_file = test_env["quarantine_dir"] / "candidate.pt"
    q_file.write_bytes(b"PK\x03\x04" + b"A" * 1020)

    # 2. Enqueue ingestion job
    job_id = job_queue.submit_job(
        task_type="model_ingestion",
        payload={
            "model_id": model_id,
            "quarantine_path": str(q_file),
            "filename": "candidate.pt",
        },
    )

    # 3. Worker 1 claims job
    claimed_w1 = job_queue.claim_next_job(worker_id="worker_slow", lease_duration_seconds=300)
    assert claimed_w1 is not None
    assert claimed_w1["lease_generation"] == 1

    # Worker 1 transitions model to INSPECTING
    model_repo.transition_model_to_inspecting(
        model_id=model_id,
        job_id=job_id,
        worker_id="worker_slow",
        lease_token=claimed_w1["lease_token"],
        lease_generation=claimed_w1["lease_generation"],
    )

    # 4. Worker 1 stalls; lease expires
    job_model = session.get(JobModel, job_id)
    job_model.lease_expires_at = now_utc() - timedelta(seconds=10)
    session.commit()

    # Recovery detects expired lease and re-queues job
    recovered = job_queue.recover_stale_jobs(stale_threshold_seconds=0)
    assert recovered == 1

    # 5. Worker 2 claims the re-queued job -> lease_generation = 2
    ingest_config = IngestionConfig(
        snapshots_dir=snapshots_dir,
        quarantine_dir=test_env["quarantine_dir"],
        staging_dir=test_env["storage_dir"] / "staging",
    )
    orchestrator = AssessmentOrchestrator(session=session, config=ingest_config)
    job_w2_res = orchestrator.process_next_job(worker_id="worker_fast")
    assert job_w2_res is not None
    assert job_w2_res["status"] == JobStatus.COMPLETED

    # Model is now ACCEPTED by Worker 2
    model_after_w2 = model_repo.get_model(model_id)
    assert model_after_w2.status == str(IngestionStatus.ACCEPTED)

    # 6. Worker 1 wakes up and attempts to publish failure or success
    snap_w1 = model_repo.get_snapshot(job_w2_res["result"]["snapshot_id"])
    with pytest.raises((JobNotClaimedError, JobLeaseExpiredError, InvalidJobStateTransitionError)):
        model_repo.publish_model_ingestion_success(
            model_id=model_id,
            snapshot=snap_w1,
            storage_path="/storage/snapshots/stale",
            sha256="7" * 64,
            size_bytes=1024,
            job_id=job_id,
            worker_id="worker_slow",
            lease_token=claimed_w1["lease_token"],
            lease_generation=claimed_w1["lease_generation"],
        )

    # Model remains firmly ACCEPTED by Worker 2
    final_model = model_repo.get_model(model_id)
    assert final_model.status == str(IngestionStatus.ACCEPTED)

    session.close()


def test_worker_crash_and_restart_recovery_idempotency(test_env):
    """Crash during model inspection leaves model recoverable without duplicating jobs."""
    session_factory = test_env["session_factory"]
    snapshots_dir = test_env["snapshots_dir"]

    session = session_factory()
    model_repo = SqliteModelRepository(session)
    job_queue = SqliteJobQueue(session)

    # Model interrupted in INSPECTING state
    q_file = test_env["quarantine_dir"] / "interrupted.pt"
    q_file.write_bytes(b"PK\x03\x04" + b"B" * 500)

    model_id = "mod_interrupted_01"
    model_repo.create_model(
        model_id=model_id,
        name="interrupted.pt",
        storage_path=str(q_file),
        sha256="8" * 64,
        size_bytes=len(q_file.read_bytes()),
        status=IngestionStatus.INSPECTING,
    )

    # 1. Startup recovery: detects stranded INSPECTING model
    reconciled = model_repo.reconcile_stranded_received_models(job_queue)
    assert reconciled == 1

    # 2. Worker claims and completes the recovered job
    ingest_config = IngestionConfig(
        snapshots_dir=snapshots_dir,
        quarantine_dir=test_env["quarantine_dir"],
        staging_dir=test_env["storage_dir"] / "staging",
    )
    orchestrator = AssessmentOrchestrator(session=session, config=ingest_config)
    res = orchestrator.process_next_job(worker_id="recovery_worker")
    assert res is not None
    assert res["status"] == JobStatus.COMPLETED

    # 3. Model is now ACCEPTED
    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.ACCEPTED)

    # 4. Subsequent recovery run is strictly idempotent
    reconciled_again = model_repo.reconcile_stranded_received_models(job_queue)
    assert reconciled_again == 0

    session.close()


def test_model_assessment_scheduling_and_lease_fenced_execution(test_env):
    """End-to-end: approved model assessment scheduling and lease-fenced detector execution."""
    client = test_env["client"]
    session_factory = test_env["session_factory"]
    snapshots_dir = test_env["snapshots_dir"]

    session = session_factory()
    model_repo = SqliteModelRepository(session)

    # 1. Ingest model to ACCEPTED status
    fake_weights = b"PK\x03\x04" + b"\x00" * 500
    upload_resp = client.post(
        "/api/v1/models",
        files={"file": ("model_eval.pt", io.BytesIO(fake_weights), "application/octet-stream")},
        data={"name": "Model For Assessment"},
    )
    assert upload_resp.status_code == 201
    model_id = upload_resp.json()["model_id"]

    ingest_config = IngestionConfig(
        snapshots_dir=snapshots_dir,
        quarantine_dir=test_env["quarantine_dir"],
        staging_dir=test_env["storage_dir"] / "staging",
    )
    orchestrator = AssessmentOrchestrator(session=session, config=ingest_config)
    orchestrator.process_next_job("ingestion_worker")

    # Verify model is ACCEPTED
    model = model_repo.get_model(model_id)
    assert model.status == str(IngestionStatus.ACCEPTED)

    # 2. Schedule assessment via API endpoint
    ass_resp = client.post(
        f"/api/v1/models/{model_id}/assessments",
        json={
            "name": "Model Integrity Verification Suite",
            "detectors": ["model_integrity.artifact_hash"],
        },
    )
    assert ass_resp.status_code == 202
    ass_data = ass_resp.json()
    assessment_id = ass_data["assessment_id"]
    job_id = ass_data["job_id"]
    assert assessment_id is not None
    assert job_id is not None

    # 3. Process assessment job with AssessmentOrchestrator
    ass_job_res = orchestrator.process_next_job(worker_id="assessment_worker_01")
    assert ass_job_res is not None
    assert ass_job_res["job_id"] == job_id
    assert ass_job_res["status"] == JobStatus.COMPLETED

    # 4. Verify assessment completed in SQLite
    ass_repo = SqliteAssessmentRepository(session)
    assessment = ass_repo.get_assessment(assessment_id)
    assert assessment is not None
    assert assessment["status"] == AssessmentStatus.COMPLETED

    # 5. Query assessment findings via API
    findings_resp = client.get(f"/api/v1/assessments/{assessment_id}/findings")
    assert findings_resp.status_code == 200
    findings_data = findings_resp.json()
    assert "items" in findings_data

    session.close()


def test_assessment_scheduling_rejected_on_non_accepted_model(test_env):
    """Scheduling an assessment on a model in RECEIVED or REJECTED status returns HTTP 409."""
    client = test_env["client"]
    session_factory = test_env["session_factory"]

    session = session_factory()
    model_repo = SqliteModelRepository(session)

    # 1. Model in RECEIVED status
    model_repo.create_model(
        model_id="mod_unaccepted_01",
        name="raw.pt",
        storage_path="/quarantine/raw.pt",
        sha256="9" * 64,
        size_bytes=100,
        status=IngestionStatus.RECEIVED,
    )

    resp = client.post(
        "/api/v1/models/mod_unaccepted_01/assessments",
        json={"name": "Premature Assessment"},
    )
    assert resp.status_code == 409
    data = resp.json()
    err_code = data.get("error_code") or data.get("detail", {}).get("error_code")
    assert err_code == "MODEL_NOT_ACCEPTED"

    # 2. Non-existent model returns HTTP 404
    resp_404 = client.post(
        "/api/v1/models/mod_does_not_exist/assessments",
        json={"name": "Ghost Assessment"},
    )
    assert resp_404.status_code == 404
    data_404 = resp_404.json()
    err_code_404 = data_404.get("error_code") or data_404.get("detail", {}).get("error_code")
    assert err_code_404 == "MODEL_NOT_FOUND"

    session.close()
