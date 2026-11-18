"""Comprehensive End-to-End API and Worker Lifecycle Integration Tests for Phase 3 Slice 3C.

Covers the complete vertical lifecycle across Tasks 1, 2, 3, and 4:
1. Ingestion Upload (POST /api/v1/models) -> RECEIVED state -> quarantine storage.
2. Worker Ingestion Execution -> lease-fenced transition RECEIVED -> INSPECTING -> ACCEPTED.
3. Model Query & Detail (GET /api/v1/models, GET /api/v1/models/{id}) -> status, snapshots, metadata.
4. Assessment Scheduling (POST /api/v1/models/{id}/assessments) -> lifecycle guards, 202 Accepted.
5. Assessment Worker Execution -> real multi-detector analysis -> visible conflict evaluation.
6. Assessment Findings & Conflict Retrieval (GET /api/v1/assessments/{id}/findings).
7. Rejection Lifecycle -> corrupted uploads, REJECTED terminal immutability, blocked assessments.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import (
    InvalidModelStateTransitionError,
    SqliteModelRepository,
)
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.domain.entities.enums import (
    FindingCategory,
    IngestionStatus,
    JobStatus,
)
from backend.app.main import create_app
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.inspection.inspect_cli import inspect_checkpoint
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture
def lifecycle_env(tmp_path: Path):
    """Sets up an isolated database, file storage, settings, and TestClient."""
    db_file = tmp_path / "lifecycle_test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    data_dir = tmp_path / "data"
    quarantine_dir = data_dir / "quarantine" / "models"
    snapshots_dir = data_dir / "snapshots"
    staging_dir = data_dir / "staging"
    for d in [quarantine_dir, snapshots_dir, staging_dir]:
        d.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        database_url=db_url,
        data_dir=data_dir,
        max_model_upload_bytes=100 * 1024 * 1024,
        max_concurrent_uploads=3,
        global_quarantine_quota_bytes=500 * 1024 * 1024,
    )

    app = create_app(custom_settings=settings)

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield {
        "client": client,
        "session_factory": session_factory,
        "settings": settings,
        "data_dir": data_dir,
        "engine": engine,
    }

    engine.dispose()


def test_full_slice3c_success_lifecycle(
    lifecycle_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verifies complete end-to-end flow: Upload -> Ingest -> Query -> Assess -> Conflict Detect -> Retrieve."""
    client: TestClient = lifecycle_env["client"]
    session_factory = lifecycle_env["session_factory"]
    data_dir: Path = lifecycle_env["data_dir"]

    # Ensure worker subprocesses can execute in hardened isolation mode
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    # -------------------------------------------------------------------------
    # STEP 1: Upload candidate PyTorch model
    # -------------------------------------------------------------------------
    model = SimpleCNN()
    weights = model.state_dict()
    model_file_temp = tmp_path / "simple_cnn.pt"
    torch.save(weights, model_file_temp)
    model_bytes = model_file_temp.read_bytes()

    # Pre-calculate state-dict fingerprint for expected reference in assessment
    insp = inspect_checkpoint(model_file_temp)
    actual_sd_fp = insp["state_dict_fingerprint"]

    upload_resp = client.post(
        "/api/v1/models",
        files={
            "file": (
                "candidate_classifier.pt",
                io.BytesIO(model_bytes),
                "application/octet-stream",
            )
        },
        data={"name": "Candidate SimpleCNN"},
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    model_id = upload_data["model_id"]
    ingestion_job_id = upload_data["job_id"]
    assert upload_data["status"] == "RECEIVED"
    assert upload_data["name"] == "Candidate SimpleCNN"
    assert ingestion_job_id is not None

    # Verify quarantine file exists in upload subfolder
    quarantine_files = list((data_dir / "quarantine" / "models").rglob("*.pt"))
    assert len(quarantine_files) == 1

    # -------------------------------------------------------------------------
    # STEP 2: Execute Ingestion via Orchestrator Worker
    # -------------------------------------------------------------------------
    ingestion_cfg = IngestionConfig(
        quarantine_dir=data_dir / "quarantine" / "models",
        staging_dir=data_dir / "staging",
        snapshots_dir=data_dir / "snapshots",
    )

    with session_factory() as session:
        orchestrator = AssessmentOrchestrator(session=session, config=ingestion_cfg)
        processed = orchestrator.process_next_job(worker_id="worker_ingest_01")
        assert processed is not None
        assert processed["status"] == JobStatus.COMPLETED

        # Verify job marked completed
        job_queue = SqliteJobQueue(session)
        job = job_queue.get_job(ingestion_job_id)
        assert job["status"] == JobStatus.COMPLETED
        snapshot_id = job["result"]["snapshot_id"]

    # -------------------------------------------------------------------------
    # STEP 3: Query Model Details and List via API
    # -------------------------------------------------------------------------
    detail_resp = client.get(f"/api/v1/models/{model_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["status"] == "ACCEPTED"
    assert detail_data["snapshot_id"] == snapshot_id
    assert detail_data["assessments_count"] == 0

    list_resp = client.get("/api/v1/models?status=ACCEPTED")
    assert list_resp.status_code == 200
    list_items = list_resp.json()["items"]
    assert any(m["model_id"] == model_id for m in list_items)

    rejected_list_resp = client.get("/api/v1/models?status=REJECTED")
    assert not any(m["model_id"] == model_id for m in rejected_list_resp.json()["items"])

    # -------------------------------------------------------------------------
    # STEP 4: Trigger Model Assessment with Contradictory Reference Expectations
    # File hash expected differs ("f"*64), but state-dict expected matches actual
    # -------------------------------------------------------------------------
    assessment_resp = client.post(
        f"/api/v1/models/{model_id}/assessments",
        json={
            "detectors": [
                "model_integrity.artifact_hash",
                "model_integrity.state_dict_integrity",
                "model_integrity.checkpoint_structure",
            ],
            "config": {
                "expected_sha256": "f" * 64,
                "expected_state_dict_fingerprint": actual_sd_fp,
            },
        },
    )
    assert assessment_resp.status_code == 202
    assessment_data = assessment_resp.json()
    assessment_id = assessment_data["assessment_id"]
    assessment_job_id = assessment_data["job_id"]
    assert assessment_data["status"] == "PENDING"

    # -------------------------------------------------------------------------
    # STEP 5: Execute Assessment via Orchestrator Worker
    # -------------------------------------------------------------------------
    with session_factory() as session:
        orchestrator = AssessmentOrchestrator(session=session, config=ingestion_cfg)
        processed = orchestrator.process_next_job(worker_id="worker_assess_01")
        assert processed is not None
        assert processed["status"] == JobStatus.COMPLETED

        job_queue = SqliteJobQueue(session)
        job = job_queue.get_job(assessment_job_id)
        assert job["status"] == JobStatus.COMPLETED

    # -------------------------------------------------------------------------
    # STEP 6: Retrieve Findings via API (Unfiltered & Filtered by CONFLICT)
    # -------------------------------------------------------------------------
    all_findings_resp = client.get(f"/api/v1/assessments/{assessment_id}/findings")
    assert all_findings_resp.status_code == 200
    all_findings = all_findings_resp.json()["items"]

    # Must contain original hash mismatch finding AND conflict finding
    assert len(all_findings) >= 2
    categories = [f["category"] for f in all_findings]
    assert FindingCategory.UNAUTHORIZED_MODEL_CHANGE in categories
    assert FindingCategory.CONFLICT in categories

    # Query filtered specifically to CONFLICT
    conflict_resp = client.get(f"/api/v1/assessments/{assessment_id}/findings?category=CONFLICT")
    assert conflict_resp.status_code == 200
    conflict_items = conflict_resp.json()["items"]
    assert len(conflict_items) == 1
    assert conflict_items[0]["category"] == FindingCategory.CONFLICT
    assert conflict_items[0]["detector_id"] == "model_integrity.conflict"
    assert "Identical State-Dict Fingerprint" in conflict_items[0]["title"]
    assert "Contradictory integrity signals detected" in conflict_items[0]["description"]
    assert conflict_items[0]["status"] == "CONFLICTED"

    # Verify model details reflects 1 completed assessment
    updated_model_resp = client.get(f"/api/v1/models/{model_id}")
    assert updated_model_resp.json()["assessments_count"] == 1


def test_slice3c_rejection_terminal_immutability(lifecycle_env):
    """Verifies that failed model ingestion marks model REJECTED and enforces terminal immutability."""
    client: TestClient = lifecycle_env["client"]
    session_factory = lifecycle_env["session_factory"]

    # 1. Upload candidate model file
    upload_resp = client.post(
        "/api/v1/models",
        files={
            "file": (
                "corrupted.pt",
                io.BytesIO(b"PK\x03\x04corrupted_payload"),
                "application/octet-stream",
            )
        },
    )
    assert upload_resp.status_code == 201
    model_id = upload_resp.json()["model_id"]
    job_id = upload_resp.json()["job_id"]

    # 2. Simulate worker ingestion failure
    with session_factory() as session:
        model_repo = SqliteModelRepository(session)
        job_queue = SqliteJobQueue(session)

        claimed = job_queue.claim_next_job(worker_id="worker_fail_01", lease_duration_seconds=300)
        assert claimed is not None
        assert claimed["job_id"] == job_id

        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id="worker_fail_01",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )

        model_repo.publish_model_ingestion_failure(
            model_id=model_id,
            error_message="Checkpoint file contains malformed ZIP structure.",
            job_id=job_id,
            worker_id="worker_fail_01",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )

    # 3. Verify Model Details reflects REJECTED state
    detail_resp = client.get(f"/api/v1/models/{model_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "REJECTED"
    assert detail_resp.json()["snapshot_id"] is None

    # 4. Attempting to schedule an assessment on REJECTED model must return 409
    ass_resp = client.post(f"/api/v1/models/{model_id}/assessments", json={})
    assert ass_resp.status_code == 409
    assert ass_resp.json()["error_code"] == "MODEL_NOT_ACCEPTED"

    # 5. Direct database repository check: Terminal state immutability
    with session_factory() as session:
        model_repo = SqliteModelRepository(session)
        job_queue = SqliteJobQueue(session)

        new_job_id = job_queue.submit_job(
            task_type="model_ingestion",
            payload={"model_id": model_id},
        )
        new_claim = job_queue.claim_next_job(worker_id="worker_fail_02", lease_duration_seconds=300)
        assert new_claim is not None
        assert new_claim["job_id"] == new_job_id

        with pytest.raises(InvalidModelStateTransitionError, match="REJECTED"):
            model_repo.transition_model_to_inspecting(
                model_id=model_id,
                job_id=new_job_id,
                worker_id="worker_fail_02",
                lease_token=new_claim["lease_token"],
                lease_generation=new_claim["lease_generation"],
            )


def test_slice3c_assessment_trigger_guards_across_states(lifecycle_env):
    """Verifies that assessment initiation strictly validates lifecycle states and snapshot availability."""
    client: TestClient = lifecycle_env["client"]
    session_factory = lifecycle_env["session_factory"]

    # 1. Nonexistent model -> 404
    assert client.post("/api/v1/models/nonexistent_model/assessments", json={}).status_code == 404

    # 2. Model in RECEIVED status -> 409 MODEL_NOT_ACCEPTED
    upload_resp = client.post(
        "/api/v1/models",
        files={
            "file": (
                "uninspected.pt",
                io.BytesIO(b"dummy_bytes"),
                "application/octet-stream",
            )
        },
    )
    rec_model_id = upload_resp.json()["model_id"]
    rec_ass_resp = client.post(f"/api/v1/models/{rec_model_id}/assessments", json={})
    assert rec_ass_resp.status_code == 409
    assert rec_ass_resp.json()["error_code"] == "MODEL_NOT_ACCEPTED"

    # 3. Model in INSPECTING status -> 409 MODEL_NOT_ACCEPTED
    with session_factory() as session:
        model_repo = SqliteModelRepository(session)
        job_queue = SqliteJobQueue(session)
        claimed = job_queue.claim_next_job(worker_id="worker_guard", lease_duration_seconds=300)
        assert claimed is not None
        model_repo.transition_model_to_inspecting(
            model_id=rec_model_id,
            job_id=claimed["job_id"],
            worker_id="worker_guard",
            lease_token=claimed["lease_token"],
            lease_generation=claimed["lease_generation"],
        )

    insp_ass_resp = client.post(f"/api/v1/models/{rec_model_id}/assessments", json={})
    assert insp_ass_resp.status_code == 409
    assert insp_ass_resp.json()["error_code"] == "MODEL_NOT_ACCEPTED"

    # 4. Model in ACCEPTED status but with no sealed snapshot -> 409 NO_SNAPSHOT_AVAILABLE
    with session_factory() as session:
        model_repo = SqliteModelRepository(session)
        orphan_model_id = "mod_orphan_no_snap"
        model_repo.create_model(
            model_id=orphan_model_id,
            name="orphan.pt",
            storage_path="/tmp/orphan.pt",
            sha256="0" * 64,
            size_bytes=100,
            status=IngestionStatus.ACCEPTED,
        )

    orphan_ass_resp = client.post(f"/api/v1/models/{orphan_model_id}/assessments", json={})
    assert orphan_ass_resp.status_code == 409
    assert orphan_ass_resp.json()["error_code"] == "NO_SNAPSHOT_AVAILABLE"
