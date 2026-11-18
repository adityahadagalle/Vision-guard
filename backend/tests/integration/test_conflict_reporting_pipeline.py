"""Integration tests for ADR-012 Multi-Detector Visible Conflict Reporting Pipeline.

Verifies end-to-end:
1. Model assessment through AssessmentOrchestrator detects conflicting findings.
2. Conflict findings with FindingCategory.CONFLICT are atomically persisted in SQLite.
3. Original detector findings remain completely unmuted, unmodified, and queryable.
4. Both original and conflict findings are retrievable via repository and REST API.
5. Filtering by category=CONFLICT accurately isolates multi-detector contradictions.
6. Clean matching models emit zero false conflicts.
"""

import hashlib
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.v1.router import api_v1_router
from backend.app.db.models import Base
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.session import get_db
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    JobStatus,
    SeverityLevel,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.conflict import ConflictType
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    """Provides an isolated SQLite test database."""
    db_file = tmp_path / "test_vg_conflict.db"
    engine = create_engine(f"sqlite:///{db_file}", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_config(tmp_path: Path) -> IngestionConfig:
    return IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        snapshots_dir=tmp_path / "snapshots",
    )


@pytest.fixture
def api_client(db_session: Session) -> TestClient:
    """Provides a TestClient wired to the test SQLite database."""
    app = FastAPI()
    app.include_router(api_v1_router, prefix="/api/v1")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _setup_model_for_conflict_test(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    model_id: str,
) -> tuple[str, str, str, str]:
    """Ingests a model checkpoint and returns (model_id, snapshot_id, file_sha256, fp_sha256)."""
    model = SimpleCNN()
    weights = model.state_dict()

    model_src = tmp_path / f"{model_id}.pt"
    torch.save(weights, model_src)

    hasher = hashlib.sha256()
    with open(model_src, "rb") as f:
        hasher.update(f.read())
    file_sha256 = hasher.hexdigest()

    test_config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_model = test_config.quarantine_dir / f"quarantine_{model_id}.pt"
    quarantine_model.write_bytes(model_src.read_bytes())

    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_repo.create_model(
        model_id=model_id,
        name=f"Model {model_id}",
        storage_path=str(quarantine_model),
        sha256=file_sha256,
        size_bytes=quarantine_model.stat().st_size,
    )

    job_queue.submit_job(
        task_type="model_ingestion",
        payload={
            "model_id": model_id,
            "quarantine_path": str(quarantine_model),
            "filename": f"{model_id}.pt",
            "expected_sha256": file_sha256,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    job_res = orchestrator.process_next_job(worker_id="ingest_worker")
    assert job_res is not None
    assert job_res["status"] == JobStatus.COMPLETED.value
    snap_id = job_res["result"]["snapshot_id"]

    # Extract state-dict fingerprint for reference
    from backend.app.modules.model_integrity.inspection.inspect_cli import (
        inspect_checkpoint,
    )

    insp = inspect_checkpoint(model_src)
    sd_fingerprint = insp["state_dict_fingerprint"]

    return model_id, snap_id, file_sha256, sd_fingerprint


def test_end_to_end_conflict_detection_and_api_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    api_client: TestClient,
) -> None:
    """Verifies end-to-end execution of assessment producing visible conflict finding.

    Scenario:
    - Candidate model has expected SHA-256 that does NOT match the file digest (tampered file hash).
    - But candidate model has expected state-dict fingerprint that DOES match (identical weights).
    - AssessmentOrchestrator must execute detectors, detect the contradiction, and emit
      a visible FindingCategory.CONFLICT finding without modifying or suppressing original findings.
    - Findings are queryable via repository and REST API.
    """
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    model_id, snap_id, actual_file_sha256, actual_sd_fp = _setup_model_for_conflict_test(
        db_session, test_config, tmp_path, "mod_conflict_e2e"
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_conflict_e2e_001"

    # Intentionally configure mismatched file digest but MATCHING state-dict fingerprint
    tampered_expected_file_sha256 = "f" * 64
    assessment_config = {
        "detectors": [
            "model_integrity.artifact_hash",
            "model_integrity.state_dict_integrity",
            "model_integrity.checkpoint_structure",
        ],
        "expected_sha256": tampered_expected_file_sha256,
        "expected_state_dict_fingerprint": actual_sd_fp,
    }

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Conflict Detection Assessment",
        snapshot_id=snap_id,
        config=assessment_config,
        model_id=model_id,
    )

    job_id = job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snap_id,
            "detectors": assessment_config["detectors"],
            "config": assessment_config,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    res = orchestrator.process_next_job(worker_id="eval_worker")

    assert res is not None
    assert res["status"] == JobStatus.COMPLETED.value
    assert res["job_id"] == job_id

    # 1. Verify in database repository
    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.COMPLETED.value

    findings, total = assessment_repo.get_findings_paginated(assessment_id)
    assert total >= 2

    # Original finding must be present
    hash_findings = [f for f in findings if f.detector_id == "model_integrity.artifact_hash"]
    assert len(hash_findings) == 1
    assert hash_findings[0].category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    assert hash_findings[0].severity == SeverityLevel.CRITICAL

    # Conflict finding must be present
    conflict_findings = [f for f in findings if f.category == FindingCategory.CONFLICT]
    assert len(conflict_findings) == 1
    cf = conflict_findings[0]
    assert cf.detector_id == "model_integrity.conflict"
    assert cf.status == CapabilityStatus.CONFLICTED
    assert cf.severity == SeverityLevel.MEDIUM
    assert "repackaging" in cf.description.lower()
    assert cf.details["conflict_type"] == ConflictType.HASH_MISMATCH_STATE_DICT_IDENTICAL

    # Verify conflict evidence records
    ev_records = assessment_repo.get_evidence_for_finding(cf.finding_id)
    assert len(ev_records) >= 1
    assert ev_records[0].finding_id == cf.finding_id

    # 2. Verify REST API Endpoint: GET /api/v1/assessments/{assessment_id}/findings
    response = api_client.get(f"/api/v1/assessments/{assessment_id}/findings")
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == total
    api_items = data["items"]

    api_conflicts = [item for item in api_items if item["category"] == "CONFLICT"]
    assert len(api_conflicts) == 1
    assert api_conflicts[0]["status"] == "CONFLICTED"
    assert api_conflicts[0]["severity"] == "MEDIUM"

    # 3. Verify REST API Filter by category: ?category=CONFLICT
    filter_resp = api_client.get(f"/api/v1/assessments/{assessment_id}/findings?category=CONFLICT")
    assert filter_resp.status_code == 200
    filter_data = filter_resp.json()
    assert filter_data["total_items"] == 1
    assert filter_data["items"][0]["category"] == "CONFLICT"
    assert filter_data["items"][0]["finding_id"] == cf.finding_id

    # 4. Verify REST API Filter by category: ?category=UNAUTHORIZED_MODEL_CHANGE
    orig_filter_resp = api_client.get(
        f"/api/v1/assessments/{assessment_id}/findings?category=UNAUTHORIZED_MODEL_CHANGE"
    )
    assert orig_filter_resp.status_code == 200
    orig_filter_data = orig_filter_resp.json()
    assert orig_filter_data["total_items"] == 1
    assert orig_filter_data["items"][0]["finding_id"] == hash_findings[0].finding_id


def test_clean_model_assessment_emits_no_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies that when all detectors agree the model is clean, zero conflict findings are produced."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    model_id, snap_id, actual_file_sha256, actual_sd_fp = _setup_model_for_conflict_test(
        db_session, test_config, tmp_path, "mod_clean_e2e"
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_clean_e2e_001"

    assessment_config = {
        "detectors": [
            "model_integrity.artifact_hash",
            "model_integrity.state_dict_integrity",
            "model_integrity.checkpoint_structure",
        ],
        "expected_sha256": actual_file_sha256,
        "expected_state_dict_fingerprint": actual_sd_fp,
    }

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Clean Assessment",
        snapshot_id=snap_id,
        config=assessment_config,
        model_id=model_id,
    )

    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snap_id,
            "detectors": assessment_config["detectors"],
            "config": assessment_config,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    res = orchestrator.process_next_job(worker_id="eval_worker")

    assert res is not None
    assert res["status"] == JobStatus.COMPLETED.value

    findings, total = assessment_repo.get_findings_paginated(assessment_id)
    assert total == 0
    conflict_findings, conflict_total = assessment_repo.get_findings_paginated(
        assessment_id, category=FindingCategory.CONFLICT.value
    )
    assert conflict_total == 0
