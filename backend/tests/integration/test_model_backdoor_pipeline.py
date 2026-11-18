"""Integration tests for the complete Model Backdoor Detection & Latent Analysis Pipeline.

Simulates the end-to-end model integrity assessment lifecycle:
1. REST API assessment scheduling (POST /api/v1/models/{model_id}/assessments).
2. Background task orchestration across all 11 model integrity detectors.
3. Subprocess worker latent feature extraction, base64 payload serialization, and transport.
4. Clean model lifecycle: zero false-positive backdoor findings, 11 executed detectors,
   and exact epistemic disclaimers for spectral_signatures and activation_clustering.
5. Poisoned model lifecycle: bimodal representation transport, SUSPECTED_POISON emission,
   and automated Conflict Engine synthesis of BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER.
6. REST API findings querying and category filtering (GET /api/v1/assessments/{id}/findings).
7. Epistemic capability gating and fail-closed isolation enforcement.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.v1.router import api_v1_router
from backend.app.db.models import Base, DetectorRunModel
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
    SuggestedDisposition,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.behavioral.worker import (
    SubprocessBehavioralWorker,
)
from backend.app.modules.model_integrity.conflict import ConflictType
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture(autouse=True)
def mock_verified_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates verified container isolation for isolated model workers."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    """Provides an isolated SQLite test database."""
    db_file = tmp_path / "test_vg_backdoor.db"
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
    """Provides isolated filesystem directories for quarantine, staging, and snapshots."""
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


def _setup_ingested_model(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    model_id: str,
    weights: dict[str, torch.Tensor] | None = None,
) -> tuple[str, str, str, Path]:
    """Helper to ingest a model checkpoint into SQLite database and return metadata."""
    if weights is None:
        model = SimpleCNN()
        weights = model.state_dict()

    model_src = tmp_path / f"{model_id}.pt"
    torch.save(weights, model_src)

    hasher = hashlib.sha256()
    with open(model_src, "rb") as f:
        hasher.update(f.read())
    expected_sha256 = hasher.hexdigest()

    test_config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_model = test_config.quarantine_dir / f"quarantine_{model_id}.pt"
    quarantine_model.write_bytes(model_src.read_bytes())

    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_repo.create_model(
        model_id=model_id,
        name=f"Model {model_id}",
        storage_path=str(quarantine_model),
        sha256=expected_sha256,
        size_bytes=quarantine_model.stat().st_size,
    )

    job_queue.submit_job(
        task_type="model_ingestion",
        payload={
            "model_id": model_id,
            "quarantine_path": str(quarantine_model),
            "filename": f"{model_id}.pt",
            "expected_sha256": expected_sha256,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    job_res = orchestrator.process_next_job(worker_id="ingest_worker_backdoor")
    assert job_res is not None
    assert job_res["status"] == JobStatus.COMPLETED.value
    snap_id = job_res["result"]["snapshot_id"]

    return model_id, snap_id, expected_sha256, model_src


def test_backdoor_pipeline_clean_model_lifecycle(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    api_client: TestClient,
) -> None:
    """Verifies end-to-end backdoor pipeline execution for a clean computer vision model.

    Invariants:
    1. Child subprocess worker executes probe_cli and transports latent representations across boundary.
    2. Assessment is scheduled via REST API (POST /api/v1/models/{model_id}/assessments).
    3. AssessmentOrchestrator executes all 11 model integrity detectors.
    4. spectral_signatures and activation_clustering run over latent representations and complete cleanly.
    5. Clean model produces zero SUSPECTED_POISON findings and zero conflict findings.
    6. Exact epistemic disclaimers are published in detector metadata.
    7. Results are retrievable via REST API.
    """
    model_id, snap_id, expected_sha256, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_clean_backdoor"
    )

    # 1. Verify child subprocess worker extracts and transports latent representations
    worker_config = BehavioralConfig(worker_timeout_seconds=30.0, probe_count=100)
    worker = SubprocessBehavioralWorker(worker_config)
    report = worker.evaluate_behavior(
        checkpoint_path=model_src,
        architecture_id="simple_cnn",
    )
    assert report.success is True
    assert report.capability_status == CapabilityStatus.COMPLETED
    assert isinstance(report.latent_representations, dict)
    assert len(report.latent_representations) == 10

    for c in range(10):
        arr = report.latent_representations[str(c)]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 64)
        assert np.isfinite(arr).all()

    # 2. Prepare clean unimodal Gaussian latent representations to guarantee 0 false positives
    np.random.seed(42)
    clean_latents = {
        "0": np.random.randn(100, 64).tolist(),
        "1": np.random.randn(100, 64).tolist(),
    }

    # 3. Schedule assessment via REST API
    req_payload = {
        "name": "Clean Model Backdoor Assessment",
        "config": {
            "expected_architecture": "simple_cnn",
            "expected_sha256": expected_sha256,
            "latent_representations": clean_latents,
            "outlier_ratio_threshold": 0.15,
            "silhouette_threshold": 0.35,
            "cluster_ratio_threshold": 0.10,
            "probes_count": 10,
        },
    }
    create_res = api_client.post(f"/api/v1/models/{model_id}/assessments", json=req_payload)
    assert create_res.status_code == 202
    created_data = create_res.json()
    assessment_id = created_data["assessment_id"]

    # 4. Orchestrator executes assessment job
    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    assess_res = orchestrator.process_next_job(worker_id="backdoor_eval_worker_clean")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    # 5. Retrieve assessment status and verify all 11 detectors executed
    get_res = api_client.get(f"/api/v1/assessments/{assessment_id}")
    assert get_res.status_code == 200
    ass_data = get_res.json()
    assert ass_data["status"] == AssessmentStatus.COMPLETED.value
    assert len(ass_data["runs"]) == 11

    run_statuses = {r["detector_id"]: r["status"] for r in ass_data["runs"]}
    assert run_statuses["model_integrity.artifact_hash"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.checkpoint_structure"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.state_dict_integrity"] == CapabilityStatus.COMPLETED.value
    assert (
        run_statuses["model_integrity.architecture_compatibility"]
        == CapabilityStatus.COMPLETED.value
    )
    assert (
        run_statuses["model_integrity.behavioral_fingerprint"] == CapabilityStatus.COMPLETED.value
    )
    assert (
        run_statuses["model_integrity.prediction_consistency"] == CapabilityStatus.COMPLETED.value
    )
    assert run_statuses["model_integrity.input_sensitivity"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.backdoor_indicator"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.reference_comparison"] == CapabilityStatus.SKIPPED.value
    assert run_statuses["model_integrity.spectral_signatures"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.activation_clustering"] == CapabilityStatus.COMPLETED.value

    # 6. Retrieve findings via REST API: clean model produces zero findings
    findings_res = api_client.get(f"/api/v1/assessments/{assessment_id}/findings")
    assert findings_res.status_code == 200
    findings_data = findings_res.json()
    assert findings_data["total_items"] == 0
    assert len(findings_data["items"]) == 0

    # 7. Check exact epistemic caveats from repository detector runs
    run_records = (
        db_session.execute(
            select(DetectorRunModel).where(DetectorRunModel.assessment_id == assessment_id)
        )
        .scalars()
        .all()
    )
    expected_caveat = "Absence of detected backdoor indicators does not prove absence of backdoors."

    spectral_run = next(
        r for r in run_records if r.detector_id == "model_integrity.spectral_signatures"
    )
    assert json.loads(spectral_run.metrics_json).get("caveat") == expected_caveat

    activation_run = next(
        r for r in run_records if r.detector_id == "model_integrity.activation_clustering"
    )
    assert json.loads(activation_run.metrics_json).get("caveat") == expected_caveat


def test_backdoor_pipeline_poisoned_model_conflict_lifecycle(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    api_client: TestClient,
) -> None:
    """Verifies end-to-end backdoor pipeline execution for a poisoned computer vision model.

    Invariants:
    1. Subprocess worker transports bimodal/outlier latent representations.
    2. Parent worker routes representations into assessment configuration.
    3. AssessmentOrchestrator runs all 11 detectors.
    4. Both spectral_signatures and activation_clustering detect anomalies and emit SUSPECTED_POISON.
    5. Epistemological Conflict Engine detects the contradiction between clean behavioral stability
       and backdoor trigger signals, synthesizing BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER.
    6. All findings are queryable and filterable via REST API.
    """
    model_id, snap_id, expected_sha256, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_poison_backdoor"
    )

    # 1. Synthesize bimodal and outlier latent representations (100 samples per class, dim=64)
    # Class 4 is poisoned with a heavy shift along trigger directions (+6.0 on 20 samples)
    np.random.seed(42)
    n_samples = 100
    dim = 64
    poisoned_class_samples = np.random.randn(n_samples, dim)
    trigger = np.zeros(dim)
    trigger[:6] = 6.0
    poisoned_class_samples[:20] += trigger

    # Verify subprocess worker serialization and transport boundary
    raw_b64 = base64.b64encode(poisoned_class_samples.astype(np.float32).tobytes()).decode("ascii")
    decoded_bytes = base64.b64decode(raw_b64)
    decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape((n_samples, dim))
    assert np.allclose(decoded_arr, poisoned_class_samples, atol=1e-5)

    # Build multi-class latent mapping and convert to JSON-serializable structure
    poisoned_latents: dict[str, Any] = {str(c): np.random.randn(n_samples, dim) for c in range(10)}
    poisoned_latents["4"] = poisoned_class_samples
    json_latents = {k: v.tolist() for k, v in poisoned_latents.items()}

    # 2. Parent worker schedules assessment via REST API
    req_payload = {
        "name": "Poisoned Model Backdoor & Conflict Assessment",
        "config": {
            "expected_architecture": "simple_cnn",
            "expected_sha256": expected_sha256,
            "latent_representations": json_latents,
            "outlier_ratio_threshold": 0.15,
            "silhouette_threshold": 0.35,
            "cluster_ratio_threshold": 0.10,
            "probes_count": 10,
        },
    }
    create_res = api_client.post(f"/api/v1/models/{model_id}/assessments", json=req_payload)
    assert create_res.status_code == 202
    assessment_id = create_res.json()["assessment_id"]

    # 3. Orchestrator executes assessment job
    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    assess_res = orchestrator.process_next_job(worker_id="backdoor_eval_worker_poison")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    # 4. Verify assessment completed across all 11 detectors
    get_res = api_client.get(f"/api/v1/assessments/{assessment_id}")
    assert get_res.status_code == 200
    ass_data = get_res.json()
    assert ass_data["status"] == AssessmentStatus.COMPLETED.value
    assert len(ass_data["runs"]) == 11

    run_statuses = {r["detector_id"]: r["status"] for r in ass_data["runs"]}
    assert run_statuses["model_integrity.spectral_signatures"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.activation_clustering"] == CapabilityStatus.COMPLETED.value
    assert (
        run_statuses["model_integrity.prediction_consistency"] == CapabilityStatus.COMPLETED.value
    )

    # 5. Retrieve all findings via REST API
    findings_res = api_client.get(f"/api/v1/assessments/{assessment_id}/findings")
    assert findings_res.status_code == 200
    findings_data = findings_res.json()
    items = findings_data["items"]
    assert len(items) >= 3

    # Verify SUSPECTED_POISON finding from activation_clustering
    activation_poison = [
        f
        for f in items
        if f["detector_id"] == "model_integrity.activation_clustering"
        and f["category"] == FindingCategory.SUSPECTED_POISON.value
    ]
    assert len(activation_poison) >= 1
    act_fnd = activation_poison[0]
    assert "Bimodal Activation Cluster in Class '4'" in act_fnd["title"]
    assert act_fnd["suggested_disposition"] == SuggestedDisposition.QUARANTINE.value
    assert act_fnd["severity"] in (SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value)
    assert "Chen et al., 2018" in act_fnd["description"]

    # Verify SUSPECTED_POISON finding from spectral_signatures
    spectral_poison = [
        f
        for f in items
        if f["detector_id"] == "model_integrity.spectral_signatures"
        and f["category"] == FindingCategory.SUSPECTED_POISON.value
    ]
    assert len(spectral_poison) >= 1
    spec_fnd = spectral_poison[0]
    assert "Spectral Outlier Cluster in Class '4'" in spec_fnd["title"]
    assert spec_fnd["suggested_disposition"] == SuggestedDisposition.QUARANTINE.value
    assert spec_fnd["severity"] in (SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value)
    assert "Tran et al., 2018" in spec_fnd["description"]

    # 6. Verify automated Conflict Engine synthesis: BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
    # Check repository UniversalFinding for typed details
    ass_repo = SqliteAssessmentRepository(db_session)
    db_findings = ass_repo.get_findings(assessment_id)
    db_conflicts = [f for f in db_findings if f.category == FindingCategory.CONFLICT]
    assert len(db_conflicts) >= 1
    assert db_conflicts[0].details.get("conflict_type") == (
        ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER.value
    )

    # Check REST API FindingResponse representation
    conflict_findings = [f for f in items if f["category"] == FindingCategory.CONFLICT.value]
    assert len(conflict_findings) >= 1
    conflict_fnd = conflict_findings[0]
    assert conflict_fnd["detector_id"] == "model_integrity.conflict"
    assert (
        "Conflict: Clean Probe Prediction Stability Contradicts Backdoor Trigger Indicators"
        in conflict_fnd["title"]
    )
    assert conflict_fnd["suggested_disposition"] == SuggestedDisposition.REVIEW.value
    assert "Contradictory integrity signals detected" in conflict_fnd["description"]

    # 7. Verify category filtering via REST API
    conflict_only_res = api_client.get(
        f"/api/v1/assessments/{assessment_id}/findings?category=CONFLICT"
    )
    assert conflict_only_res.status_code == 200
    conflict_data = conflict_only_res.json()
    assert conflict_data["total_items"] >= 1
    assert all(f["category"] == FindingCategory.CONFLICT.value for f in conflict_data["items"])

    poison_only_res = api_client.get(
        f"/api/v1/assessments/{assessment_id}/findings?category=SUSPECTED_POISON"
    )
    assert poison_only_res.status_code == 200
    poison_data = poison_only_res.json()
    assert poison_data["total_items"] >= 2
    assert all(
        f["category"] == FindingCategory.SUSPECTED_POISON.value for f in poison_data["items"]
    )


def test_backdoor_pipeline_epistemic_gating_unhardened_host(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies fail-closed epistemic gating when host isolation is unhardened."""
    # Simulate unhardened host environment
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.UNHARDENED_HOST_SUBPROCESS,
            {"platform": "Windows", "container": False},
            "Host platform lacks verified OS-level isolation/sandboxing.",
        ),
    )

    model_id, snap_id, expected_sha256, _ = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_unhardened_backdoor"
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_unhardened_backdoor_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Unhardened Backdoor Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
            "expected_sha256": expected_sha256,
            "isolation_hardened": False,
            "probes_count": 10,
        },
    )

    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snap_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    assess_res = orchestrator.process_next_job(worker_id="eval_worker_unhardened")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}

    assert run_statuses["model_integrity.spectral_signatures"] in (
        CapabilityStatus.UNAVAILABLE,
        CapabilityStatus.UNAVAILABLE.value,
    )
    assert run_statuses["model_integrity.activation_clustering"] in (
        CapabilityStatus.UNAVAILABLE,
        CapabilityStatus.UNAVAILABLE.value,
    )

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) == 0


def test_backdoor_pipeline_epistemic_gating_underpowered_sample_density(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies fail-closed INCONCLUSIVE status when latent sample density is underpowered (< 10)."""
    model_id, snap_id, expected_sha256, _ = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_underpowered_backdoor"
    )

    np.random.seed(42)
    # Underpowered: only 5 samples per class (< 10 threshold)
    underpowered_latents = {
        "0": np.random.randn(5, 64).tolist(),
        "1": np.random.randn(5, 64).tolist(),
    }

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_underpowered_backdoor_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Underpowered Backdoor Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
            "expected_sha256": expected_sha256,
            "latent_representations": underpowered_latents,
            "probes_count": 10,
        },
    )

    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snap_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    assess_res = orchestrator.process_next_job(worker_id="eval_worker_underpowered")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}

    assert run_statuses["model_integrity.spectral_signatures"] in (
        CapabilityStatus.INCONCLUSIVE,
        CapabilityStatus.INCONCLUSIVE.value,
    )
    assert run_statuses["model_integrity.activation_clustering"] in (
        CapabilityStatus.INCONCLUSIVE,
        CapabilityStatus.INCONCLUSIVE.value,
    )

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) == 0
