"""Integration tests for the complete Model Behavioral Analysis & Backdoor Detection Pipeline.

Verifies end-to-end execution through AssessmentOrchestrator, SQLite persistence,
job lease fencing, and fail-closed capability gating across all 5 behavioral detectors.
"""

import hashlib
from pathlib import Path

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    IngestionStatus,
    JobStatus,
    SeverityLevel,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    db_file = tmp_path / "test_vg_behavioral.db"
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


def _setup_ingested_model(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    model_id: str,
    weights: dict[str, torch.Tensor] | None = None,
) -> tuple[str, str, Path]:
    """Helper to ingest a model checkpoint through orchestrator and produce a snapshot."""
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
    job_res = orchestrator.process_next_job(worker_id="ingest_worker")
    assert job_res is not None
    assert job_res["status"] == JobStatus.COMPLETED.value
    snap_id = job_res["result"]["snapshot_id"]

    return model_id, snap_id, model_src


def test_behavioral_pipeline_clean_model(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies end-to-end evaluation of clean model across all 11 detectors under verified isolation."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    model_id, snap_id, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_behavioral_clean"
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_behavioral_clean_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Behavioral Clean Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
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
    assess_res = orchestrator.process_next_job(worker_id="eval_worker_1")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.COMPLETED
    assert len(ass["runs"]) == 11

    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}
    assert (
        run_statuses["model_integrity.behavioral_fingerprint"] == CapabilityStatus.COMPLETED.value
    )
    assert (
        run_statuses["model_integrity.prediction_consistency"] == CapabilityStatus.COMPLETED.value
    )
    assert run_statuses["model_integrity.input_sensitivity"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.backdoor_indicator"] == CapabilityStatus.COMPLETED.value
    assert run_statuses["model_integrity.reference_comparison"] == CapabilityStatus.SKIPPED.value

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) == 0


def test_behavioral_pipeline_with_trusted_reference_agreement(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies that providing an identical reference model produces COMPLETED with 100% agreement."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    model_id, snap_id, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_ref_clean"
    )

    ref_weights = torch.load(model_src, weights_only=True)
    ref_file = tmp_path / "trusted_ref.pt"
    torch.save({"state_dict": ref_weights, "meta": "independent_ref_run"}, ref_file)
    ref_sha256 = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    model_repo = SqliteModelRepository(db_session)
    model_repo.create_model(
        model_id="mod_trusted_ref_001",
        name="Trusted Reference Model",
        storage_path=str(ref_file.resolve()),
        sha256=ref_sha256,
        size_bytes=ref_file.stat().st_size,
        status=IngestionStatus.ACCEPTED,
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_ref_clean_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Reference Agreement Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
            "reference_model_path": str(ref_file),
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
    assess_res = orchestrator.process_next_job(worker_id="eval_worker_2")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.COMPLETED

    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}
    assert run_statuses["model_integrity.reference_comparison"] == CapabilityStatus.COMPLETED.value

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) == 0


def test_behavioral_pipeline_with_divergent_reference_flags_finding(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies that a divergent reference model generates an UNAUTHORIZED_MODEL_CHANGE finding."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )

    # Force candidate to predict class 0
    candidate_model = SimpleCNN()
    cand_weights = candidate_model.state_dict()
    cand_weights["fc2.bias"] = torch.zeros(10)
    cand_weights["fc2.bias"][0] = 50.0

    model_id, snap_id, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_divergent_candidate", weights=cand_weights
    )

    # Force reference to predict class 9
    ref_model = SimpleCNN()
    ref_weights = ref_model.state_dict()
    ref_weights["fc2.bias"] = torch.zeros(10)
    ref_weights["fc2.bias"][9] = 50.0
    ref_file = tmp_path / "divergent_ref.pt"
    torch.save(ref_weights, ref_file)
    ref_sha256 = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    model_repo = SqliteModelRepository(db_session)
    model_repo.create_model(
        model_id="mod_divergent_ref_001",
        name="Divergent Reference Model",
        storage_path=str(ref_file.resolve()),
        sha256=ref_sha256,
        size_bytes=ref_file.stat().st_size,
        status=IngestionStatus.ACCEPTED,
    )

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_ref_diverge_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Reference Divergence Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
            "reference_model_path": str(ref_file),
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
    assess_res = orchestrator.process_next_job(worker_id="eval_worker_3")
    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.COMPLETED

    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}
    assert run_statuses["model_integrity.reference_comparison"] == CapabilityStatus.COMPLETED.value

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) >= 1
    diverge_finding = [
        f for f in findings if f.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    ]
    assert len(diverge_finding) >= 1
    assert diverge_finding[0].severity in (SeverityLevel.HIGH, SeverityLevel.MEDIUM)


def test_behavioral_pipeline_unhardened_host_fails_closed(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies that all behavioral and inspection detectors report UNAVAILABLE on unhardened hosts."""
    # Unmocked HostIsolationAssessor assesses the actual Windows / unconfined host
    model_id, snap_id, model_src = _setup_ingested_model(
        db_session, test_config, tmp_path, "mod_unhardened"
    )

    hasher = hashlib.sha256()
    with open(model_src, "rb") as f:
        hasher.update(f.read())
    expected_sha256 = hasher.hexdigest()

    assessment_repo = SqliteAssessmentRepository(db_session)
    job_queue = SqliteJobQueue(db_session)
    assessment_id = "asm_unhardened_001"

    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Unhardened Host Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_architecture": "simple_cnn",
            "expected_sha256": expected_sha256,
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
    assert ass["status"] == AssessmentStatus.COMPLETED

    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}

    # Artifact hash detector is pure byte hashing: COMPLETED
    assert run_statuses["model_integrity.artifact_hash"] in (
        CapabilityStatus.COMPLETED,
        CapabilityStatus.COMPLETED.value,
    )

    # 3 Inspection detectors: UNAVAILABLE
    for did in [
        "model_integrity.checkpoint_structure",
        "model_integrity.state_dict_integrity",
        "model_integrity.architecture_compatibility",
    ]:
        assert run_statuses[did] in (
            CapabilityStatus.UNAVAILABLE,
            CapabilityStatus.UNAVAILABLE.value,
        )

    # 4 Behavioral detectors without reference dependency: UNAVAILABLE
    for did in [
        "model_integrity.behavioral_fingerprint",
        "model_integrity.prediction_consistency",
        "model_integrity.input_sensitivity",
        "model_integrity.backdoor_indicator",
    ]:
        assert run_statuses[did] in (
            CapabilityStatus.UNAVAILABLE,
            CapabilityStatus.UNAVAILABLE.value,
        )

    # Reference comparison detector: SKIPPED when no reference model is supplied
    assert run_statuses["model_integrity.reference_comparison"] in (
        CapabilityStatus.SKIPPED,
        CapabilityStatus.SKIPPED.value,
    )
