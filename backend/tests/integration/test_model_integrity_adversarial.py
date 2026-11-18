"""Adversarial and end-to-end integration tests for Model Integrity Engine.

Validates complete workflow: model ingestion, snapshot creation, isolated inspection,
adversarial exploit blocking, non-finite weight detection, architecture tampering,
and lease-fenced result publication.
"""

import hashlib
import pickle
from pathlib import Path

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.job_repository import (
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    JobStatus,
    SeverityLevel,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.artifact import ModelArtifactManager
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.services.orchestrator import AssessmentOrchestrator


@pytest.fixture(autouse=True)
def mock_verified_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates verified container isolation for end-to-end integration and adversarial tests."""
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
    db_file = tmp_path / "test_vg.db"
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


def test_scenario_01_clean_model_assessment_end_to_end(
    db_session: Session, test_config: IngestionConfig, tmp_path: Path
) -> None:
    """Scenario 1: End-to-end clean model ingestion, snapshotting, and 11-detector assessment."""
    # 1. Prepare candidate model
    model = SimpleCNN()
    model_src = tmp_path / "simple_model.pt"
    torch.save(model.state_dict(), model_src)

    hasher = hashlib.sha256()
    with open(model_src, "rb") as f:
        hasher.update(f.read())
    expected_sha256 = hasher.hexdigest()

    # 2. Stage candidate in quarantine
    quarantine_model = test_config.quarantine_dir / "quarantine_simple.pt"
    test_config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_model.write_bytes(model_src.read_bytes())

    # 3. Create model record and enqueue ingestion task
    model_repo = SqliteModelRepository(db_session)
    job_queue = SqliteJobQueue(db_session)

    model_id = "mod_e2e_clean"
    model_repo.create_model(
        model_id=model_id,
        name="Clean SimpleCNN Checkpoint",
        storage_path=str(quarantine_model),
        sha256=expected_sha256,
        size_bytes=quarantine_model.stat().st_size,
    )

    job_queue.submit_job(
        task_type="model_ingestion",
        payload={
            "model_id": model_id,
            "quarantine_path": str(quarantine_model),
            "filename": "simple_model.pt",
            "expected_sha256": expected_sha256,
        },
    )

    # 4. Orchestrator executes model ingestion
    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    job_res = orchestrator.process_next_job(worker_id="worker_alpha")

    assert job_res is not None
    assert job_res["status"] == JobStatus.COMPLETED.value
    snap_id = job_res["result"]["snapshot_id"]

    # 5. Schedule assessment for the ingested model snapshot
    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_e2e_clean_001"
    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Clean Model Assessment",
        snapshot_id=snap_id,
        model_id=model_id,
        config={
            "expected_sha256": expected_sha256,
            "expected_architecture": "simple_cnn",
        },
    )

    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snap_id,
        },
    )

    # 6. Orchestrator executes model assessment
    assess_res = orchestrator.process_next_job(worker_id="worker_beta")

    assert assess_res is not None
    assert assess_res["status"] == JobStatus.COMPLETED.value

    # 7. Verify assessment results
    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.COMPLETED
    assert (
        len(ass["runs"]) == 11
    )  # All 11 model integrity detectors executed (4 inspection + 7 behavioral/latent)

    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) == 0  # Clean model produces zero findings


def _persist_model_snapshot(db_session: Session, ingest_result) -> None:
    model_repo = SqliteModelRepository(db_session)
    snap = ingest_result.snapshot
    model_repo.create_model(
        model_id=ingest_result.model_id,
        name=ingest_result.filename,
        storage_path=str(ingest_result.storage_path),
        sha256=ingest_result.sha256,
        size_bytes=ingest_result.size_bytes,
    )
    model_repo.create_snapshot(
        snapshot_id=snap.snapshot_id,
        model_id=ingest_result.model_id,
        manifest_sha256=snap.manifest_sha256,
        storage_path=str(ingest_result.snapshot_dir),
        file_count=snap.file_count,
        total_size_bytes=snap.total_size_bytes,
        asset_type=snap.asset_type,
        schema_version=snap.schema_version,
        metadata=snap.metadata,
    )


def test_scenario_02_malicious_pickle_payload_blocked(
    db_session: Session, test_config: IngestionConfig, tmp_path: Path
) -> None:
    """Scenario 2: Malicious pickle bytecode exploit payload is blocked by weights_only=True."""
    exploit_file = tmp_path / "exploit_model.pt"

    class ArbitraryExecutionPayload:
        def __reduce__(self):
            import subprocess

            return (subprocess.check_output, (["calc.exe"],))

    with open(exploit_file, "wb") as f:
        pickle.dump(ArbitraryExecutionPayload(), f)

    mgr = ModelArtifactManager(
        ModelIntegrityConfig(
            models_dir=tmp_path / "models",
            snapshots_dir=test_config.snapshots_dir,
        )
    )
    ingest_result = mgr.ingest_model(exploit_file, filename="exploit_model.pt")
    _persist_model_snapshot(db_session, ingest_result)

    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_exploit_001"
    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Exploit Test Assessment",
        snapshot_id=ingest_result.snapshot.snapshot_id,
        model_id=ingest_result.model_id,
        config={},
    )

    job_queue = SqliteJobQueue(db_session)
    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": ingest_result.snapshot.snapshot_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    orchestrator.process_next_job(worker_id="worker_gamma")

    # The checkpoint structure detector catches the unpickler error and creates a CRITICAL finding
    findings = assessment_repo.get_findings(assessment_id)
    assert len(findings) >= 1
    assert any(
        f.severity == SeverityLevel.CRITICAL and f.category == FindingCategory.STRUCTURAL_DEFECT
        for f in findings
    )


def test_scenario_03_nan_weights_detected(
    db_session: Session, test_config: IngestionConfig, tmp_path: Path
) -> None:
    """Scenario 3: Checkpoint containing NaN weights triggers HIGH severity defect finding."""
    nan_model_file = tmp_path / "nan_weights_model.pt"
    t = torch.randn(8, 8)
    t[1, 1] = float("nan")
    t[2, 2] = float("nan")
    torch.save({"fc.weight": t}, nan_model_file)

    mgr = ModelArtifactManager(
        ModelIntegrityConfig(
            models_dir=tmp_path / "models",
            snapshots_dir=test_config.snapshots_dir,
        )
    )
    ingest_result = mgr.ingest_model(nan_model_file, filename="nan_weights_model.pt")
    _persist_model_snapshot(db_session, ingest_result)

    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_nan_001"
    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="NaN Weights Assessment",
        snapshot_id=ingest_result.snapshot.snapshot_id,
        model_id=ingest_result.model_id,
        config={},
    )

    job_queue = SqliteJobQueue(db_session)
    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": ingest_result.snapshot.snapshot_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    orchestrator.process_next_job(worker_id="worker_delta")

    findings = assessment_repo.get_findings(assessment_id)
    assert any(
        f.category == FindingCategory.STRUCTURAL_DEFECT and "Non-Finite" in f.title
        for f in findings
    )


def test_scenario_04_architecture_mismatch_detected(
    db_session: Session, test_config: IngestionConfig, tmp_path: Path
) -> None:
    """Scenario 4: Checkpoint deviating from expected architecture triggers incompatibility finding."""
    tampered_file = tmp_path / "tampered_arch.pt"
    # Create SimpleCNN but delete fc2 and add a backdoor layer
    model = SimpleCNN()
    s = dict(model.state_dict())
    del s["fc2.weight"]
    s["trojan_trigger.weight"] = torch.randn(1, 1)
    torch.save(s, tampered_file)

    mgr = ModelArtifactManager(
        ModelIntegrityConfig(
            models_dir=tmp_path / "models",
            snapshots_dir=test_config.snapshots_dir,
        )
    )
    ingest_result = mgr.ingest_model(tampered_file, filename="tampered_arch.pt")
    _persist_model_snapshot(db_session, ingest_result)

    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_tampered_arch_001"
    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Architecture Tampering Assessment",
        snapshot_id=ingest_result.snapshot.snapshot_id,
        model_id=ingest_result.model_id,
        config={"expected_architecture": "simple_cnn"},
    )

    job_queue = SqliteJobQueue(db_session)
    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": ingest_result.snapshot.snapshot_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    orchestrator.process_next_job(worker_id="worker_epsilon")

    findings = assessment_repo.get_findings(assessment_id)
    assert any("Architecture Incompatibility" in f.title for f in findings)


def test_scenario_05_fenced_publication_rejects_missing_credentials(
    db_session: Session, tmp_path: Path
) -> None:
    """Scenario 5: Publication without mandatory worker credentials is strictly rejected (F-02)."""
    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_fenced_001"

    # Direct publication attempt without job credentials must raise JobNotClaimedError
    with pytest.raises(JobNotClaimedError) as exc_info:
        assessment_repo.publish_assessment_results(
            assessment_id=assessment_id,
            detector_runs=[],
            findings=[],
            evidence_records=[],
            final_status=AssessmentStatus.COMPLETED,
            job_id=None,
            worker_id=None,
            lease_token=None,
            lease_generation=None,
        )
    assert "Publication requires all lease credentials" in str(exc_info.value)


def test_scenario_06_unhardened_host_orchestrator_marks_capability_unavailable(
    db_session: Session,
    test_config: IngestionConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 6: On an unhardened host without container sandbox, inspection detectors report UNAVAILABLE."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.UNHARDENED_HOST_SUBPROCESS,
            {"platform": "Windows"},
            "Unhardened host without container isolation.",
        ),
    )
    model = SimpleCNN()
    model_src = tmp_path / "model_unhardened.pt"
    torch.save(model.state_dict(), model_src)

    mgr = ModelArtifactManager(
        ModelIntegrityConfig(
            models_dir=tmp_path / "models",
            snapshots_dir=test_config.snapshots_dir,
        )
    )
    ingest_result = mgr.ingest_model(model_src, filename="model_unhardened.pt")
    _persist_model_snapshot(db_session, ingest_result)

    assessment_repo = SqliteAssessmentRepository(db_session)
    assessment_id = "asm_unhardened_001"
    assessment_repo.create_assessment(
        assessment_id=assessment_id,
        name="Unhardened Assessment",
        snapshot_id=ingest_result.snapshot.snapshot_id,
        model_id=ingest_result.model_id,
        config={
            "expected_sha256": ingest_result.sha256,
            "expected_architecture": "simple_cnn",
        },
    )

    job_queue = SqliteJobQueue(db_session)
    job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": ingest_result.snapshot.snapshot_id,
        },
    )

    orchestrator = AssessmentOrchestrator(session=db_session, config=test_config)
    orchestrator.process_next_job(worker_id="worker_unhardened")

    ass = assessment_repo.get_assessment(assessment_id)
    assert ass is not None
    # Artifact hash detector completes; inspection detectors report UNAVAILABLE
    run_statuses = {r["detector_id"]: r["status"] for r in ass["runs"]}
    assert run_statuses["model_integrity.artifact_hash"] in (
        CapabilityStatus.COMPLETED,
        CapabilityStatus.COMPLETED.value,
    )
    assert run_statuses["model_integrity.checkpoint_structure"] in (
        CapabilityStatus.UNAVAILABLE,
        CapabilityStatus.UNAVAILABLE.value,
    )
    assert run_statuses["model_integrity.state_dict_integrity"] in (
        CapabilityStatus.UNAVAILABLE,
        CapabilityStatus.UNAVAILABLE.value,
    )
    assert run_statuses["model_integrity.architecture_compatibility"] in (
        CapabilityStatus.UNAVAILABLE,
        CapabilityStatus.UNAVAILABLE.value,
    )
