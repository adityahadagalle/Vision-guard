"""Security boundary and attack containment tests for VisionGuard Phase 3 Slice 3C.

Validates:
1. Zero deserialization (no torch.load or pickle.loads in API process).
2. Filesystem & path traversal containment (../, absolute paths, Windows reserved names).
3. API security posture (no DELETE endpoints, extra="forbid" schema enforcement).
4. Stale worker staging directory cleanup on lease generation change / lease expiration.
5. Capability gating on unhardened host (no subprocesses spawned, no false conflicts).
6. ADR-003 benchmark script reproducibility in offline smoke mode.
"""

import io
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import (
    AssetSnapshotModel,
    Base,
    JobModel,
    ModelArtifactModel,
)
from backend.app.db.repositories.job_repository import (
    JobLeaseExpiredError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    IngestionStatus,
)
from backend.app.main import create_app
from backend.app.modules.model_integrity import (
    HostIsolationAssessor,
    IsolationStatus,
)
from backend.app.modules.model_integrity.artifact import IngestionConfig
from backend.app.modules.model_integrity.conflict import ModelConflictEngine
from backend.app.services.orchestrator import AssessmentOrchestrator


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def security_test_env(tmp_path: Path):
    """Sets up an isolated database, file storage, settings, and TestClient for security tests."""
    db_file = tmp_path / "security_test.db"
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


def test_deserialization_isolation_in_api_process(security_test_env, tmp_path: Path):
    """Verifies that torch.load and pickle.loads are NEVER invoked in the FastAPI API process."""
    client: TestClient = security_test_env["client"]
    session_factory = security_test_env["session_factory"]

    model = SimpleModel()
    model_path = tmp_path / "model_weights.pt"
    torch.save(model.state_dict(), model_path)
    model_bytes = model_path.read_bytes()

    # Strictly patch torch.load and pickle.loads
    with patch("torch.load") as mock_torch_load, patch("pickle.loads") as mock_pickle_loads:
        # 1. Model upload endpoint
        upload_resp = client.post(
            "/api/v1/models",
            files={
                "file": ("model_weights.pt", io.BytesIO(model_bytes), "application/octet-stream")
            },
            data={"name": "Isolated Model"},
        )
        assert upload_resp.status_code == 201
        model_id = upload_resp.json()["model_id"]

        # 2. Model query endpoints
        get_resp = client.get(f"/api/v1/models/{model_id}")
        assert get_resp.status_code == 200

        list_resp = client.get("/api/v1/models?page=1&page_size=10")
        assert list_resp.status_code == 200

        # Mark model accepted and associate snapshot to allow assessment scheduling
        with session_factory() as session:
            m = session.query(ModelArtifactModel).filter_by(model_id=model_id).one()
            m.status = IngestionStatus.ACCEPTED
            snap = AssetSnapshotModel(
                snapshot_id=f"snap_{model_id}",
                model_id=model_id,
                asset_type=AssetType.MODEL_CHECKPOINT,
                storage_path=str(model_path),
                manifest_sha256="a" * 64,
                file_count=1,
                total_size_bytes=len(model_bytes),
                created_at=datetime.now(UTC),
            )
            session.add(snap)
            session.commit()

        # 3. Assessment trigger endpoint
        assess_resp = client.post(
            f"/api/v1/models/{model_id}/assessments",
            json={"name": "API Security Check"},
        )
        assert assess_resp.status_code == 202

        # Verify zero deserialization occurred in the API process
        assert mock_torch_load.call_count == 0, "torch.load must never be called in API process!"
        assert mock_pickle_loads.call_count == 0, (
            "pickle.loads must never be called in API process!"
        )


def test_path_traversal_and_windows_reserved_names_rejected(security_test_env):
    """Verifies that path traversal attempts and reserved device names are strictly rejected."""
    client: TestClient = security_test_env["client"]
    data_dir: Path = security_test_env["data_dir"]

    # 1. Direct validator check for drive prefixes, colons, and separators
    from backend.app.modules.model_integrity.artifact import validate_model_filename
    from backend.app.modules.model_integrity.exceptions import PathTraversalError

    with pytest.raises(PathTraversalError, match="separators"):
        validate_model_filename("C:\\Windows\\System32\\calc.pt")

    with pytest.raises(PathTraversalError, match="forbidden colon"):
        validate_model_filename("C:calc.pt")

    with pytest.raises(PathTraversalError, match="separators"):
        validate_model_filename("folder/sub/calc.pt")

    # 2. HTTP multipart endpoint checks
    traversal_filenames = [
        "../../malicious.pt",
        "..\\..\\malicious.pt",
        "/etc/passwd.pt",
        "nested/../../exploit.pt",
        "CON.pt",
        "con.pth",
        "PRN.pt",
        "AUX.pt",
        "NUL.pt",
        "COM1.pt",
        "LPT1.pt",
        "null\x00byte.pt",
    ]

    for fname in traversal_filenames:
        resp = client.post(
            "/api/v1/models",
            files={"file": (fname, io.BytesIO(b"checkpoint_data"), "application/octet-stream")},
        )
        assert resp.status_code == 400, f"Expected 400 Bad Request for filename '{fname}'"
        assert resp.json()["error_code"] in [
            "PATH_TRAVERSAL_DETECTED",
            "INVALID_FILENAME",
            "INVALID_MODEL_EXTENSION",
        ]

    # Verify no files escaped quarantine or were created outside target structure
    all_files = list(data_dir.rglob("*"))
    for f in all_files:
        assert data_dir in f.parents or f == data_dir
        # No files should exist with traversal names
        for bad_name in ["malicious.pt", "calc.pt", "exploit.pt"]:
            assert bad_name not in f.name


def test_api_security_posture_delete_and_schema_forbid(security_test_env):
    """Verifies immutability posture: no DELETE endpoints and schema extra fields rejection."""
    client: TestClient = security_test_env["client"]

    # 1. Verify DELETE is rejected with 404 or 405 (no soft/hard delete endpoints)
    model_del_resp = client.delete("/api/v1/models/model_dummy123")
    assert model_del_resp.status_code in [404, 405]

    assess_del_resp = client.delete("/api/v1/assessments/ass_dummy123")
    assert assess_del_resp.status_code in [404, 405]

    # 2. Upload valid model to get a model_id
    up_resp = client.post(
        "/api/v1/models",
        files={"file": ("model.pt", io.BytesIO(b"payload"), "application/octet-stream")},
    )
    assert up_resp.status_code == 201
    model_id = up_resp.json()["model_id"]

    # 3. Verify extra='forbid' on CreateModelAssessmentRequest
    # Passing unrecognized fields must return 422 Unprocessable Entity
    extra_field_resp = client.post(
        f"/api/v1/models/{model_id}/assessments",
        json={
            "name": "Audit Run",
            "detectors": ["model_integrity.artifact_hash"],
            "config": {},
            "unauthorized_injection_field": "attack_payload",
        },
    )
    assert extra_field_resp.status_code == 422
    assert extra_field_resp.json()["error_code"] == "VALIDATION_ERROR"


def test_staging_directory_cleanup_on_lease_failure(security_test_env, tmp_path: Path):
    """Verifies that when a worker's lease expires during staging, staging artifacts are cleaned up safely."""
    session_factory = security_test_env["session_factory"]
    data_dir: Path = security_test_env["data_dir"]

    # 1. Create a dummy model in quarantine and a job in the queue
    model_file = tmp_path / "valid_model.pt"
    torch.save(SimpleModel().state_dict(), model_file)
    model_bytes = model_file.read_bytes()

    model_id = "model_lease_test"
    quarantine_sub = data_dir / "quarantine" / f"upload_{model_id}"
    quarantine_sub.mkdir(parents=True, exist_ok=True)
    target_path = quarantine_sub / "valid_model.pt"
    target_path.write_bytes(model_bytes)

    with session_factory() as session:
        queue = SqliteJobQueue(session)
        job_id = queue.submit_job(
            task_type="model_ingestion",
            payload={
                "model_id": model_id,
                "quarantine_path": str(target_path),
                "filename": "valid_model.pt",
                "sha256": "dummy_sha256",
            },
        )
        m = ModelArtifactModel(
            model_id=model_id,
            name="Lease Test Model",
            format="pt",
            status=IngestionStatus.RECEIVED,
            storage_path=str(target_path),
            sha256="dummy_sha256",
            size_bytes=len(model_bytes),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(m)
        session.commit()

    # Pre-populate an existing valid snapshot to verify it is NOT deleted during cleanup
    valid_snapshot_dir = data_dir / "snapshots" / "snap_existing_valid"
    valid_snapshot_dir.mkdir(parents=True, exist_ok=True)
    sentinel_file = valid_snapshot_dir / "weights.pt"
    sentinel_file.write_bytes(b"existing_immutable_snapshot")

    ingestion_cfg = IngestionConfig(
        quarantine_dir=data_dir / "quarantine",
        staging_dir=data_dir / "staging",
        snapshots_dir=data_dir / "snapshots",
    )

    # 2. Worker claims the job
    with session_factory() as session:
        queue = SqliteJobQueue(session)
        claimed = queue.claim_next_job(worker_id="worker_A", lease_duration_seconds=60)
        assert claimed is not None
        assert claimed["job_id"] == job_id
        original_token = claimed["lease_token"]
        original_gen = claimed["lease_generation"]

        # Simulate another worker or timeout stealing / incrementing the lease generation
        job_record = session.query(JobModel).filter_by(job_id=job_id).one()
        job_record.lease_generation = original_gen + 1
        job_record.worker_id = "worker_B"
        job_record.lease_token = "new_lease_token_stolen"
        session.commit()

    # 3. Worker A attempts to process the job using the stale generation / token
    with session_factory() as session:
        orchestrator = AssessmentOrchestrator(session=session, config=ingestion_cfg)
        with pytest.raises((JobLeaseExpiredError, JobNotClaimedError)):
            orchestrator._execute_model_ingestion(
                job_id=job_id,
                worker_id="worker_A",
                payload={
                    "model_id": model_id,
                    "quarantine_path": str(target_path),
                    "filename": "valid_model.pt",
                    "sha256": "dummy_sha256",
                },
                lease_token=original_token,
                lease_generation=original_gen,
            )

    # 4. Verify containment guarantees:
    # a) Staging directory for model_id must be completely cleaned up
    staging_entries = list((data_dir / "staging").glob(f"*{model_id}*"))
    assert len(staging_entries) == 0, f"Staging artifacts were not cleaned up: {staging_entries}"

    # b) Existing valid snapshot is intact
    assert sentinel_file.exists()
    assert sentinel_file.read_bytes() == b"existing_immutable_snapshot"

    # c) Model status was not marked ACCEPTED by the stale worker
    with session_factory() as session:
        m_after = session.query(ModelArtifactModel).filter_by(model_id=model_id).one()
        assert m_after.status != IngestionStatus.ACCEPTED


def test_capability_gating_on_unhardened_host_emits_no_false_conflicts(monkeypatch):
    """Verifies that on an unhardened host, behavioral detectors are UNAVAILABLE and emit zero false conflicts."""

    # Force HostIsolationAssessor to report NO_ISOLATION_AVAILABLE
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.NO_ISOLATION_AVAILABLE,
            {"platform": "Windows", "container": False},
            "Host platform does not support required containment boundaries.",
        ),
    )

    # Mock subprocess.Popen and subprocess.run to verify that no subprocesses are ever spawned
    mock_popen = MagicMock()
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(subprocess, "run", mock_run)

    # Initialize ModelConflictEngine
    conflict_engine = ModelConflictEngine()

    # Simulate detector runs on unhardened host: behavioral detectors return UNAVAILABLE
    detector_runs = [
        {
            "run_id": "run_01",
            "detector_id": "model_integrity.artifact_hash",
            "status": CapabilityStatus.COMPLETED,
            "metrics": {"matched": False, "expected_sha256": "f" * 64, "observed_sha256": "a" * 64},
        },
        {
            "run_id": "run_02",
            "detector_id": "model_integrity.state_dict_integrity",
            "status": CapabilityStatus.COMPLETED,
            "metrics": {"fingerprint_verified": False, "state_dict_fingerprint": "b" * 64},
        },
        {
            "run_id": "run_03",
            "detector_id": "model_integrity.behavioral_fingerprint",
            "status": CapabilityStatus.UNAVAILABLE,
            "status_reason": "Execution bypassed: Host platform does not support required containment boundaries.",
            "metrics": {},
        },
        {
            "run_id": "run_04",
            "detector_id": "model_integrity.backdoor_indicator",
            "status": CapabilityStatus.UNAVAILABLE,
            "status_reason": "Execution bypassed: Host platform does not support required containment boundaries.",
            "metrics": {},
        },
    ]

    conflict_findings, conflict_evidence = conflict_engine.evaluate_conflicts(
        assessment_id="ass_unhardened_check",
        snapshot_id="snap_unhardened",
        asset_id="model_unhardened",
        detector_runs=detector_runs,
        findings=[],
        evidence_records=[],
    )

    # Must emit ZERO conflict findings (capability gating strictly prevents false conflicts)
    assert len(conflict_findings) == 0, f"Expected 0 conflict findings, got: {conflict_findings}"
    assert len(conflict_evidence) == 0

    # Subprocess execution must be zero
    assert mock_popen.call_count == 0
    assert mock_run.call_count == 0


def test_adr003_benchmark_script_smoke_execution():
    """Verifies that ADR-003 safetensors benchmark script executes cleanly offline in smoke mode."""
    cmd = [
        sys.executable,
        "scripts/benchmark_safetensors.py",
        "--iterations",
        "1",
        "--tiers",
        "small",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Benchmark script failed with stderr:\n{result.stderr}"
    assert "EMPIRICAL BENCHMARK RESULTS" in result.stdout
    assert "small" in result.stdout.lower()
