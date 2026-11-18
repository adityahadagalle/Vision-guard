"""Integration tests for the complete VisionGuard workflow: Ingestion to Assessment."""

import io
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.session import _set_sqlite_pragmas
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    IngestionStatus,
    JobStatus,
)
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.services.orchestrator import AssessmentOrchestrator


def _create_coco_zip() -> bytes:
    """Helper creating an in-memory valid COCO format dataset with 2 images."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Image 1: 100x100 RGB
        img1 = Image.new("RGB", (100, 100), color="red")
        img1_buf = io.BytesIO()
        img1.save(img1_buf, format="JPEG")
        zf.writestr("images/0001.jpg", img1_buf.getvalue())

        # 2. Image 2: 100x100 RGB (distinct)
        img2 = Image.new("RGB", (100, 100), color="blue")
        img2_buf = io.BytesIO()
        img2.save(img2_buf, format="JPEG")
        zf.writestr("images/0002.jpg", img2_buf.getvalue())

        # 3. Annotations
        coco_data = {
            "images": [
                {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "0002.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
        zf.writestr("annotations/instances.json", json.dumps(coco_data))

    return buf.getvalue()


@pytest.fixture
def workflow_env(tmp_path: Path):
    """Sets up an isolated database, quarantine, staging, and snapshots directories."""
    db_file = tmp_path / "workflow.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    config = IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        snapshots_dir=tmp_path / "snapshots",
    )

    yield session_factory, config
    engine.dispose()


def test_full_ingestion_to_assessment_workflow(workflow_env):
    """End-to-end integration test:

    Upload archive -> Ingestion Worker -> Snapshot Creation -> Snapshot Persistence ->
    Snapshot Verification -> Assessment Creation -> Tier-1 Detector Execution ->
    Finding Persistence -> Assessment Completion.
    """
    session_factory, config = workflow_env
    session = session_factory()

    # Step 1: Write archive into quarantine
    archive_bytes = _create_coco_zip()
    quarantine_path = config.quarantine_dir / "coco_upload.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_bytes(archive_bytes)

    # Step 2: Register dataset in database
    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_coco_001",
        name="COCO Test Archive",
        dataset_format="AUTODETECT",
        storage_path=str(quarantine_path),
        sha256="0" * 64,
        size_bytes=len(archive_bytes),
        status=IngestionStatus.QUARANTINED,
    )

    # Step 3: Enqueue and execute ingestion job via orchestrator
    orchestrator = AssessmentOrchestrator(session=session, config=config)
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_coco_001",
            "quarantine_path": str(quarantine_path),
            "filename": "coco_upload.zip",
        },
    )

    completed_job = orchestrator.process_next_job(worker_id="test_worker_1")
    assert completed_job is not None
    assert completed_job["status"] == JobStatus.COMPLETED

    # Verify dataset is accepted and snapshot exists
    reloaded_ds = ds_repo.get_dataset("ds_coco_001")
    assert reloaded_ds.status == IngestionStatus.ACCEPTED
    snapshot_id = completed_job["result"]["snapshot_id"]
    snapshot = ds_repo.get_snapshot(snapshot_id)
    assert snapshot is not None
    assert snapshot.file_count >= 2

    # Step 4: Create assessment for the snapshot
    ass_repo = SqliteAssessmentRepository(session)
    assessment_id = "ass_coco_001"
    ass_repo.create_assessment(
        assessment_id=assessment_id,
        name="COCO Integrity Assessment",
        snapshot_id=snapshot_id,
        config={
            "detectors": [
                "data_integrity.corrupt_image",
                "data_integrity.exact_duplicate",
                "data_integrity.annotation_geometry",
            ]
        },
        dataset_id="ds_coco_001",
    )

    # Step 5: Enqueue and execute assessment job via orchestrator
    orchestrator.job_queue.submit_job(
        task_type="assessment",
        payload={
            "assessment_id": assessment_id,
            "snapshot_id": snapshot_id,
            "dataset_id": "ds_coco_001",
        },
    )

    completed_ass_job = orchestrator.process_next_job(worker_id="test_worker_2")
    assert completed_ass_job is not None
    assert completed_ass_job["status"] == JobStatus.COMPLETED

    # Step 6: Verify assessment outcome in database
    ass_record = ass_repo.get_assessment(assessment_id)
    assert ass_record["status"] == AssessmentStatus.COMPLETED
    assert len(ass_record["runs"]) == 3  # All 3 Tier-1 detectors executed
    assert all(r["status"] == CapabilityStatus.COMPLETED for r in ass_record["runs"])

    session.close()


def test_snapshot_tamper_causes_assessment_failure(workflow_env):
    """Tests that modifying snapshot contents causes cryptographic verification to fail,

    aborting assessment and transitioning it to FAILED without running detectors.
    """
    session_factory, config = workflow_env
    session = session_factory()

    # 1. Ingest clean dataset
    archive_bytes = _create_coco_zip()
    quarantine_path = config.quarantine_dir / "tamper_test.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_bytes(archive_bytes)

    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_tamper",
        name="Tamper Dataset",
        dataset_format="AUTODETECT",
        storage_path=str(quarantine_path),
        sha256="1" * 64,
        size_bytes=len(archive_bytes),
    )

    orchestrator = AssessmentOrchestrator(session=session, config=config)
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_tamper",
            "quarantine_path": str(quarantine_path),
            "filename": "tamper.zip",
        },
    )
    ingest_res = orchestrator.process_next_job(worker_id="worker_ingest")
    snapshot_id = ingest_res["result"]["snapshot_id"]

    # 2. Adversary tampers with a file in the snapshot directory
    storage_path = ds_repo.get_snapshot_storage_path(snapshot_id)
    target_img = Path(storage_path) / "images" / "0001.jpg"
    os.chmod(target_img, stat.S_IWRITE)
    target_img.write_bytes(b"CORRUPTED_ADVERSARIAL_PAYLOAD")

    # 3. Create and trigger assessment
    ass_repo = SqliteAssessmentRepository(session)
    ass_id = "ass_tamper_detect"
    ass_repo.create_assessment(
        assessment_id=ass_id,
        name="Tamper Audit",
        snapshot_id=snapshot_id,
        config={},
        dataset_id="ds_tamper",
    )
    orchestrator.job_queue.submit_job(
        task_type="assessment",
        payload={"assessment_id": ass_id, "snapshot_id": snapshot_id, "dataset_id": "ds_tamper"},
    )

    # 4. Orchestrator executes assessment job
    job_result = orchestrator.process_next_job(worker_id="worker_assess")
    assert job_result["status"] == JobStatus.FAILED
    assert "Snapshot verification failed" in job_result["error_message"]

    # Assessment must be marked FAILED, not COMPLETED
    ass = ass_repo.get_assessment(ass_id)
    assert ass["status"] == AssessmentStatus.FAILED
    assert "Snapshot verification failed" in ass["status_reason"]
    assert len(ass["runs"]) == 0  # No detectors run on tampered snapshot

    session.close()


class CrashingFaultyDetector(BaseDetector):
    """Test detector simulating an unhandled algorithmic crash."""

    @property
    def detector_id(self) -> str:
        return "data_integrity.crashing_faulty"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements()

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        raise RuntimeError("Mathematical non-convergence: SVD failed to converge.")


def test_detector_crash_causes_assessment_failure(workflow_env):
    """Tests that a detector crash fails the assessment and does NOT mark it clean."""
    session_factory, config = workflow_env
    session = session_factory()

    # 1. Ingest clean dataset
    archive_bytes = _create_coco_zip()
    quarantine_path = config.quarantine_dir / "crash_test.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_bytes(archive_bytes)

    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_crash",
        name="Crash Dataset",
        dataset_format="AUTODETECT",
        storage_path=str(quarantine_path),
        sha256="2" * 64,
        size_bytes=len(archive_bytes),
    )

    registry = DetectorRegistry()
    registry.register(CrashingFaultyDetector())

    orchestrator = AssessmentOrchestrator(session=session, config=config, registry=registry)
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_crash",
            "quarantine_path": str(quarantine_path),
            "filename": "crash.zip",
        },
    )
    ingest_res = orchestrator.process_next_job(worker_id="worker_ingest")
    snapshot_id = ingest_res["result"]["snapshot_id"]

    # 2. Trigger assessment with the crashing detector
    ass_repo = SqliteAssessmentRepository(session)
    ass_id = "ass_crash_eval"
    ass_repo.create_assessment(
        assessment_id=ass_id,
        name="Crash Test",
        snapshot_id=snapshot_id,
        config={"detectors": ["data_integrity.crashing_faulty"]},
        dataset_id="ds_crash",
    )
    orchestrator.job_queue.submit_job(
        task_type="assessment",
        payload={"assessment_id": ass_id, "snapshot_id": snapshot_id, "dataset_id": "ds_crash"},
    )

    # 3. Process assessment job
    job_result = orchestrator.process_next_job(worker_id="worker_assess")
    assert job_result["status"] == JobStatus.FAILED

    # Assessment must resolve to FAILED
    ass = ass_repo.get_assessment(ass_id)
    assert ass["status"] == AssessmentStatus.FAILED
    assert len(ass["runs"]) == 1
    assert ass["runs"][0]["status"] == CapabilityStatus.FAILED
    assert "SVD failed to converge" in ass["runs"][0]["status_reason"]

    session.close()
