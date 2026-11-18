"""Unit tests for SQLite persistence models and repository adapters."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import (
    Base,
)
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.session import _set_sqlite_pragmas, now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    AssetType,
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    IngestionStatus,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)


@pytest.fixture
def sqlite_session(tmp_path: Path):
    """Provides an isolated, file-backed SQLite session with foreign keys enabled."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def test_dataset_and_snapshot_persistence_and_reconstruction(sqlite_session):
    """Tests creating dataset, snapshot, and reconstructing domain entities."""
    repo = SqliteDatasetRepository(sqlite_session)

    # 1. Create dataset
    ds = repo.create_dataset(
        dataset_id="ds_test_001",
        name="Test COCO Dataset",
        dataset_format="COCO_DETECTION",
        storage_path="/data/quarantine/ds_test_001.zip",
        sha256="a" * 64,
        size_bytes=1024,
        status=IngestionStatus.ACCEPTED,
        metadata={"annotator": "tester"},
    )
    assert ds.dataset_id == "ds_test_001"

    # 2. Create snapshot
    snap = repo.create_snapshot(
        snapshot_id="snap_test_001",
        dataset_id="ds_test_001",
        manifest_sha256="b" * 64,
        storage_path="/data/snapshots/snap_test_001",
        file_count=10,
        total_size_bytes=2048,
        metadata={"format": "COCO_DETECTION"},
    )

    # Check domain entity reconstruction
    assert snap.snapshot_id == "snap_test_001"
    assert snap.manifest_sha256 == "b" * 64
    assert snap.file_count == 10
    assert snap.asset_type == AssetType.DATASET_SNAPSHOT

    # 3. Retrieve snapshot through repository
    retrieved = repo.get_snapshot("snap_test_001")
    assert retrieved is not None
    assert retrieved.snapshot_id == "snap_test_001"
    assert retrieved.manifest_sha256 == "b" * 64
    assert retrieved.metadata["format"] == "COCO_DETECTION"

    # Authoritative digest lookup
    auth_digest = repo.get_authoritative_manifest_sha256("snap_test_001")
    assert auth_digest == "b" * 64


def test_foreign_key_enforcement_and_cascade_delete(sqlite_session):
    """Tests foreign key enforcement and cascade deletion on datasets."""
    repo = SqliteDatasetRepository(sqlite_session)

    # Creating snapshot referencing nonexistent dataset fails
    with pytest.raises(IntegrityError):
        repo.create_snapshot(
            snapshot_id="snap_orphan",
            dataset_id="nonexistent_dataset",
            manifest_sha256="c" * 64,
            storage_path="/path",
            file_count=1,
            total_size_bytes=100,
        )
    sqlite_session.rollback()

    # Create parent dataset and child snapshot
    repo.create_dataset(
        dataset_id="ds_parent",
        name="Parent Dataset",
        dataset_format="FOLDER_PER_CLASS",
        storage_path="/path",
        sha256="d" * 64,
        size_bytes=500,
    )
    repo.create_snapshot(
        snapshot_id="snap_child",
        dataset_id="ds_parent",
        manifest_sha256="e" * 64,
        storage_path="/path/child",
        file_count=5,
        total_size_bytes=500,
    )

    assert repo.get_snapshot("snap_child") is not None

    # Delete parent dataset
    parent = repo.get_dataset("ds_parent")
    sqlite_session.delete(parent)
    sqlite_session.commit()

    # Child snapshot should be deleted via cascade
    assert repo.get_snapshot("snap_child") is None


def test_assessment_repository_lifecycle_findings_and_evidence(sqlite_session):
    """Tests full assessment lifecycle, detector run, findings, and evidence persistence."""
    ds_repo = SqliteDatasetRepository(sqlite_session)
    ass_repo = SqliteAssessmentRepository(sqlite_session)

    ds_repo.create_dataset(
        dataset_id="ds_eval",
        name="Evaluation Dataset",
        dataset_format="COCO_DETECTION",
        storage_path="/path/quarantine",
        sha256="1" * 64,
        size_bytes=1000,
    )
    ds_repo.create_snapshot(
        snapshot_id="snap_eval",
        dataset_id="ds_eval",
        manifest_sha256="2" * 64,
        storage_path="/path/snap",
        file_count=5,
        total_size_bytes=1000,
    )

    # 1. Create Assessment
    ass_id = "ass_001"
    ass_repo.create_assessment(
        assessment_id=ass_id,
        name="Integrity Audit 001",
        snapshot_id="snap_eval",
        config={"detectors": ["data_integrity.corrupt_image"]},
    )

    ass = ass_repo.get_assessment(ass_id)
    assert ass is not None
    assert ass["status"] == AssessmentStatus.PENDING
    assert ass["config"]["detectors"] == ["data_integrity.corrupt_image"]

    # 2. Update status
    ass_repo.update_status(ass_id, AssessmentStatus.ANALYZING)
    ass = ass_repo.get_assessment(ass_id)
    assert ass["status"] == AssessmentStatus.ANALYZING

    # 3. Record detector run
    now = now_utc()
    ass_repo.record_detector_run(
        run_id="run_001",
        assessment_id=ass_id,
        detector_id="data_integrity.corrupt_image",
        detector_version="1.0.0",
        status=CapabilityStatus.COMPLETED,
        duration_ms=45,
        started_at=now,
        completed_at=now,
    )

    # 4. Record findings and evidence
    finding_id = f"find_{uuid.uuid4().hex}"
    evidence_id = f"evi_{uuid.uuid4().hex}"

    finding = UniversalFinding(
        finding_id=finding_id,
        assessment_id=ass_id,
        detector_id="data_integrity.corrupt_image",
        detector_version="1.0.0",
        category=FindingCategory.CORRUPTED_FILE,
        severity=SeverityLevel.HIGH,
        title="Corrupted JPEG Header",
        description="Truncated file detected at images/0001.jpg",
        confidence=1.0,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now,
            seed=42,
            software_versions={"pillow": "13.0.0"},
            detector_config={},
        ),
        created_at=now,
        run_id="run_001",
        status=CapabilityStatus.COMPLETED,
        suggested_disposition=SuggestedDisposition.QUARANTINE,
        affected_assets=[
            AffectedAssetReference(
                asset_id="ds_eval",
                sample_path="images/0001.jpg",
                sha256="3" * 64,
            )
        ],
        evidence_ids=[evidence_id],
    )

    evidence = EvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.IMAGE_CORRUPTION,
        finding_id=finding_id,
        signal_name="image_decode",
        rationale="Pillow raised UnidentifiedImageError",
        created_at=now,
        sample_reference="images/0001.jpg",
        metadata={"file_size": 12},
    )

    ass_repo.record_findings(ass_id, [finding])
    ass_repo.record_evidence([evidence])

    # 5. Query findings and evidence
    retrieved_findings = ass_repo.get_findings(ass_id)
    assert len(retrieved_findings) == 1
    rf = retrieved_findings[0]
    assert rf.finding_id == finding_id
    assert rf.category == FindingCategory.CORRUPTED_FILE
    assert rf.severity == SeverityLevel.HIGH
    assert len(rf.affected_assets) == 1
    assert rf.affected_assets[0].sample_path == "images/0001.jpg"

    retrieved_evidence = ass_repo.get_evidence_for_finding(finding_id)
    assert len(retrieved_evidence) == 1
    re = retrieved_evidence[0]
    assert re.evidence_id == evidence_id
    assert re.signal_name == "image_decode"
    assert re.metadata["file_size"] == 12

    # 6. Complete assessment
    ass_repo.update_status(ass_id, AssessmentStatus.COMPLETED)
    ass = ass_repo.get_assessment(ass_id)
    assert ass["status"] == AssessmentStatus.COMPLETED
    assert ass["completed_at"] is not None
