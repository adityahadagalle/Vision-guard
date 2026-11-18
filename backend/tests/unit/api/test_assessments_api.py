"""Unit tests for assessment query endpoints and findings pagination."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.session import _set_sqlite_pragmas, get_db, now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    IngestionStatus,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.main import create_app


@pytest.fixture
def assessment_api_client(tmp_path: Path):
    """Provides test client pre-populated with assessment, detector runs, and findings."""
    db_file = tmp_path / "assess_api.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    settings = Settings(database_url=db_url, data_dir=tmp_path / "data")
    app = create_app(custom_settings=settings)

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Seed data
    session = session_factory()
    ds_repo = SqliteDatasetRepository(session)
    ass_repo = SqliteAssessmentRepository(session)

    ds_repo.create_dataset(
        dataset_id="ds_seed",
        name="Seeded Dataset",
        dataset_format="COCO_DETECTION",
        storage_path="/path/seed",
        sha256="f" * 64,
        size_bytes=1000,
        status=IngestionStatus.ACCEPTED,
    )
    ds_repo.create_snapshot(
        snapshot_id="snap_seed",
        dataset_id="ds_seed",
        manifest_sha256="e" * 64,
        storage_path="/path/snap",
        file_count=5,
        total_size_bytes=1000,
    )

    assessment_id = "ass_seeded"
    ass_repo.create_assessment(
        assessment_id=assessment_id,
        name="Pre-seeded Audit",
        snapshot_id="snap_seed",
        config={"detectors": ["data_integrity.corrupt_image"]},
        dataset_id="ds_seed",
    )

    now = now_utc()
    ass_repo.record_detector_run(
        run_id="run_seeded_01",
        assessment_id=assessment_id,
        detector_id="data_integrity.corrupt_image",
        detector_version="1.0.0",
        status=CapabilityStatus.COMPLETED,
        duration_ms=120,
        started_at=now,
        completed_at=now,
    )

    # Seed 5 findings with diverse categories and severities for pagination/filtering tests
    findings = []
    severities = [
        SeverityLevel.LOW,
        SeverityLevel.MEDIUM,
        SeverityLevel.HIGH,
        SeverityLevel.HIGH,
        SeverityLevel.CRITICAL,
    ]
    categories = [
        FindingCategory.STRUCTURAL_DEFECT,
        FindingCategory.DUPLICATE_SAMPLE,
        FindingCategory.CORRUPTED_FILE,
        FindingCategory.CORRUPTED_FILE,
        FindingCategory.SPLIT_LEAKAGE,
    ]

    for i in range(5):
        fid = f"find_{i:03d}"
        f = UniversalFinding(
            finding_id=fid,
            assessment_id=assessment_id,
            detector_id="data_integrity.corrupt_image",
            detector_version="1.0.0",
            category=categories[i],
            severity=severities[i],
            title=f"Finding Title {i}",
            description=f"Detailed description {i}",
            confidence=0.95,
            reproducibility=ReproducibilityMetadata(
                execution_timestamp=now,
                seed=42,
                software_versions={"torch": "2.15.0"},
                detector_config={},
            ),
            created_at=now,
            run_id="run_seeded_01",
            status=CapabilityStatus.COMPLETED,
            suggested_disposition=SuggestedDisposition.REVIEW,
            affected_assets=[
                AffectedAssetReference(
                    asset_id="ds_seed",
                    sample_path=f"images/{i:04d}.jpg",
                    sha256="1" * 64,
                )
            ],
        )
        findings.append(f)

    ass_repo.record_findings(assessment_id, findings)
    ass_repo.update_status(assessment_id, AssessmentStatus.COMPLETED)
    session.close()

    client = TestClient(app)
    yield client, assessment_id
    engine.dispose()


def test_get_assessment_returns_summary_and_runs(assessment_api_client):
    """Tests retrieving assessment details and runs."""
    client, assessment_id = assessment_api_client

    response = client.get(f"/api/v1/assessments/{assessment_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["assessment_id"] == assessment_id
    assert data["status"] == AssessmentStatus.COMPLETED
    assert len(data["runs"]) == 1
    assert data["runs"][0]["detector_id"] == "data_integrity.corrupt_image"


def test_get_assessment_nonexistent_returns_404(assessment_api_client):
    """Tests 404 behavior for unknown assessment."""
    client, _ = assessment_api_client
    response = client.get("/api/v1/assessments/ass_unknown")
    assert response.status_code == 404
    assert response.json()["error_code"] == "ASSESSMENT_NOT_FOUND"


def test_findings_pagination_and_filtering(assessment_api_client):
    """Tests pagination bounds, total pages calculation, and category/severity filtering."""
    client, assessment_id = assessment_api_client

    # 1. Page 1 with page_size = 2 -> 2 items, total_items = 5, total_pages = 3
    resp_p1 = client.get(f"/api/v1/assessments/{assessment_id}/findings?page=1&page_size=2")
    assert resp_p1.status_code == 200
    p1 = resp_p1.json()
    assert p1["page"] == 1
    assert p1["page_size"] == 2
    assert p1["total_items"] == 5
    assert p1["total_pages"] == 3
    assert len(p1["items"]) == 2

    # 2. Page 3 with page_size = 2 -> 1 remaining item
    resp_p3 = client.get(f"/api/v1/assessments/{assessment_id}/findings?page=3&page_size=2")
    assert resp_p3.status_code == 200
    p3 = resp_p3.json()
    assert len(p3["items"]) == 1

    # 3. Filter by category CORRUPTED_FILE -> 2 items
    resp_cat = client.get(f"/api/v1/assessments/{assessment_id}/findings?category=CORRUPTED_FILE")
    assert resp_cat.status_code == 200
    assert resp_cat.json()["total_items"] == 2

    # 4. Filter by severity HIGH -> 2 items
    resp_sev = client.get(f"/api/v1/assessments/{assessment_id}/findings?severity=HIGH")
    assert resp_sev.status_code == 200
    assert resp_sev.json()["total_items"] == 2
