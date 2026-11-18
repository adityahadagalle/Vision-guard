"""Unit tests for FastAPI dataset ingestion and assessment endpoints."""

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.domain.entities.enums import IngestionStatus
from backend.app.main import create_app


@pytest.fixture
def api_environment(tmp_path: Path):
    """Configures an isolated environment with temporary database and storage dirs."""
    db_file = tmp_path / "api_test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        database_url=db_url,
        data_dir=data_dir,
        max_upload_mb=5,  # 5 MB threshold for testing
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
    yield client, session_factory, data_dir
    engine.dispose()


def test_valid_dataset_upload_returns_201_and_queues_job(api_environment):
    """Tests that uploading a valid ZIP archive streams to quarantine and returns 201."""
    client, session_factory, _ = api_environment

    # Build a small dummy zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.txt", "hello world")
    zip_bytes = zip_buffer.getvalue()

    response = client.post(
        "/api/v1/datasets",
        files={"file": ("dataset.zip", zip_bytes, "application/zip")},
        data={"name": "My Test Dataset"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["dataset_id"].startswith("ds_")
    assert data["name"] == "My Test Dataset"
    assert data["status"] == IngestionStatus.QUARANTINED
    assert data["job_id"] is not None
    assert data["size_bytes"] == len(zip_bytes)


def test_oversized_upload_returns_413(api_environment):
    """Tests that uploads exceeding max_upload_mb are rejected with HTTP 413."""
    client, _, _ = api_environment

    # Create dummy buffer exceeding 5 MB threshold
    huge_bytes = b"0" * (6 * 1024 * 1024)

    response = client.post(
        "/api/v1/datasets",
        files={"file": ("huge.zip", huge_bytes, "application/zip")},
    )

    assert response.status_code == 413
    data = response.json()
    assert data["error_code"] == "FILE_TOO_LARGE"


def test_empty_filename_rejected_with_400(api_environment):
    """Tests that missing or empty filename returns 400."""
    client, _, _ = api_environment

    response = client.post(
        "/api/v1/datasets",
        files={"file": ("", b"dummy content", "application/zip")},
    )

    assert response.status_code in (400, 422)


def test_get_dataset_detail_and_nonexistent_returns_404(api_environment):
    """Tests querying existing dataset and 404 behavior for unknown identifier."""
    client, _, _ = api_environment

    # Unknown identifier
    resp_404 = client.get("/api/v1/datasets/ds_nonexistent")
    assert resp_404.status_code == 404
    assert resp_404.json()["error_code"] == "DATASET_NOT_FOUND"

    # Create one via upload first
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("0001.jpg", "image")
    upload_resp = client.post(
        "/api/v1/datasets",
        files={"file": ("valid.zip", zip_buffer.getvalue(), "application/zip")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    # Query details
    resp_200 = client.get(f"/api/v1/datasets/{dataset_id}")
    assert resp_200.status_code == 200
    assert resp_200.json()["dataset_id"] == dataset_id


def test_trigger_assessment_without_snapshot_rejected(api_environment):
    """Tests that triggering assessment on a quarantined dataset without approved snapshot returns 400."""
    client, _, _ = api_environment

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("test.txt", "data")
    upload_resp = client.post(
        "/api/v1/datasets",
        files={"file": ("test.zip", zip_buffer.getvalue(), "application/zip")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    # Attempt to trigger assessment immediately before ingestion snapshot exists
    ass_resp = client.post(
        f"/api/v1/datasets/{dataset_id}/assessments",
        json={"name": "Premature Assessment"},
    )

    assert ass_resp.status_code == 400
    assert ass_resp.json()["error_code"] == "NO_SNAPSHOT_AVAILABLE"
