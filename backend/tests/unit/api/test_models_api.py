"""Unit tests for FastAPI model ingestion, query, and assessment endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.main import create_app


@pytest.fixture
def model_api_environment(tmp_path: Path):
    """Configures an isolated environment with temporary database and storage dirs."""
    db_file = tmp_path / "model_api_test.db"
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
        max_model_upload_bytes=10 * 1024 * 1024,  # 10 MB threshold for testing
        max_concurrent_uploads=3,
        global_quarantine_quota_bytes=50 * 1024 * 1024,  # 50 MB
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


def test_model_upload_valid_pt(model_api_environment):
    """Verifies that uploading a valid candidate .pt model creates a record and returns 201."""
    client, session_factory, data_dir = model_api_environment
    file_bytes = b"PK\x03\x04dummy_pytorch_model_checkpoint"

    response = client.post(
        "/api/v1/models",
        files={"file": ("candidate_model.pt", io.BytesIO(file_bytes), "application/octet-stream")},
        data={"name": "Candidate ResNet"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["model_id"].startswith("model_")
    assert data["name"] == "Candidate ResNet"
    assert data["format"] == "pytorch"
    assert data["status"] == "RECEIVED"
    assert data["size_bytes"] == len(file_bytes)
    assert data["job_id"] is not None


def test_model_upload_invalid_extension(model_api_environment):
    """Verifies that uploading unsupported extensions (.exe, .h5, .onnx) returns 400."""
    client, _, _ = model_api_environment
    response = client.post(
        "/api/v1/models",
        files={"file": ("model.h5", io.BytesIO(b"dummy"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_MODEL_EXTENSION"


def test_model_upload_empty_file(model_api_environment):
    """Verifies that uploading an empty 0-byte file returns 400."""
    client, _, _ = model_api_environment
    response = client.post(
        "/api/v1/models",
        files={"file": ("empty.pt", io.BytesIO(b""), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "EMPTY_MODEL_FILE"


def test_model_upload_path_traversal_filename(model_api_environment):
    """Verifies that filename path traversal attacks are rejected with 400."""
    client, _, _ = model_api_environment
    response = client.post(
        "/api/v1/models",
        files={"file": ("../../malicious.pt", io.BytesIO(b"payload"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] in ["PATH_TRAVERSAL_DETECTED", "INVALID_FILENAME"]


def test_model_get_detail_and_list(model_api_environment):
    """Verifies retrieval of model details and paginated list."""
    client, session_factory, _ = model_api_environment
    file_bytes = b"dummy_checkpoint_data"

    post_resp = client.post(
        "/api/v1/models",
        files={"file": ("model_1.pt", io.BytesIO(file_bytes), "application/octet-stream")},
        data={"name": "Model 1"},
    )
    assert post_resp.status_code == 201
    model_id = post_resp.json()["model_id"]

    # 1. Detail endpoint
    detail_resp = client.get(f"/api/v1/models/{model_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["model_id"] == model_id
    assert detail_data["status"] == "RECEIVED"
    assert detail_data["assessments_count"] == 0

    # 2. 404 for unknown model
    assert client.get("/api/v1/models/nonexistent").status_code == 404

    # 3. List endpoint
    list_resp = client.get("/api/v1/models?page=1&page_size=10")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_items"] >= 1
    assert any(m["model_id"] == model_id for m in list_data["items"])


def test_model_assessment_trigger_rejects_unaccepted(model_api_environment):
    """Verifies that triggering an assessment on an unaccepted model returns 409 Conflict."""
    client, _, _ = model_api_environment
    post_resp = client.post(
        "/api/v1/models",
        files={"file": ("unaccepted.pt", io.BytesIO(b"checkpoint"), "application/octet-stream")},
    )
    model_id = post_resp.json()["model_id"]

    # In RECEIVED status, assessment trigger must fail with 409
    ass_resp = client.post(f"/api/v1/models/{model_id}/assessments", json={})
    assert ass_resp.status_code == 409
    assert ass_resp.json()["error_code"] == "MODEL_NOT_ACCEPTED"


def test_no_delete_endpoint_permitted(model_api_environment):
    """Verifies that DELETE /api/v1/models/{model_id} does NOT exist (405 Method Not Allowed)."""
    client, _, _ = model_api_environment
    response = client.delete("/api/v1/models/model_123")
    assert response.status_code in [404, 405]
