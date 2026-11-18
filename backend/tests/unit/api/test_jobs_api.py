"""Unit tests for jobs API endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.domain.entities.enums import JobStatus
from backend.app.main import create_app


@pytest.fixture
def jobs_api_client(tmp_path: Path):
    """Provides test client with a seeded job record."""
    db_file = tmp_path / "jobs_api.db"
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

    session = session_factory()
    queue = SqliteJobQueue(session)
    job_id = queue.submit_job(
        task_type="ingestion",
        payload={"dataset_id": "ds_job_01"},
        job_id="job_seeded_01",
    )
    session.close()

    client = TestClient(app)
    yield client, job_id
    engine.dispose()


def test_get_job_status_returns_200(jobs_api_client):
    """Tests querying existing background job status."""
    client, job_id = jobs_api_client
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["task_type"] == "ingestion"
    assert data["status"] == JobStatus.QUEUED
    assert data["attempts"] == 0
    assert data["max_attempts"] == 3


def test_get_job_nonexistent_returns_404(jobs_api_client):
    """Tests 404 for unknown job identifier."""
    client, _ = jobs_api_client
    response = client.get("/api/v1/jobs/job_unknown")
    assert response.status_code == 404
    assert response.json()["error_code"] == "JOB_NOT_FOUND"
