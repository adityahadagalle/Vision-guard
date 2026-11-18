"""Integration tests for the distribution shift REST API and end-to-end evaluation pipeline.

Verifies:
1. POST /api/v1/distribution-shift/evaluate correctly evaluates clean distributions (is_drifting = False).
2. POST /api/v1/distribution-shift/evaluate correctly evaluates shifted distributions (is_drifting = True).
3. POST /api/v1/distribution-shift/evaluate fails closed with INCONCLUSIVE on micro-batches (< 30 samples).
4. Request schema rejects rogue extra attributes (extra="forbid").
5. GET /api/v1/distribution-shift/reports/{model_id} retrieves historical reports from SQLite.
6. Direct relational persistence in SQLite is verified.
"""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.v1.routes.distribution_shift import DriftReportModel
from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.main import create_app


@pytest.fixture
def rng() -> np.random.Generator:
    """Provides a deterministic random generator for distribution synthesis."""
    return np.random.default_rng(seed=1337)


@pytest.fixture
def test_env(
    tmp_path: Path,
) -> Generator[tuple[TestClient, sessionmaker[Session]], None, None]:
    """Provides an isolated database and an in-memory TestClient for distribution shift API tests."""
    db_file = tmp_path / "test_distribution_shift.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    settings = Settings(
        database_url=db_url,
        data_dir=tmp_path / "data",
    )

    app = create_app(custom_settings=settings)

    def override_get_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app, base_url="http://testserver/api/v1")

    yield client, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


def test_evaluate_clean_distribution_no_drift(
    test_env: tuple[TestClient, sessionmaker[Session]],
    rng: np.random.Generator,
) -> None:
    """Verifies that evaluating identical clean distributions yields is_drifting = False."""
    client, session_factory = test_env
    model_id = "model_resnet50_clean"

    ref_data = rng.normal(loc=0.5, scale=0.1, size=120).tolist()
    prod_data = rng.normal(loc=0.5, scale=0.1, size=120).tolist()

    payload = {
        "model_id": model_id,
        "target_feature": "confidence_score",
        "reference_data": ref_data,
        "production_data": prod_data,
        "ks_alpha": 0.05,
        "wasserstein_threshold": 0.10,
        "min_samples": 30,
    }

    response = client.post("/distribution-shift/evaluate", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["dataset_or_model_id"] == model_id
    assert data["is_drifting"] is False
    assert data["overall_status"] == "COMPLETED"
    assert data["reference_sample_count"] == 120
    assert data["production_sample_count"] == 120
    assert len(data["metrics"]) == 2

    # Check metric level
    ks_m = next(m for m in data["metrics"] if "kolmogorov" in m["metric_name"])
    w_m = next(m for m in data["metrics"] if "wasserstein" in m["metric_name"])
    assert ks_m["is_drifting"] is False
    assert w_m["is_drifting"] is False


def test_evaluate_shifted_distribution_triggers_drift(
    test_env: tuple[TestClient, sessionmaker[Session]],
    rng: np.random.Generator,
) -> None:
    """Verifies that an altered distribution triggers is_drifting = True via the REST API."""
    client, session_factory = test_env
    model_id = "model_yolov8_shifted"

    ref_data = rng.normal(loc=0.0, scale=1.0, size=100).tolist()
    # Large mean shift of +2.0 units
    prod_data = rng.normal(loc=2.0, scale=1.0, size=100).tolist()

    payload = {
        "model_id": model_id,
        "target_feature": "logit_norm",
        "reference_data": ref_data,
        "production_data": prod_data,
        "ks_alpha": 0.05,
        "wasserstein_threshold": 0.10,
    }

    response = client.post("/distribution-shift/evaluate", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["dataset_or_model_id"] == model_id
    assert data["is_drifting"] is True
    assert data["overall_status"] == "COMPLETED"

    ks_m = next(m for m in data["metrics"] if "kolmogorov" in m["metric_name"])
    w_m = next(m for m in data["metrics"] if "wasserstein" in m["metric_name"])
    assert ks_m["is_drifting"] is True
    assert w_m["is_drifting"] is True


def test_evaluate_insufficient_samples_returns_inconclusive(
    test_env: tuple[TestClient, sessionmaker[Session]],
) -> None:
    """Verifies that evaluating < 30 samples triggers INCONCLUSIVE without failing the request."""
    client, session_factory = test_env
    model_id = "model_insufficient_samples"

    ref_data = [0.1] * 15  # 15 samples (< 30)
    prod_data = [0.2] * 50

    payload = {
        "model_id": model_id,
        "target_feature": "aspect_ratio",
        "reference_data": ref_data,
        "production_data": prod_data,
        "min_samples": 30,
    }

    response = client.post("/distribution-shift/evaluate", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["is_drifting"] is False
    assert data["overall_status"] == "INCONCLUSIVE"
    assert "Insufficient sample density" in data["status_reason"]


def test_request_schema_extra_forbid(
    test_env: tuple[TestClient, sessionmaker[Session]],
) -> None:
    """Verifies that rogue fields in request payloads are rejected with HTTP 422."""
    client, session_factory = test_env

    payload = {
        "model_id": "model_test",
        "target_feature": "confidence",
        "reference_data": [0.5] * 35,
        "production_data": [0.5] * 35,
        "rogue_field": "malicious_injection",
    }

    response = client.post("/distribution-shift/evaluate", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_historical_drift_reports(
    test_env: tuple[TestClient, sessionmaker[Session]],
    rng: np.random.Generator,
) -> None:
    """Verifies querying historical drift assessment reports for a model."""
    client, session_factory = test_env
    model_id = "model_audit_history_01"

    # Evaluate twice for this model
    for i in range(2):
        ref = rng.normal(size=50).tolist()
        prod = rng.normal(loc=i * 1.5, size=50).tolist()
        resp = client.post(
            "/distribution-shift/evaluate",
            json={
                "model_id": model_id,
                "target_feature": f"feature_{i}",
                "reference_data": ref,
                "production_data": prod,
            },
        )
        assert resp.status_code == status.HTTP_201_CREATED

    # Query historical reports
    get_resp = client.get(f"/distribution-shift/reports/{model_id}")
    assert get_resp.status_code == status.HTTP_200_OK

    reports = get_resp.json()
    assert len(reports) == 2
    assert all(r["dataset_or_model_id"] == model_id for r in reports)

    # Check direct SQLite persistence
    with session_factory() as session:
        count = session.query(DriftReportModel).filter_by(model_id=model_id).count()
        assert count == 2


def test_get_reports_nonexistent_model_returns_empty_list(
    test_env: tuple[TestClient, sessionmaker[Session]],
) -> None:
    """Verifies that querying an unknown model returns an empty list, not a 404."""
    client, session_factory = test_env
    response = client.get("/distribution-shift/reports/unknown_model_uuid")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []
