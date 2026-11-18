"""Chaos benchmark test fixtures and environment provisioning.

Configures isolated ephemeral database, test client, and cryptographic
identities for adversarial chaos pipeline simulations.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.main import create_app
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.inference_integrity.signer import InferenceSigner
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


@pytest.fixture(autouse=True)
def mock_verified_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates verified container isolation for isolated model assessment workers."""
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
def chaos_env(
    tmp_path: Path,
) -> Generator[tuple[TestClient, sessionmaker[Session], IngestionConfig], None, None]:
    """Provides an isolated database, TestClient, and ingestion config for chaos tests."""
    db_file = tmp_path / "test_chaos_benchmark.db"
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
    )

    app = create_app(custom_settings=settings)

    def override_get_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app, base_url="http://testserver")

    ingestion_cfg = IngestionConfig(
        staging_dir=tmp_path / "staging",
        quarantine_dir=tmp_path / "quarantine",
        snapshots_dir=tmp_path / "snapshots",
    )

    yield client, session_factory, ingestion_cfg

    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def edge_keypair() -> tuple[bytes, bytes]:
    """Generates an Ed25519 keypair for telemetry signing."""
    return InferenceSigner.generate_keypair("edge-camera-chaos-defense-01")
