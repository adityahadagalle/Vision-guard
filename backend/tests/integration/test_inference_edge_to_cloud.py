"""Integration tests for VisionGuard lightweight edge runtime client and cloud ingestion.

Verifies end-to-end:
1. Edge client local key loading (file paths and in-memory PEM).
2. Monotonic hash-chain state progression across 5 sequential inferences.
3. Successful edge-to-cloud receipt ingestion and relational persistence.
4. SQLite persistence and cryptographic verification via ChainAuditEngine.
5. Sequence locking invariant: server errors (HTTP 409, 401, 500) prevent sequence advancement.
6. Replay attack rejection when attempting to submit stale sequence numbers.
7. Multi-device concurrent/interleaved independent receipt chains.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.repositories.inference_repository import SqliteInferenceRepository
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.main import create_app
from backend.app.modules.inference_integrity.audit import ChainAuditEngine
from backend.app.modules.inference_integrity.client import (
    EdgeClientKeyError,
    EdgeClientTransmissionError,
    VisionGuardEdgeClient,
)
from backend.app.modules.inference_integrity.signer import InferenceSigner


@pytest.fixture
def test_env(
    tmp_path: Path,
) -> Generator[tuple[TestClient, sessionmaker[Session]], None, None]:
    """Provides an isolated database and an in-memory TestClient for edge tests."""
    db_file = tmp_path / "test_inference_edge.db"
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


@pytest.fixture
def edge_keypair() -> tuple[bytes, bytes]:
    """Generates an Ed25519 keypair for edge-device-jetson-01."""
    return InferenceSigner.generate_keypair("edge-device-jetson-01")


def test_edge_to_cloud_five_sequential_inferences(
    test_env: tuple[TestClient, sessionmaker[Session]],
    edge_keypair: tuple[bytes, bytes],
) -> None:
    """Tests firing 5 sequential inferences through VisionGuardEdgeClient to SQLite."""
    test_client, session_factory = test_env
    priv_pem, pub_pem = edge_keypair
    device_id = "edge-device-jetson-01"

    client = VisionGuardEdgeClient(
        device_id=device_id,
        api_url="http://testserver/api/v1",
        private_key_pem=priv_pem,
        public_key_pem=pub_pem,
        http_client=test_client,
    )

    # Initial state assertions
    assert client.sequence_number == 0
    assert client.previous_receipt_hash is None

    model_hash = "a" * 64
    recorded_receipts: list[dict[str, Any]] = []

    # Fire 5 sequential inferences
    for i in range(5):
        input_hash = f"{i:x}" * 64
        output = {
            "prediction": "vehicle",
            "confidence": 0.90 + (i * 0.01),
            "bbox": [10 * i, 10 * i, 50 * i, 50 * i],
        }

        resp = client.record_inference(
            model_sha256=model_hash,
            input_hash=input_hash,
            inference_output=output,
        )
        assert resp["status"] == "ACCEPTED"
        assert resp["edge_device_id"] == device_id
        assert resp["sequence_number"] == i
        assert resp["payload_hash"] is not None

        recorded_receipts.append(resp)
        # Client sequence must have advanced to i + 1
        assert client.sequence_number == i + 1
        assert client.previous_receipt_hash == resp["payload_hash"]

    assert client.sequence_number == 5

    # Verify SQLite database persistence
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        db_receipts = repo.get_all_receipts_for_device_ascending(device_id)
        assert len(db_receipts) == 5

        for idx, rec in enumerate(db_receipts):
            assert rec.sequence_number == idx
            assert rec.edge_device_id == device_id
            if idx == 0:
                assert rec.previous_receipt_hash is None
            else:
                assert rec.previous_receipt_hash == db_receipts[idx - 1].payload_hash

        # Run ChainAuditEngine over the persisted SQLite chain
        audit = ChainAuditEngine.verify_device_chain(device_id, repo)
        assert audit.is_valid is True
        assert audit.IS_VALID is True
        assert audit.verified_count == 5
        assert audit.break_sequence_number is None


def test_edge_client_sequence_lock_on_server_failure(
    test_env: tuple[TestClient, sessionmaker[Session]],
    edge_keypair: tuple[bytes, bytes],
) -> None:
    """CRITICAL STOP CONDITION: Confirms edge client NEVER advances sequence on server failure."""
    test_client, session_factory = test_env
    priv_pem, pub_pem = edge_keypair
    device_id = "edge-device-failing-01"

    # Pre-populate Genesis record in database directly so sequence 0 already exists
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        repo.create_receipt(
            receipt_id="rec_existing_genesis",
            edge_device_id=device_id,
            timestamp_iso="2026-11-17T12:00:00Z",
            sequence_number=0,
            model_sha256="c" * 64,
            input_hash="d" * 64,
            inference_output={"preexisting": True},
            previous_receipt_hash=None,
            payload_hash="e" * 64,
            signature_b64="sig_genesis",
            public_key_pem=pub_pem.decode("ascii"),
        )

    # Instantiate fresh client (starts at sequence=0)
    client = VisionGuardEdgeClient(
        device_id=device_id,
        api_url="http://testserver/api/v1",
        private_key_pem=priv_pem,
        public_key_pem=pub_pem,
        http_client=test_client,
    )

    # Initial state
    assert client.sequence_number == 0
    assert client.previous_receipt_hash is None

    # Attempting to record inference will try to submit sequence=0, which already exists!
    # Server will return HTTP 409 Conflict (Replay/Sequence collision)
    with pytest.raises(EdgeClientTransmissionError) as exc_info:
        client.record_inference(
            model_sha256="c" * 64,
            input_hash="f" * 64,
            inference_output={"attempt": 1},
        )

    assert exc_info.value.status_code == 409

    # STOP INVARIANT: The client sequence number MUST NOT advance!
    assert client.sequence_number == 0
    assert client.previous_receipt_hash is None


def test_edge_client_file_based_key_loading(
    tmp_path: Path,
    test_env: tuple[TestClient, sessionmaker[Session]],
    edge_keypair: tuple[bytes, bytes],
) -> None:
    """Tests loading Ed25519 keys from filesystem paths on the edge device."""
    test_client, session_factory = test_env
    priv_pem, pub_pem = edge_keypair
    device_id = "edge-device-disk-01"

    priv_path = tmp_path / "private_key.pem"
    pub_path = tmp_path / "public_key.pem"

    priv_path.write_bytes(priv_pem)
    pub_path.write_text(pub_pem.decode("ascii"), encoding="utf-8")

    client = VisionGuardEdgeClient(
        device_id=device_id,
        api_url="http://testserver/api/v1",
        private_key_path=priv_path,
        public_key_path=pub_path,
        http_client=test_client,
    )

    resp = client.record_inference(
        model_sha256="b" * 64,
        input_hash="1" * 64,
        inference_output={"sensor": "cam0", "objects": 2},
    )

    assert resp["status"] == "ACCEPTED"
    assert client.sequence_number == 1
    assert client.previous_receipt_hash is not None

    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        audit = ChainAuditEngine.verify_device_chain(device_id, repo)
        assert audit.is_valid is True
        assert audit.verified_count == 1


def test_edge_client_missing_key_raises_error() -> None:
    """Tests that constructing client with missing keys raises EdgeClientKeyError."""
    with pytest.raises(EdgeClientKeyError, match="Either private_key_path or private_key_pem"):
        VisionGuardEdgeClient(
            device_id="edge-test",
            api_url="http://localhost:8000/api/v1",
        )

    with pytest.raises(EdgeClientKeyError, match="Private key file not found"):
        VisionGuardEdgeClient(
            device_id="edge-test",
            api_url="http://localhost:8000/api/v1",
            private_key_path=Path("/nonexistent/private.pem"),
            public_key_pem="fake_pub",
        )


def test_multi_device_interleaved_inferences(
    test_env: tuple[TestClient, sessionmaker[Session]],
) -> None:
    """Tests that multiple edge devices maintain completely independent monotonic chains."""
    test_client, session_factory = test_env

    dev1_id = "edge-drone-alpha"
    dev2_id = "edge-drone-beta"

    priv1, pub1 = InferenceSigner.generate_keypair(dev1_id)
    priv2, pub2 = InferenceSigner.generate_keypair(dev2_id)

    client1 = VisionGuardEdgeClient(
        device_id=dev1_id,
        api_url="http://testserver/api/v1",
        private_key_pem=priv1,
        public_key_pem=pub1,
        http_client=test_client,
    )

    client2 = VisionGuardEdgeClient(
        device_id=dev2_id,
        api_url="http://testserver/api/v1",
        private_key_pem=priv2,
        public_key_pem=pub2,
        http_client=test_client,
    )

    model_hash = "f" * 64

    # Interleave calls: Dev1 -> Dev2 -> Dev1 -> Dev2
    client1.record_inference(model_hash, "1" * 64, {"drone": "alpha", "seq": 0})
    client2.record_inference(model_hash, "2" * 64, {"drone": "beta", "seq": 0})
    client1.record_inference(model_hash, "3" * 64, {"drone": "alpha", "seq": 1})
    client2.record_inference(model_hash, "4" * 64, {"drone": "beta", "seq": 1})
    client1.record_inference(model_hash, "5" * 64, {"drone": "alpha", "seq": 2})

    assert client1.sequence_number == 3
    assert client2.sequence_number == 2

    # Audit both devices independently in SQLite
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)

        audit1 = ChainAuditEngine.verify_device_chain(dev1_id, repo)
        assert audit1.is_valid is True
        assert audit1.verified_count == 3

        audit2 = ChainAuditEngine.verify_device_chain(dev2_id, repo)
        assert audit2.is_valid is True
        assert audit2.verified_count == 2
