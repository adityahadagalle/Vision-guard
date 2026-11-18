"""Unit tests for FastAPI inference receipt ingestion and verification API."""

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings
from backend.app.db.models import Base
from backend.app.db.repositories.inference_repository import SqliteInferenceRepository
from backend.app.db.session import _set_sqlite_pragmas, get_db
from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.main import create_app
from backend.app.modules.inference_integrity.chain import MonotonicHashChainEngine
from backend.app.modules.inference_integrity.signer import InferenceSigner


@pytest.fixture
def inference_test_env(
    tmp_path: Path,
) -> Generator[tuple[TestClient, sessionmaker[Session]], None, None]:
    """Provides an isolated TestClient and temporary SQLite database for inference API tests."""
    db_file = tmp_path / "test_inference_api.db"
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
    client = TestClient(app)

    yield client, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def device_keys() -> tuple[bytes, bytes]:
    """Generates an Ed25519 keypair for edge-camera-01."""
    return InferenceSigner.generate_keypair("edge-camera-01")


@pytest.fixture
def alternate_keys() -> tuple[bytes, bytes]:
    """Generates an alternate Ed25519 keypair."""
    return InferenceSigner.generate_keypair("edge-camera-02")


def test_genesis_receipt_ingestion_success(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests successful ingestion of a Genesis receipt (sequence=0, previous_receipt_hash=None)."""
    client, session_factory = inference_test_env
    priv, pub = device_keys

    payload, payload_hash, sig_b64 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "pedestrian", "confidence": 0.96},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )

    request_data = {
        "payload": payload.model_dump(mode="json"),
        "signature": sig_b64,
        "public_key_pem": pub.decode("ascii"),
    }

    response = client.post("/api/v1/inference/receipts", json=request_data)
    assert response.status_code == 201

    data = response.json()
    assert data["edge_device_id"] == "edge-camera-01"
    assert data["sequence_number"] == 0
    assert data["payload_hash"] == payload_hash
    assert data["status"] == "ACCEPTED"
    assert data["signature"] == sig_b64

    # Verify database persistence
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        latest = repo.get_latest_receipt("edge-camera-01")
        assert latest is not None
        assert latest.sequence_number == 0
        assert latest.payload_hash == payload_hash
        assert latest.previous_receipt_hash is None


def test_chained_receipt_ingestion_success(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests sequentially ingesting Genesis (seq 0) and Block 1 (seq 1)."""
    client, session_factory = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    # 1. Post Genesis
    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"detections": 1},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    r0 = client.post(
        "/api/v1/inference/receipts",
        json={"payload": p0.model_dump(mode="json"), "signature": s0, "public_key_pem": pub_pem},
    )
    assert r0.status_code == 201

    # 2. Post Chained Block 1
    p1, h1, s1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="c" * 64,
        output={"detections": 2},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )
    r1 = client.post(
        "/api/v1/inference/receipts",
        json={"payload": p1.model_dump(mode="json"), "signature": s1, "public_key_pem": pub_pem},
    )
    assert r1.status_code == 201
    data1 = r1.json()
    assert data1["sequence_number"] == 1
    assert data1["payload_hash"] == h1

    # Verify latest receipt in DB
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        latest = repo.get_latest_receipt("edge-camera-01")
        assert latest is not None
        assert latest.sequence_number == 1
        assert latest.previous_receipt_hash == h0


def test_invalid_signature_rejected_with_401(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
    alternate_keys: tuple[bytes, bytes],
) -> None:
    """Tests that an invalid or forged signature returns HTTP 401 Unauthorized immediately."""
    client, session_factory = inference_test_env
    priv, _ = device_keys
    _, wrong_pub = alternate_keys

    payload, _, sig_b64 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "car"},
        private_key=priv,
        public_key=device_keys[1],
        previous_receipt=None,
    )

    # Submit with mismatched public key
    bad_req = {
        "payload": payload.model_dump(mode="json"),
        "signature": sig_b64,
        "public_key_pem": wrong_pub.decode("ascii"),
    }
    response = client.post("/api/v1/inference/receipts", json=bad_req)
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_SIGNATURE"

    # Verify nothing was persisted
    with session_factory() as session:
        repo = SqliteInferenceRepository(session)
        assert repo.get_latest_receipt("edge-camera-01") is None


def test_corrupted_signature_rejected_with_401(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that a corrupted signature string returns HTTP 401."""
    client, _ = inference_test_env
    priv, pub = device_keys

    payload, _, _ = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "truck"},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )

    corrupted_sig = "A" * 88  # Valid base64 length, invalid cryptographic content
    bad_req = {
        "payload": payload.model_dump(mode="json"),
        "signature": corrupted_sig,
        "public_key_pem": pub.decode("ascii"),
    }
    response = client.post("/api/v1/inference/receipts", json=bad_req)
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_SIGNATURE"


def test_replay_attack_rejected_with_409(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that replaying sequence 0 returns HTTP 409 Conflict."""
    client, _ = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    # 1. Post Genesis
    p0, _, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"step": 0},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    req0 = {"payload": p0.model_dump(mode="json"), "signature": s0, "public_key_pem": pub_pem}
    r0 = client.post("/api/v1/inference/receipts", json=req0)
    assert r0.status_code == 201

    # 2. Replay the exact same sequence 0
    replay_resp = client.post("/api/v1/inference/receipts", json=req0)
    assert replay_resp.status_code == 409
    assert replay_resp.json()["error_code"] == "REPLAY_DETECTED"

    # 3. Post sequence 1
    p1, _, s1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="c" * 64,
        output={"step": 1},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )
    r1 = client.post(
        "/api/v1/inference/receipts",
        json={"payload": p1.model_dump(mode="json"), "signature": s1, "public_key_pem": pub_pem},
    )
    assert r1.status_code == 201

    # 4. Attempt replaying sequence 0 again when latest is 1
    replay_old = client.post("/api/v1/inference/receipts", json=req0)
    assert replay_old.status_code == 409
    assert replay_old.json()["error_code"] == "REPLAY_DETECTED"


def test_sequence_gap_rejected_with_409(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that skipping a sequence number (0 -> 2) returns HTTP 409 Conflict."""
    client, _ = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    # 1. Post Genesis (seq 0)
    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"step": 0},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    client.post(
        "/api/v1/inference/receipts",
        json={"payload": p0.model_dump(mode="json"), "signature": s0, "public_key_pem": pub_pem},
    )

    # 2. Post Gap Sequence 2 (expected 1)
    gap_payload = InferenceReceiptPayload(
        edge_device_id="edge-camera-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=2,  # Skipped 1
        model_sha256="a" * 64,
        input_hash="c" * 64,
        inference_output={"step": 2},
        previous_receipt_hash=h0,
    )
    gap_sig = InferenceSigner.sign_payload(gap_payload, priv)

    gap_resp = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": gap_payload.model_dump(mode="json"),
            "signature": gap_sig,
            "public_key_pem": pub_pem,
        },
    )
    assert gap_resp.status_code == 409
    assert gap_resp.json()["error_code"] == "SEQUENCE_GAP_DETECTED"


def test_genesis_non_zero_sequence_rejected_with_409(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that the initial receipt must have sequence 0."""
    client, _ = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    # First receipt attempted with seq 1
    bad_genesis = InferenceReceiptPayload(
        edge_device_id="edge-camera-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=1,  # Must be 0 for initial
        model_sha256="a" * 64,
        input_hash="b" * 64,
        inference_output={"init": True},
        previous_receipt_hash=None,
    )
    sig = InferenceSigner.sign_payload(bad_genesis, priv)

    resp = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": bad_genesis.model_dump(mode="json"),
            "signature": sig,
            "public_key_pem": pub_pem,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "SEQUENCE_ERROR"


def test_genesis_non_null_previous_hash_rejected_with_409(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that the initial receipt must have previous_receipt_hash=None."""
    client, _ = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    bad_genesis = InferenceReceiptPayload(
        edge_device_id="edge-camera-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=0,
        model_sha256="a" * 64,
        input_hash="b" * 64,
        inference_output={"init": True},
        previous_receipt_hash="f" * 64,  # Must be None
    )
    sig = InferenceSigner.sign_payload(bad_genesis, priv)

    resp = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": bad_genesis.model_dump(mode="json"),
            "signature": sig,
            "public_key_pem": pub_pem,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "CHAIN_LINK_ERROR"


def test_hash_link_mismatch_rejected_with_409(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that a chained receipt with mismatched previous_receipt_hash is rejected with HTTP 409."""
    client, _ = inference_test_env
    priv, pub = device_keys
    pub_pem = pub.decode("ascii")

    # 1. Post Genesis
    p0, _, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"step": 0},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    client.post(
        "/api/v1/inference/receipts",
        json={"payload": p0.model_dump(mode="json"), "signature": s0, "public_key_pem": pub_pem},
    )

    # 2. Post receipt 1 with corrupted previous_receipt_hash
    p1_corrupt = InferenceReceiptPayload(
        edge_device_id="edge-camera-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=1,
        model_sha256="a" * 64,
        input_hash="c" * 64,
        inference_output={"step": 1},
        previous_receipt_hash="0" * 64,  # Corrupted link
    )
    s1_corrupt = InferenceSigner.sign_payload(p1_corrupt, priv)

    resp = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": p1_corrupt.model_dump(mode="json"),
            "signature": s1_corrupt,
            "public_key_pem": pub_pem,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HASH_LINK_MISMATCH"


def test_flat_payload_normalization_supported(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests that clients can submit flat payload structures for ergonomic integration."""
    client, _ = inference_test_env
    priv, pub = device_keys

    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"flat": True},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )

    flat_data = {
        "edge_device_id": p0.edge_device_id,
        "timestamp_iso": p0.timestamp_iso,
        "sequence_number": p0.sequence_number,
        "model_sha256": p0.model_sha256,
        "input_hash": p0.input_hash,
        "inference_output": p0.inference_output,
        "previous_receipt_hash": p0.previous_receipt_hash,
        "signature": s0,
        "public_key_pem": pub.decode("ascii"),
    }

    resp = client.post("/api/v1/inference/receipts", json=flat_data)
    assert resp.status_code == 201
    assert resp.json()["payload_hash"] == h0


def test_get_latest_receipt_endpoint(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
) -> None:
    """Tests GET /api/v1/inference/receipts/{edge_device_id}/latest."""
    client, _ = inference_test_env
    priv, pub = device_keys

    # 404 when no receipts exist
    r_empty = client.get("/api/v1/inference/receipts/nonexistent-device/latest")
    assert r_empty.status_code == 404
    assert r_empty.json()["error_code"] == "RECEIPT_NOT_FOUND"

    # Post genesis
    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "test"},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": p0.model_dump(mode="json"),
            "signature": s0,
            "public_key_pem": pub.decode("ascii"),
        },
    )

    # 200 when receipt exists
    r_found = client.get("/api/v1/inference/receipts/edge-camera-01/latest")
    assert r_found.status_code == 200
    data = r_found.json()
    assert data["edge_device_id"] == "edge-camera-01"
    assert data["sequence_number"] == 0
    assert data["payload_hash"] == h0


def test_multi_device_chain_isolation(
    inference_test_env: tuple[TestClient, sessionmaker[Session]],
    device_keys: tuple[bytes, bytes],
    alternate_keys: tuple[bytes, bytes],
) -> None:
    """Tests that two independent edge devices maintain separate, non-conflicting hash chains."""
    client, _ = inference_test_env
    priv1, pub1 = device_keys
    priv2, pub2 = alternate_keys

    # Device 1 Genesis (seq 0)
    p1_0, _, s1_0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"dev": 1, "seq": 0},
        private_key=priv1,
        public_key=pub1,
        previous_receipt=None,
    )
    r1_0 = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": p1_0.model_dump(mode="json"),
            "signature": s1_0,
            "public_key_pem": pub1.decode("ascii"),
        },
    )
    assert r1_0.status_code == 201

    # Device 2 Genesis (seq 0) - independent sequence counter
    p2_0, _, s2_0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-02",
        model_sha256="a" * 64,
        input_hash="c" * 64,
        output={"dev": 2, "seq": 0},
        private_key=priv2,
        public_key=pub2,
        previous_receipt=None,
    )
    r2_0 = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": p2_0.model_dump(mode="json"),
            "signature": s2_0,
            "public_key_pem": pub2.decode("ascii"),
        },
    )
    assert r2_0.status_code == 201

    # Device 1 Block 1 (seq 1)
    p1_1, _, s1_1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-camera-01",
        model_sha256="a" * 64,
        input_hash="d" * 64,
        output={"dev": 1, "seq": 1},
        private_key=priv1,
        public_key=pub1,
        previous_receipt=p1_0,
    )
    r1_1 = client.post(
        "/api/v1/inference/receipts",
        json={
            "payload": p1_1.model_dump(mode="json"),
            "signature": s1_1,
            "public_key_pem": pub1.decode("ascii"),
        },
    )
    assert r1_1.status_code == 201

    # Query latest for each device
    dev1_latest = client.get("/api/v1/inference/receipts/edge-camera-01/latest").json()
    dev2_latest = client.get("/api/v1/inference/receipts/edge-camera-02/latest").json()
    assert dev1_latest["sequence_number"] == 1
    assert dev2_latest["sequence_number"] == 0
