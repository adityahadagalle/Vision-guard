"""Unit tests for monotonic hash-chaining engine and SQLite receipt persistence."""

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.inference_repository import (
    DuplicateReceiptError,
    SqliteInferenceRepository,
)
from backend.app.db.session import _set_sqlite_pragmas
from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.chain import (
    ChainDeviceMismatchError,
    ChainLinkError,
    ChainSequenceError,
    ChainSignatureVerificationError,
    MonotonicHashChainEngine,
)
from backend.app.modules.inference_integrity.signer import InferenceSigner


@pytest.fixture
def sqlite_session(tmp_path: Path) -> Generator[Session, None, None]:
    """Provides an isolated SQLite database session with foreign keys and schema initialized."""
    db_file = tmp_path / "test_chain.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def device_keypair() -> tuple[bytes, bytes]:
    """Generates an Ed25519 keypair for edge-node-01."""
    return InferenceSigner.generate_keypair("edge-node-01")


@pytest.fixture
def alternate_keypair() -> tuple[bytes, bytes]:
    """Generates an alternate Ed25519 keypair for edge-node-02."""
    return InferenceSigner.generate_keypair("edge-node-02")


def test_genesis_receipt_construction(device_keypair: tuple[bytes, bytes]) -> None:
    """Tests constructing a genesis receipt (Rule 1: sequence=0, previous_receipt_hash=None)."""
    priv, pub = device_keypair
    model_sha = "a" * 64
    input_sha = "b" * 64
    output = {"label": "person", "score": 0.98}

    payload, payload_hash, signature_b64 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256=model_sha,
        input_hash=input_sha,
        output=output,
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )

    assert payload.edge_device_id == "edge-node-01"
    assert payload.sequence_number == 0
    assert payload.previous_receipt_hash is None
    assert payload.model_sha256 == model_sha
    assert payload.input_hash == input_sha
    assert payload.inference_output == output
    assert payload_hash == compute_payload_hash(payload)
    assert InferenceSigner.verify_signature(payload, signature_b64, pub) is True


def test_sequential_receipt_chaining(device_keypair: tuple[bytes, bytes]) -> None:
    """Tests constructing next receipt (n+1) and verifying cryptographic hash linkage."""
    priv, pub = device_keypair
    model_sha = "a" * 64
    input_sha = "b" * 64

    # 1. Genesis block (n=0)
    p0, h0, _ = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256=model_sha,
        input_hash=input_sha,
        output={"label": "person"},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )

    # 2. Block 1 (n=1)
    p1, h1, s1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256=model_sha,
        input_hash="c" * 64,
        output={"label": "vehicle"},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )

    assert p1.sequence_number == 1
    assert p1.previous_receipt_hash == h0
    assert h1 == compute_payload_hash(p1)
    assert InferenceSigner.verify_signature(p1, s1, pub) is True

    # 3. Block 2 (n=2)
    p2, h2, s2 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256=model_sha,
        input_hash="d" * 64,
        output={"label": "animal"},
        private_key=priv,
        public_key=pub,
        previous_receipt=p1,
    )

    assert p2.sequence_number == 2
    assert p2.previous_receipt_hash == h1
    assert h2 == compute_payload_hash(p2)
    assert InferenceSigner.verify_signature(p2, s2, pub) is True


def test_chain_device_mismatch_rejected(
    device_keypair: tuple[bytes, bytes], alternate_keypair: tuple[bytes, bytes]
) -> None:
    """Tests that linking receipts between different edge devices is strictly rejected."""
    priv1, pub1 = device_keypair
    priv2, pub2 = alternate_keypair

    # Genesis on device 1
    p0, _, _ = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "cat"},
        private_key=priv1,
        public_key=pub1,
        previous_receipt=None,
    )

    # Attempting to chain on device 2 with p0 from device 1
    with pytest.raises(ChainDeviceMismatchError) as exc_info:
        MonotonicHashChainEngine.construct_next_receipt(
            device_id="edge-node-02",
            model_sha256="a" * 64,
            input_hash="c" * 64,
            output={"label": "dog"},
            private_key=priv2,
            public_key=pub2,
            previous_receipt=p0,
        )

    assert "Device mismatch" in str(exc_info.value)


def test_invalid_keypair_verification_failure(
    device_keypair: tuple[bytes, bytes], alternate_keypair: tuple[bytes, bytes]
) -> None:
    """Tests that a mismatched public key triggers immediate verification failure."""
    priv1, _ = device_keypair
    _, pub2 = alternate_keypair

    with pytest.raises(ChainSignatureVerificationError):
        MonotonicHashChainEngine.construct_next_receipt(
            device_id="edge-node-01",
            model_sha256="a" * 64,
            input_hash="b" * 64,
            output={"label": "chair"},
            private_key=priv1,
            public_key=pub2,  # Wrong public key
            previous_receipt=None,
        )


def test_verify_receipt_link_rules(device_keypair: tuple[bytes, bytes]) -> None:
    """Tests strict verification rules between consecutive receipts."""
    priv, pub = device_keypair
    p0, h0, _ = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"step": 0},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    p1, _, _ = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="c" * 64,
        output={"step": 1},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )

    # Valid links
    MonotonicHashChainEngine.verify_receipt_link(p0, None)
    MonotonicHashChainEngine.verify_receipt_link(p1, p0)

    # Invalid genesis: non-zero sequence
    bad_genesis_seq = InferenceReceiptPayload(
        edge_device_id="edge-node-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=1,
        model_sha256="a" * 64,
        input_hash="b" * 64,
        inference_output={},
        previous_receipt_hash=None,
    )
    with pytest.raises(ChainSequenceError):
        MonotonicHashChainEngine.verify_receipt_link(bad_genesis_seq, None)

    # Invalid genesis: non-null previous hash
    bad_genesis_hash = InferenceReceiptPayload(
        edge_device_id="edge-node-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=0,
        model_sha256="a" * 64,
        input_hash="b" * 64,
        inference_output={},
        previous_receipt_hash="f" * 64,
    )
    with pytest.raises(ChainLinkError):
        MonotonicHashChainEngine.verify_receipt_link(bad_genesis_hash, None)

    # Sequence gap: jumping from 0 to 2
    gap_receipt = InferenceReceiptPayload(
        edge_device_id="edge-node-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=2,  # Expected 1
        model_sha256="a" * 64,
        input_hash="c" * 64,
        inference_output={},
        previous_receipt_hash=h0,
    )
    with pytest.raises(ChainSequenceError) as exc_info:
        MonotonicHashChainEngine.verify_receipt_link(gap_receipt, p0)
    assert "Discontinuous sequence" in str(exc_info.value)

    # Broken hash link
    tampered_hash_receipt = InferenceReceiptPayload(
        edge_device_id="edge-node-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=1,
        model_sha256="a" * 64,
        input_hash="c" * 64,
        inference_output={},
        previous_receipt_hash="0" * 64,  # Corrupted digest
    )
    with pytest.raises(ChainLinkError) as exc_info:
        MonotonicHashChainEngine.verify_receipt_link(tampered_hash_receipt, p0)
    assert "Broken hash chain link" in str(exc_info.value)


def test_verify_chain_and_subchain(device_keypair: tuple[bytes, bytes]) -> None:
    """Tests batch verification over an entire chain."""
    priv, pub = device_keypair
    chain: list[InferenceReceiptPayload] = []

    prev = None
    for i in range(5):
        payload, _, _ = MonotonicHashChainEngine.construct_next_receipt(
            device_id="edge-node-01",
            model_sha256="a" * 64,
            input_hash=f"{i:064x}",
            output={"index": i},
            private_key=priv,
            public_key=pub,
            previous_receipt=prev,
        )
        chain.append(payload)
        prev = payload

    # Full chain verification
    MonotonicHashChainEngine.verify_chain(chain)

    # Subchain verification (skipping genesis)
    MonotonicHashChainEngine.verify_subchain(chain[1:])

    # Corrupt chain at index 3
    corrupt_chain = list(chain)
    corrupt_chain[3] = InferenceReceiptPayload(
        edge_device_id="edge-node-01",
        timestamp_iso=corrupt_chain[3].timestamp_iso,
        sequence_number=99,  # Sequence violation
        model_sha256=corrupt_chain[3].model_sha256,
        input_hash=corrupt_chain[3].input_hash,
        inference_output=corrupt_chain[3].inference_output,
        previous_receipt_hash=corrupt_chain[3].previous_receipt_hash,
    )
    with pytest.raises(ChainSequenceError):
        MonotonicHashChainEngine.verify_chain(corrupt_chain)


def test_sqlite_repository_persistence_and_queries(
    sqlite_session: Session, device_keypair: tuple[bytes, bytes]
) -> None:
    """Tests persisting receipts and executing queries via SqliteInferenceRepository."""
    repo = SqliteInferenceRepository(sqlite_session)
    priv, pub = device_keypair
    pub_pem_str = pub.decode("ascii")

    # Initial query for unknown device
    assert repo.get_latest_receipt("edge-node-01") is None
    assert repo.count_receipts_for_device("edge-node-01") == 0

    # 1. Create genesis receipt
    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"label": "person", "score": 0.95},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    r0 = repo.create_receipt(
        edge_device_id=p0.edge_device_id,
        timestamp_iso=p0.timestamp_iso,
        sequence_number=p0.sequence_number,
        model_sha256=p0.model_sha256,
        input_hash=p0.input_hash,
        inference_output=p0.inference_output,
        previous_receipt_hash=p0.previous_receipt_hash,
        payload_hash=h0,
        signature_b64=s0,
        public_key_pem=pub_pem_str,
    )

    assert r0.id is not None
    assert r0.sequence_number == 0
    assert r0.previous_receipt_hash is None
    assert r0.payload_hash == h0

    # Query latest receipt
    latest = repo.get_latest_receipt("edge-node-01")
    assert latest is not None
    assert latest.sequence_number == 0
    assert latest.payload_hash == h0

    # 2. Create second receipt
    p1, h1, s1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="c" * 64,
        output={"label": "car", "score": 0.88},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )
    r1 = repo.create_receipt(
        edge_device_id=p1.edge_device_id,
        timestamp_iso=p1.timestamp_iso,
        sequence_number=p1.sequence_number,
        model_sha256=p1.model_sha256,
        input_hash=p1.input_hash,
        inference_output=p1.inference_output,
        previous_receipt_hash=p1.previous_receipt_hash,
        payload_hash=h1,
        signature_b64=s1,
        public_key_pem=pub_pem_str,
    )

    assert r1.sequence_number == 1
    assert r1.previous_receipt_hash == h0

    # Verify latest receipt updated
    latest_after = repo.get_latest_receipt("edge-node-01")
    assert latest_after is not None
    assert latest_after.sequence_number == 1
    assert latest_after.previous_receipt_hash == h0

    # Query lookups
    by_id = repo.get_receipt_by_id(r1.id)
    assert by_id is not None and by_id.id == r1.id

    by_hash = repo.get_receipt_by_payload_hash(h1)
    assert by_hash is not None and by_hash.sequence_number == 1

    by_seq = repo.get_receipt_by_sequence("edge-node-01", 0)
    assert by_seq is not None and by_seq.sequence_number == 0

    receipts_list = repo.list_receipts_for_device("edge-node-01")
    assert len(receipts_list) == 2
    assert receipts_list[0].sequence_number == 0
    assert receipts_list[1].sequence_number == 1
    assert repo.count_receipts_for_device("edge-node-01") == 2


def test_database_unique_sequence_constraint_rejected(
    sqlite_session: Session, device_keypair: tuple[bytes, bytes]
) -> None:
    """Tests that the database-level unique constraint on (edge_device_id, sequence_number) fundamentally rejects collisions."""
    repo = SqliteInferenceRepository(sqlite_session)
    priv, pub = device_keypair
    pub_pem_str = pub.decode("ascii")

    # Persist receipt sequence 0
    p0, h0, s0 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="b" * 64,
        output={"step": 0},
        private_key=priv,
        public_key=pub,
        previous_receipt=None,
    )
    repo.create_receipt(
        edge_device_id=p0.edge_device_id,
        timestamp_iso=p0.timestamp_iso,
        sequence_number=p0.sequence_number,
        model_sha256=p0.model_sha256,
        input_hash=p0.input_hash,
        inference_output=p0.inference_output,
        previous_receipt_hash=p0.previous_receipt_hash,
        payload_hash=h0,
        signature_b64=s0,
        public_key_pem=pub_pem_str,
    )

    # Attempt inserting duplicate sequence 0 on edge-node-01 with a different payload_hash
    with pytest.raises(DuplicateReceiptError) as exc_info:
        repo.create_receipt(
            edge_device_id="edge-node-01",
            timestamp_iso=datetime.now(UTC).isoformat(),
            sequence_number=0,  # Duplicate sequence number!
            model_sha256="a" * 64,
            input_hash="c" * 64,
            inference_output={"step": 0, "different": True},
            previous_receipt_hash=None,
            payload_hash="1" * 64,  # Unique hash, but sequence collides
            signature_b64=s0,
            public_key_pem=pub_pem_str,
        )

    assert "Receipt constraint violation" in str(exc_info.value)
    assert exc_info.value.details["sequence_number"] == 0

    # Confirm session recovered from rollback: subsequent valid insert for sequence 1 succeeds
    p1, h1, s1 = MonotonicHashChainEngine.construct_next_receipt(
        device_id="edge-node-01",
        model_sha256="a" * 64,
        input_hash="d" * 64,
        output={"step": 1},
        private_key=priv,
        public_key=pub,
        previous_receipt=p0,
    )
    r1 = repo.create_receipt(
        edge_device_id=p1.edge_device_id,
        timestamp_iso=p1.timestamp_iso,
        sequence_number=p1.sequence_number,
        model_sha256=p1.model_sha256,
        input_hash=p1.input_hash,
        inference_output=p1.inference_output,
        previous_receipt_hash=p1.previous_receipt_hash,
        payload_hash=h1,
        signature_b64=s1,
        public_key_pem=pub_pem_str,
    )
    assert r1.sequence_number == 1


def test_database_unique_payload_hash_constraint_rejected(
    sqlite_session: Session, device_keypair: tuple[bytes, bytes]
) -> None:
    """Tests that the database-level unique constraint on payload_hash rejects duplicate payloads."""
    repo = SqliteInferenceRepository(sqlite_session)
    _, pub = device_keypair
    pub_pem_str = pub.decode("ascii")

    # Insert receipt with payload_hash H
    repo.create_receipt(
        edge_device_id="edge-node-01",
        timestamp_iso=datetime.now(UTC).isoformat(),
        sequence_number=0,
        model_sha256="a" * 64,
        input_hash="b" * 64,
        inference_output={"data": 1},
        previous_receipt_hash=None,
        payload_hash="e" * 64,
        signature_b64="sig1",
        public_key_pem=pub_pem_str,
    )

    # Insert another receipt on edge-node-02 with the SAME payload_hash H
    with pytest.raises(DuplicateReceiptError) as exc_info:
        repo.create_receipt(
            edge_device_id="edge-node-02",
            timestamp_iso=datetime.now(UTC).isoformat(),
            sequence_number=0,
            model_sha256="a" * 64,
            input_hash="b" * 64,
            inference_output={"data": 1},
            previous_receipt_hash=None,
            payload_hash="e" * 64,  # Duplicate payload hash!
            signature_b64="sig2",
            public_key_pem=pub_pem_str,
        )

    assert "Receipt constraint violation" in str(exc_info.value)
