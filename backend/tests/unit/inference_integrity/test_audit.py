"""Unit tests for offline chain audit verification engine and tamper detection suite."""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.inference_repository import SqliteInferenceRepository
from backend.app.db.session import _set_sqlite_pragmas
from backend.app.modules.inference_integrity.audit import ChainAuditEngine
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.chain import MonotonicHashChainEngine
from backend.app.modules.inference_integrity.signer import InferenceSigner


@pytest.fixture
def sqlite_session(tmp_path: Path) -> Generator[Session, None, None]:
    """Provides an isolated SQLite database session with foreign keys and schema initialized."""
    db_file = tmp_path / "test_audit.db"
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
    """Generates an Ed25519 keypair for edge-camera-01."""
    return InferenceSigner.generate_keypair("edge-camera-01")


@pytest.fixture
def populated_chain_repo(
    sqlite_session: Session, device_keypair: tuple[bytes, bytes]
) -> tuple[SqliteInferenceRepository, list[str]]:
    """Populates the database with a 5-receipt valid hash chain (seq 0 to 4)."""
    repo = SqliteInferenceRepository(sqlite_session)
    priv, pub = device_keypair
    pub_pem_str = pub.decode("ascii")

    receipt_ids: list[str] = []
    prev_payload = None

    for i in range(5):
        payload, p_hash, sig_b64 = MonotonicHashChainEngine.construct_next_receipt(
            device_id="edge-camera-01",
            model_sha256="a" * 64,
            input_hash=f"{i:064x}",
            output={"step": i, "detections": [{"class": "person", "score": 0.95 + i * 0.01}]},
            private_key=priv,
            public_key=pub,
            previous_receipt=prev_payload,
        )
        model = repo.create_receipt(
            edge_device_id=payload.edge_device_id,
            timestamp_iso=payload.timestamp_iso,
            sequence_number=payload.sequence_number,
            model_sha256=payload.model_sha256,
            input_hash=payload.input_hash,
            inference_output=payload.inference_output,
            previous_receipt_hash=payload.previous_receipt_hash,
            payload_hash=p_hash,
            signature_b64=sig_b64,
            public_key_pem=pub_pem_str,
        )
        receipt_ids.append(model.id)
        prev_payload = payload

    return repo, receipt_ids


def test_valid_chain_audit_succeeds(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
) -> None:
    """Tests that a clean, untampered 5-receipt chain returns IS_VALID=True."""
    repo, _ = populated_chain_repo

    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is True
    assert result.IS_VALID is True
    assert bool(result) is True
    assert result.device_id == "edge-camera-01"
    assert result.verified_count == 5
    assert result.break_sequence_number is None
    assert result.reason is None
    assert result.error_code is None


def test_mutated_inference_output_detected(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
    sqlite_session: Session,
) -> None:
    """Tests that tampering with a record's inference_output causes signature verification to fail."""
    repo, receipt_ids = populated_chain_repo

    # Mutate middle record (sequence 2) in the SQLite database directly
    target_id = receipt_ids[2]
    sqlite_session.execute(
        text("UPDATE inference_receipts SET inference_output = :output WHERE id = :rid"),
        {"output": '{"tampered": true, "step": 2}', "rid": target_id},
    )
    sqlite_session.commit()

    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is False
    assert result.IS_VALID is False
    assert bool(result) is False
    assert result.break_sequence_number == 2
    # The payload hash or signature verification will fail on the mutated output
    assert result.error_code in ("SIGNATURE_MISMATCH", "PAYLOAD_HASH_MISMATCH")
    assert result.verified_count == 2  # Records 0 and 1 passed


def test_deleted_middle_record_sequence_gap_detected(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
    sqlite_session: Session,
) -> None:
    """Tests that deleting a record from the database triggers a Sequence Gap failure."""
    repo, receipt_ids = populated_chain_repo

    # Delete record at sequence 2
    target_id = receipt_ids[2]
    sqlite_session.execute(
        text("DELETE FROM inference_receipts WHERE id = :rid"),
        {"rid": target_id},
    )
    sqlite_session.commit()

    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is False
    assert result.break_sequence_number == 3  # Sequence gap discovered at record 3 (expected 2)
    assert result.reason == "Sequence gap"
    assert result.error_code == "SEQUENCE_GAP"
    assert result.details["expected_sequence"] == 2
    assert result.details["actual_sequence"] == 3


def test_recalculated_signature_hash_link_break_detected(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
    sqlite_session: Session,
    device_keypair: tuple[bytes, bytes],
) -> None:
    """Tests that if an attacker re-signs a mutated record, the next record's hash link fails."""
    repo, receipt_ids = populated_chain_repo
    priv, _ = device_keypair

    # Fetch record at sequence 2
    r2 = repo.get_receipt_by_id(receipt_ids[2])
    assert r2 is not None

    # Mutate record 2 output and generate valid signature with the device key
    mutated_output = {"step": 2, "tampered": True}
    from backend.app.domain.entities.inference import InferenceReceiptPayload

    mutated_payload = InferenceReceiptPayload(
        edge_device_id=r2.edge_device_id,
        timestamp_iso=r2.timestamp_iso,
        sequence_number=r2.sequence_number,
        model_sha256=r2.model_sha256,
        input_hash=r2.input_hash,
        inference_output=mutated_output,
        previous_receipt_hash=r2.previous_receipt_hash,
    )
    new_hash = compute_payload_hash(mutated_payload)
    new_sig = InferenceSigner.sign_payload(mutated_payload, priv)

    # Update record 2 with the re-signed data in the database
    sqlite_session.execute(
        text(
            "UPDATE inference_receipts SET inference_output = :output, payload_hash = :phash, "
            "signature_b64 = :sig WHERE id = :rid"
        ),
        {
            "output": '{"step": 2, "tampered": true}',
            "phash": new_hash,
            "sig": new_sig,
            "rid": r2.id,
        },
    )
    sqlite_session.commit()

    # The audit must detect that record 3's previous_receipt_hash does NOT match the new payload_hash of record 2!
    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is False
    assert result.break_sequence_number == 3
    assert result.reason == "Hash link broken"
    assert result.error_code == "HASH_LINK_BROKEN"
    assert result.verified_count == 3  # Records 0, 1, 2 verified; break at 3


def test_genesis_deletion_detected(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
    sqlite_session: Session,
) -> None:
    """Tests that deleting the Genesis record causes an immediate genesis sequence error."""
    repo, receipt_ids = populated_chain_repo

    # Delete record 0
    sqlite_session.execute(
        text("DELETE FROM inference_receipts WHERE id = :rid"),
        {"rid": receipt_ids[0]},
    )
    sqlite_session.commit()

    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is False
    assert result.break_sequence_number == 1
    assert result.error_code == "GENESIS_SEQUENCE_ERROR"
    assert "genesis sequence must be 0" in result.reason


def test_public_key_substitution_detected(
    populated_chain_repo: tuple[SqliteInferenceRepository, list[str]],
    sqlite_session: Session,
) -> None:
    """Tests that replacing a public key in the middle of a chain is detected."""
    repo, receipt_ids = populated_chain_repo
    _, alt_pub = InferenceSigner.generate_keypair("edge-camera-02")
    alt_pub_pem = alt_pub.decode("ascii")

    # Substitute public key on record 2
    sqlite_session.execute(
        text("UPDATE inference_receipts SET public_key_pem = :pem WHERE id = :rid"),
        {"pem": alt_pub_pem, "rid": receipt_ids[2]},
    )
    sqlite_session.commit()

    result = ChainAuditEngine.verify_device_chain("edge-camera-01", repo)

    assert result.is_valid is False
    assert result.break_sequence_number == 2
    assert result.error_code in ("PUBLIC_KEY_MISMATCH", "SIGNATURE_MISMATCH")


def test_empty_chain_audit(sqlite_session: Session) -> None:
    """Tests auditing a device with zero records."""
    repo = SqliteInferenceRepository(sqlite_session)

    result = ChainAuditEngine.verify_device_chain("nonexistent-device", repo)

    assert result.is_valid is True
    assert result.verified_count == 0
    assert result.break_sequence_number is None
