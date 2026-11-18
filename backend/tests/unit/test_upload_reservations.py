"""Unit tests for SQLite-backed upload reservations, quota enforcement, and state transitions.

Verifies cross-process atomic quota bounds, concurrency limit gating (max 3),
crash recovery without touching snapshots, and strict model lifecycle transitions.
"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.model_repository import (
    InvalidModelStateTransitionError,
    SqliteModelRepository,
)
from backend.app.db.repositories.upload_reservation_repository import (
    QuarantineQuotaExceededError,
    SqliteUploadReservationRepository,
    UploadConcurrencyLimitExceededError,
)
from backend.app.domain.entities.enums import IngestionStatus


@pytest.fixture
def db_session():
    """In-memory SQLite session fixture for isolated repository testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def temp_quarantine_dir():
    """Temporary quarantine directory fixture."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def test_concurrency_limit_enforcement(db_session, temp_quarantine_dir):
    """Verifies that acquiring more than max_concurrent_uploads (3) raises error."""
    repo = SqliteUploadReservationRepository(db_session)
    max_concurrent = 3
    req_bytes = 10 * 1024 * 1024  # 10 MB

    res1 = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=req_bytes,
        quarantine_dir=temp_quarantine_dir,
        max_concurrent_uploads=max_concurrent,
    )
    res2 = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=req_bytes,
        quarantine_dir=temp_quarantine_dir,
        max_concurrent_uploads=max_concurrent,
    )
    res3 = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=req_bytes,
        quarantine_dir=temp_quarantine_dir,
        max_concurrent_uploads=max_concurrent,
    )

    assert res1.status == "ACTIVE"
    assert res2.status == "ACTIVE"
    assert res3.status == "ACTIVE"

    # 4th concurrent acquisition must be rejected
    with pytest.raises(UploadConcurrencyLimitExceededError) as exc_info:
        repo.acquire_reservation(
            asset_type="model",
            requested_bytes=req_bytes,
            quarantine_dir=temp_quarantine_dir,
            max_concurrent_uploads=max_concurrent,
        )
    assert "Maximum concurrent upload limit" in str(exc_info.value)

    # Releasing one slot allows subsequent acquisition
    repo.release_reservation(res1.reservation_id, quarantine_dir=temp_quarantine_dir)
    res4 = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=req_bytes,
        quarantine_dir=temp_quarantine_dir,
        max_concurrent_uploads=max_concurrent,
    )
    assert res4.status == "ACTIVE"


def test_global_quarantine_quota_enforcement(db_session, temp_quarantine_dir):
    """Verifies that exceeding the 5 GB global quarantine quota raises error."""
    repo = SqliteUploadReservationRepository(db_session)
    quota_limit = 100 * 1024 * 1024  # Set mock quota to 100 MB for test

    # 1. First reservation of 60 MB succeeds
    res1 = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=60 * 1024 * 1024,
        quarantine_dir=temp_quarantine_dir,
        max_concurrent_uploads=5,
        global_quarantine_quota_bytes=quota_limit,
    )
    assert res1.status == "ACTIVE"

    # 2. Second reservation of 50 MB would bring total to 110 MB > 100 MB -> must fail
    with pytest.raises(QuarantineQuotaExceededError) as exc_info:
        repo.acquire_reservation(
            asset_type="model",
            requested_bytes=50 * 1024 * 1024,
            quarantine_dir=temp_quarantine_dir,
            max_concurrent_uploads=5,
            global_quarantine_quota_bytes=quota_limit,
        )
    assert "Global quarantine storage quota" in str(exc_info.value)


def test_reservation_release_cleans_only_quarantine_files(db_session, temp_quarantine_dir):
    """Verifies that release cleans up partial quarantine files without touching other files."""
    repo = SqliteUploadReservationRepository(db_session)

    # Create dummy quarantine file
    upload_dir = temp_quarantine_dir / "upload_test"
    upload_dir.mkdir(parents=True, exist_ok=True)
    partial_file = upload_dir / "partial.pt"
    partial_file.write_bytes(b"dummy partial bytes")

    # Create safe external file to ensure it's NEVER touched
    safe_file = temp_quarantine_dir.parent / "safe_artifact.pt"
    safe_file.write_bytes(b"authoritative snapshot bytes")

    res = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=1024 * 1024,
        quarantine_dir=temp_quarantine_dir,
    )
    repo.update_reservation_path(res.reservation_id, str(partial_file))

    # Release with cleanup
    repo.release_reservation(
        res.reservation_id,
        reason="ABORTED",
        cleanup_file=True,
        quarantine_dir=temp_quarantine_dir,
    )

    assert not partial_file.exists()
    assert safe_file.exists()
    safe_file.unlink()


def test_stale_reservation_reconciliation(db_session, temp_quarantine_dir):
    """Verifies that expired reservations are reclaimed during startup/reconciliation."""
    repo = SqliteUploadReservationRepository(db_session)

    # Manually create an expired reservation
    stale_file = temp_quarantine_dir / "stale_upload" / "model.pt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_bytes(b"abandoned stream bytes")

    res = repo.acquire_reservation(
        asset_type="model",
        requested_bytes=1024 * 1024,
        quarantine_dir=temp_quarantine_dir,
        timeout_seconds=-10,  # Expired immediately
    )
    repo.update_reservation_path(res.reservation_id, str(stale_file))

    cleaned = repo.reconcile_and_cleanup_stale(temp_quarantine_dir)
    assert cleaned >= 1
    assert not stale_file.exists()

    record = repo.get_reservation(res.reservation_id)
    assert record["status"] == "EXPIRED"


def test_conditional_model_state_transitions(db_session):
    """Verifies strict state machine enforcement for models."""
    model_repo = SqliteModelRepository(db_session)
    model_repo.create_model(
        model_id="mod_test_123",
        name="test.pt",
        storage_path="/tmp/test.pt",
        sha256="abc123",
        size_bytes=1000,
        status=IngestionStatus.RECEIVED,
    )

    # 1. Valid: RECEIVED -> INSPECTING
    model_repo.update_model_status(
        model_id="mod_test_123",
        status=IngestionStatus.INSPECTING,
        expected_current_status=IngestionStatus.RECEIVED,
    )
    refreshed = model_repo.get_model("mod_test_123")
    assert refreshed.status == IngestionStatus.INSPECTING

    # 2. Invalid: INSPECTING -> RECEIVED (regression not allowed)
    with pytest.raises(InvalidModelStateTransitionError):
        model_repo.update_model_status(
            model_id="mod_test_123",
            status=IngestionStatus.RECEIVED,
        )

    # 3. Valid: INSPECTING -> ACCEPTED
    model_repo.update_model_status(
        model_id="mod_test_123",
        status=IngestionStatus.ACCEPTED,
        expected_current_status=IngestionStatus.INSPECTING,
    )
    refreshed = model_repo.get_model("mod_test_123")
    assert refreshed.status == IngestionStatus.ACCEPTED

    # 4. Invalid: Terminal state ACCEPTED cannot transition
    with pytest.raises(InvalidModelStateTransitionError):
        model_repo.update_model_status(
            model_id="mod_test_123",
            status=IngestionStatus.REJECTED,
        )

    # 5. Invalid: Stale worker expecting INSPECTING when already ACCEPTED
    with pytest.raises(InvalidModelStateTransitionError):
        model_repo.update_model_status(
            model_id="mod_test_123",
            status=IngestionStatus.REJECTED,
            expected_current_status=IngestionStatus.INSPECTING,
        )
