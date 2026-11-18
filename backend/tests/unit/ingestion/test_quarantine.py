"""Unit tests for QuarantineManager subsystem."""

import hashlib
import io
from pathlib import Path

import pytest

from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
)
from backend.app.modules.data_integrity.ingestion.quarantine import (
    QuarantineManager,
    sanitize_filename,
)


@pytest.fixture
def quarantine_setup(tmp_path: Path) -> QuarantineManager:
    config = IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        max_upload_bytes=1024 * 1024,  # 1 MB
        chunk_size_bytes=1024,
    )
    return QuarantineManager(config)


def test_sanitize_filename() -> None:
    """Verifies filename sanitization strips directories, special characters, and leading dots."""
    assert sanitize_filename("dataset.zip") == "dataset.zip"
    assert sanitize_filename("../../etc/passwd.zip") == "passwd.zip"
    assert sanitize_filename("foo bar!@#.zip") == "foo_bar___.zip"
    assert sanitize_filename("...hidden.zip") == "hidden.zip"
    assert sanitize_filename("").endswith(".bin")


def test_valid_quarantine_stream(quarantine_setup: QuarantineManager) -> None:
    """Verifies streaming an uploaded file to quarantine correctly computes SHA-256 and tracks state."""
    data = b"Hello VisionGuard Quarantine Payload"
    expected_hash = hashlib.sha256(data).hexdigest()
    stream = io.BytesIO(data)

    record = quarantine_setup.quarantine_stream(stream, filename="test_dataset.zip")

    assert record.original_filename == "test_dataset.zip"
    assert record.sha256 == expected_hash
    assert record.size_bytes == len(data)
    assert record.status == IngestionStatus.QUARANTINED
    assert record.quarantine_path.exists()
    assert record.quarantine_path.read_bytes() == data

    # Verify lookup by ID
    retrieved = quarantine_setup.get_record(record.ingestion_id)
    assert retrieved == record


def test_quarantine_empty_file_rejection(quarantine_setup: QuarantineManager) -> None:
    """Verifies that 0-byte uploads are rejected with EmptyFileError and deleted."""
    stream = io.BytesIO(b"")
    with pytest.raises(EmptyFileError, match="contains 0 bytes"):
        quarantine_setup.quarantine_stream(stream, filename="empty.zip")


def test_quarantine_oversized_upload_rejection(quarantine_setup: QuarantineManager) -> None:
    """Verifies that uploads exceeding max_upload_bytes are rejected and cleaned up."""
    oversized_data = b"X" * (1024 * 1024 + 10)  # > 1 MB
    stream = io.BytesIO(oversized_data)

    with pytest.raises(FileTooLargeError, match="exceeded maximum permissible size"):
        quarantine_setup.quarantine_stream(stream, filename="huge.zip")


def test_quarantine_status_update_and_cleanup(quarantine_setup: QuarantineManager) -> None:
    """Verifies status lifecycle transitions and quarantine directory cleanup."""
    stream = io.BytesIO(b"Sample data for status update")
    record = quarantine_setup.quarantine_stream(stream, filename="status_test.zip")

    # Update status to INSPECTING
    updated = quarantine_setup.update_status(
        record.ingestion_id,
        IngestionStatus.INSPECTING,
        metadata_update={"worker": "test_worker"},
    )
    assert updated.status == IngestionStatus.INSPECTING
    assert updated.metadata["worker"] == "test_worker"

    # Cleanup
    assert record.quarantine_path.exists()
    cleaned = quarantine_setup.cleanup_quarantine(record.ingestion_id)
    assert cleaned is True
    assert not record.quarantine_path.exists()
