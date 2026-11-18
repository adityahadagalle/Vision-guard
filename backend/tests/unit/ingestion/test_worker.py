"""Unit tests for isolated extraction worker subprocess boundary."""

import zipfile
from pathlib import Path

import pytest

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    WorkerExecutionError,
    WorkerIsolationUnavailableError,
    WorkerTimeoutError,
)
from backend.app.modules.data_integrity.ingestion.worker import SubprocessExtractionWorker


def _create_sample_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("images/001.jpg", b"fake image bytes 1")
        zf.writestr("images/002.jpg", b"fake image bytes 2")
        zf.writestr("annotations/data.json", b'{"key": "value"}')
    return zip_path


def test_worker_successful_extraction(tmp_path: Path) -> None:
    """Verifies that the worker successfully decompresses files into the designated staging root."""
    zip_path = _create_sample_zip(tmp_path)
    output_dir = tmp_path / "extracted_output"

    worker = SubprocessExtractionWorker()
    result = worker.extract(zip_path, output_dir)

    assert result.success is True
    assert result.exit_code == 0
    assert result.file_count == 3
    assert (output_dir / "images" / "001.jpg").exists()
    assert (output_dir / "images" / "002.jpg").exists()
    assert (output_dir / "annotations" / "data.json").exists()


def test_worker_timeout_rejection(tmp_path: Path) -> None:
    """Verifies that an exceeded timeout kills the child subprocess and cleans up staging."""
    zip_path = _create_sample_zip(tmp_path)
    output_dir = tmp_path / "timeout_output"

    # Set hard timeout to 0 seconds to immediately trigger timeout
    config = IngestionConfig(worker_timeout_seconds=0)
    worker = SubprocessExtractionWorker(config=config)

    with pytest.raises(WorkerTimeoutError, match="exceeded hard execution timeout"):
        worker.extract(zip_path, output_dir)

    # Output directory must be cleaned up
    assert not output_dir.exists()


def test_worker_hardened_isolation_unavailable(tmp_path: Path) -> None:
    """Verifies that require_hardened_isolation raises WorkerIsolationUnavailableError on non-container hosts."""
    zip_path = _create_sample_zip(tmp_path)
    output_dir = tmp_path / "hardened_output"

    worker = SubprocessExtractionWorker()
    # On Windows or unprivileged environments, hardened container isolation is unavailable
    with pytest.raises(
        WorkerIsolationUnavailableError, match="Hardened sandbox isolation is unavailable"
    ):
        worker.extract(zip_path, output_dir, require_hardened_isolation=True)

    assert not output_dir.exists()


def test_worker_corrupt_archive_handling(tmp_path: Path) -> None:
    """Verifies that corrupted archives fail safely inside worker without crashing supervisor."""
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"garbage not a zip file")
    output_dir = tmp_path / "corrupt_output"

    worker = SubprocessExtractionWorker()
    with pytest.raises(WorkerExecutionError):
        worker.extract(bad_zip, output_dir)

    assert not output_dir.exists()


def test_worker_with_verified_sha256_succeeds(tmp_path: Path) -> None:
    """Verifies that extraction succeeds when expected_sha256 matches archive digest."""
    import hashlib

    zip_path = _create_sample_zip(tmp_path)
    output_dir = tmp_path / "valid_digest_output"
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    worker = SubprocessExtractionWorker()
    result = worker.extract(zip_path, output_dir, expected_sha256=digest)
    assert result.success is True
    assert result.file_count == 3


def test_worker_archive_replacement_toctou_detected(tmp_path: Path) -> None:
    """Verifies that substituting an archive between inspection and extraction triggers digest mismatch."""
    zip_path = _create_sample_zip(tmp_path)
    output_dir = tmp_path / "toctou_output"
    fake_expected_digest = "a" * 64

    worker = SubprocessExtractionWorker()
    with pytest.raises(WorkerExecutionError, match="Archive digest mismatch"):
        worker.extract(zip_path, output_dir, expected_sha256=fake_expected_digest)

    assert not output_dir.exists()


def test_safe_extract_direct_invocation_symlink_rejection(tmp_path: Path) -> None:
    """Verifies that safe_extract directly catches and rejects symlink entries during extraction."""
    from backend.app.modules.data_integrity.ingestion.extract_cli import safe_extract

    zip_path = tmp_path / "symlink_direct.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zinfo = zipfile.ZipInfo("symlink_entry")
        zinfo.external_attr = 0o120777 << 16
        zf.writestr(zinfo, b"target/path")

    output_dir = tmp_path / "direct_symlink_out"
    result = safe_extract(zip_path, output_dir, max_bytes=10_000_000, max_files=100)
    assert result["success"] is False
    assert result["code"] == "SYMLINK_NOT_PERMITTED"


def test_safe_extract_direct_invocation_case_collision_rejection(tmp_path: Path) -> None:
    """Verifies that safe_extract directly catches and rejects case-colliding entries during extraction."""
    from backend.app.modules.data_integrity.ingestion.extract_cli import safe_extract

    zip_path = tmp_path / "collision_direct.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("File.txt", b"content 1")
        zf.writestr("file.txt", b"content 2")

    output_dir = tmp_path / "direct_collision_out"
    result = safe_extract(zip_path, output_dir, max_bytes=10_000_000, max_files=100)
    assert result["success"] is False
    assert result["code"] == "DUPLICATE_ENTRY_DETECTED"
