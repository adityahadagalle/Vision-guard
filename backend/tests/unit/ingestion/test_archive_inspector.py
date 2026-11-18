"""Adversarial and defensive unit tests for DefensiveArchiveInspector."""

import zipfile
from pathlib import Path

import pytest

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    AlternateDataStreamError,
    DecompressionBombError,
    DuplicateEntryError,
    EntryCountLimitExceededError,
    MalformedArchiveError,
    PathTraversalError,
    ReservedDeviceNameError,
    SymlinkNotPermittedError,
)
from backend.app.modules.data_integrity.ingestion.inspector import DefensiveArchiveInspector


def _create_zip(tmp_path: Path, entries: dict[str, bytes], zip_name: str = "test.zip") -> Path:
    """Helper to construct in-memory/disk ZIP archives for testing."""
    zip_path = tmp_path / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return zip_path


def test_valid_zip_inspection(tmp_path: Path) -> None:
    """Verifies that a well-formed ZIP archive passes inspection."""
    zip_path = _create_zip(
        tmp_path,
        {
            "images/001.jpg": b"fake_jpeg_1",
            "images/002.jpg": b"fake_jpeg_2",
            "annotations.json": b'{"images": [], "annotations": []}',
        },
    )

    result = DefensiveArchiveInspector.inspect_zip(zip_path)
    assert result.total_entries == 3
    assert len(result.entries) == 3
    assert "images/001.jpg" in result.entries
    assert "annotations.json" in result.entries


def test_zip_slip_parent_traversal_rejection(tmp_path: Path) -> None:
    """Verifies rejection of directory traversal paths in ZIP entries (ZIP Slip)."""
    for bad_name in ["../escape.txt", "images/../../evil.sh", "a/b/../../../etc/passwd"]:
        zip_path = _create_zip(
            tmp_path, {bad_name: b"malicious content"}, zip_name=f"slip_{hash(bad_name)}.zip"
        )
        with pytest.raises(PathTraversalError, match="Path traversal component"):
            DefensiveArchiveInspector.inspect_zip(zip_path)


def test_absolute_path_rejection(tmp_path: Path) -> None:
    """Verifies rejection of absolute paths in ZIP entries."""
    for abs_path in ["/etc/passwd", "/images/001.jpg", "C:/Windows/cmd.exe"]:
        zip_path = _create_zip(
            tmp_path, {abs_path: b"payload"}, zip_name=f"abs_{hash(abs_path)}.zip"
        )
        with pytest.raises(PathTraversalError, match="Absolute entry paths are strictly forbidden"):
            DefensiveArchiveInspector.inspect_zip(zip_path)


def test_windows_reserved_device_names_rejection(tmp_path: Path) -> None:
    """Verifies rejection of Windows device names in entry paths."""
    for dev in ["con", "prn.txt", "aux", "nul.json", "com1", "data/COM2/sample.jpg", "lpt1"]:
        zip_path = _create_zip(tmp_path, {dev: b"device_payload"}, zip_name=f"dev_{hash(dev)}.zip")
        with pytest.raises(ReservedDeviceNameError, match="reserved Windows device name"):
            DefensiveArchiveInspector.inspect_zip(zip_path)


def test_alternate_data_stream_rejection(tmp_path: Path) -> None:
    """Verifies rejection of NTFS Alternate Data Streams."""
    zip_path = _create_zip(tmp_path, {"sample.jpg:hidden_payload": b"stealth"})
    with pytest.raises(AlternateDataStreamError, match="Alternate Data Stream"):
        DefensiveArchiveInspector.inspect_zip(zip_path)


def test_trailing_space_or_dot_rejection(tmp_path: Path) -> None:
    """Verifies rejection of path segments with trailing dots or spaces."""
    for bad in ["images./001.jpg", "images /001.jpg", "file.txt."]:
        zip_path = _create_zip(tmp_path, {bad: b"payload"}, zip_name=f"trailing_{hash(bad)}.zip")
        with pytest.raises(PathTraversalError, match="trailing space or dot"):
            DefensiveArchiveInspector.inspect_zip(zip_path)


def test_case_insensitive_duplicate_collision_rejection(tmp_path: Path) -> None:
    """Verifies rejection of duplicate or case-colliding entries."""
    zip_path = tmp_path / "case_collision.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Image.jpg", b"image 1")
        zf.writestr("image.jpg", b"image 2")  # Collision on case-insensitive filesystems

    with pytest.raises(DuplicateEntryError, match="Duplicate or case-colliding entry"):
        DefensiveArchiveInspector.inspect_zip(zip_path)


def test_symlink_entry_rejection(tmp_path: Path) -> None:
    """Verifies that ZIP entries with symlink attributes are rejected."""
    zip_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo("symlink_entry")
        # Unix mode for symlink: S_IFLNK (0o120000) | 0o777
        info.external_attr = 0o120777 << 16
        zf.writestr(info, b"/etc/passwd")

    with pytest.raises(SymlinkNotPermittedError, match="Symlink entries are strictly forbidden"):
        DefensiveArchiveInspector.inspect_zip(zip_path)


def test_excessive_entry_count_rejection(tmp_path: Path) -> None:
    """Verifies rejection when entry count exceeds max_file_count limit."""
    config = IngestionConfig(max_file_count=5)
    entries = {f"file_{i}.txt": b"content" for i in range(10)}
    zip_path = _create_zip(tmp_path, entries, zip_name="too_many_entries.zip")

    with pytest.raises(EntryCountLimitExceededError, match="entry count"):
        DefensiveArchiveInspector.inspect_zip(zip_path, config=config)


def test_decompression_bomb_uncompressed_size_rejection(tmp_path: Path) -> None:
    """Verifies rejection when total uncompressed size exceeds max_uncompressed_bytes."""
    config = IngestionConfig(max_uncompressed_bytes=1000)
    entries = {"big.txt": b"A" * 2000}
    zip_path = _create_zip(tmp_path, entries, zip_name="bomb.zip")

    with pytest.raises(DecompressionBombError, match="Archive uncompressed size exceeds"):
        DefensiveArchiveInspector.inspect_zip(zip_path, config=config)


def test_malformed_zip_rejection(tmp_path: Path) -> None:
    """Verifies rejection of non-ZIP or corrupted files."""
    corrupt_path = tmp_path / "corrupted.zip"
    corrupt_path.write_bytes(b"PK\x03\x04This is not a real zip central directory")

    with pytest.raises(MalformedArchiveError):
        DefensiveArchiveInspector.inspect_zip(corrupt_path)
