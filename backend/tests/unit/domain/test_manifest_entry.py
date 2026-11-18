"""Unit tests for ManifestEntry domain model."""

import pytest
from pydantic import ValidationError

from backend.app.domain.entities.asset import ManifestEntry

_VALID_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_valid_manifest_entry() -> None:
    """Verifies that a valid ManifestEntry instantiates and normalizes properly."""
    entry = ManifestEntry(
        relative_path="images/train/0001.jpg",
        sha256=_VALID_SHA256,
        size_bytes=204800,
        mime_type="image/jpeg",
        metadata={"width": 640, "height": 480},
    )
    assert entry.relative_path == "images/train/0001.jpg"
    assert entry.sha256 == _VALID_SHA256
    assert entry.size_bytes == 204800
    assert entry.mime_type == "image/jpeg"
    assert entry.metadata["width"] == 640


def test_manifest_entry_path_normalization() -> None:
    """Verifies deterministic path normalization across Windows and POSIX conventions."""
    # Backslash conversion
    entry1 = ManifestEntry(
        relative_path=r"images\val\0002.png",
        sha256=_VALID_SHA256,
        size_bytes=100,
    )
    assert entry1.relative_path == "images/val/0002.png"

    # Leading redundant ./ stripping
    entry2 = ManifestEntry(
        relative_path="./annotations/instances.json",
        sha256=_VALID_SHA256,
        size_bytes=100,
    )
    assert entry2.relative_path == "annotations/instances.json"

    # Multiple redundant leading ./
    entry3 = ManifestEntry(
        relative_path="././data/labels.txt",
        sha256=_VALID_SHA256,
        size_bytes=100,
    )
    assert entry3.relative_path == "data/labels.txt"


def test_manifest_entry_absolute_path_rejection() -> None:
    """Verifies rejection of Unix and Windows absolute paths."""
    for abs_path in [
        "/etc/passwd",
        "/images/001.jpg",
        r"\Windows\System32",
        "C:/data/img.jpg",
        "D:\\test",
    ]:
        with pytest.raises(ValidationError):
            ManifestEntry(
                relative_path=abs_path,
                sha256=_VALID_SHA256,
                size_bytes=100,
            )


def test_manifest_entry_parent_traversal_rejection() -> None:
    """Verifies rejection of directory traversal attempts (..)."""
    for bad_path in [
        "../secret.txt",
        "images/../../etc/passwd",
        "a/b/../..",
        r"images\..\evil.jpg",
    ]:
        with pytest.raises(ValidationError, match="Path traversal component"):
            ManifestEntry(
                relative_path=bad_path,
                sha256=_VALID_SHA256,
                size_bytes=100,
            )


def test_manifest_entry_empty_path_rejection() -> None:
    """Verifies rejection of empty or whitespace-only paths."""
    for empty in ["", "   ", "\t", "\n"]:
        with pytest.raises(ValidationError):
            ManifestEntry(
                relative_path=empty,
                sha256=_VALID_SHA256,
                size_bytes=100,
            )


def test_manifest_entry_windows_device_names_rejection() -> None:
    """Verifies rejection of Windows reserved device names."""
    for device in ["con", "prn", "aux", "nul", "com1", "lpt1", "images/CON/sample.jpg", "AUX.txt"]:
        with pytest.raises(ValidationError, match="Reserved Windows device name"):
            ManifestEntry(
                relative_path=device,
                sha256=_VALID_SHA256,
                size_bytes=100,
            )


def test_manifest_entry_windows_ads_rejection() -> None:
    """Verifies rejection of Windows Alternate Data Streams."""
    with pytest.raises(ValidationError, match="Windows Alternate Data Streams"):
        ManifestEntry(
            relative_path="image.jpg:hidden_stream",
            sha256=_VALID_SHA256,
            size_bytes=100,
        )


def test_manifest_entry_invalid_sha256() -> None:
    """Verifies rejection of non-SHA256 strings."""
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        ManifestEntry(
            relative_path="image.jpg",
            sha256="not_a_valid_sha256",
            size_bytes=100,
        )


def test_manifest_entry_negative_size() -> None:
    """Verifies rejection of negative file sizes."""
    with pytest.raises(ValidationError, match="File size cannot be negative"):
        ManifestEntry(
            relative_path="image.jpg",
            sha256=_VALID_SHA256,
            size_bytes=-10,
        )


def test_manifest_entry_immutability() -> None:
    """Verifies that ManifestEntry is frozen and immutable."""
    entry = ManifestEntry(
        relative_path="image.jpg",
        sha256=_VALID_SHA256,
        size_bytes=100,
    )
    with pytest.raises(ValidationError):
        entry.size_bytes = 200  # type: ignore[misc]
