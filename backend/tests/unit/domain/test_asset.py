"""Unit tests for AssetSnapshot domain model."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType

_VALID_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_valid_asset_snapshot() -> None:
    """Verifies that a valid AssetSnapshot instantiates correctly."""
    now = datetime.now(UTC)
    snapshot = AssetSnapshot(
        snapshot_id="snap_123456",
        asset_id="asset_cifar10_clean",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256=_VALID_SHA256,
        created_at=now,
        file_count=50000,
        total_size_bytes=104857600,
        schema_version="1.0.0",
        metadata={"format": "CIFAR10_BINARY"},
    )
    assert snapshot.snapshot_id == "snap_123456"
    assert snapshot.asset_id == "asset_cifar10_clean"
    assert snapshot.asset_type == AssetType.DATASET_SNAPSHOT
    assert snapshot.manifest_sha256 == _VALID_SHA256
    assert snapshot.created_at == now
    assert snapshot.file_count == 50000
    assert snapshot.total_size_bytes == 104857600
    assert snapshot.metadata["format"] == "CIFAR10_BINARY"


def test_asset_snapshot_invalid_sha256() -> None:
    """Verifies rejection of non-64 hex or uppercase SHA-256 strings."""
    now = datetime.now(UTC)
    # Too short
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256="abcd123",
            created_at=now,
            file_count=10,
            total_size_bytes=1000,
        )

    # Uppercase hex
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256=_VALID_SHA256.upper(),
            created_at=now,
            file_count=10,
            total_size_bytes=1000,
        )

    # Non-hex characters
    invalid_hex = "g" * 64
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256=invalid_hex,
            created_at=now,
            file_count=10,
            total_size_bytes=1000,
        )


def test_asset_snapshot_negative_file_count() -> None:
    """Verifies rejection of negative file counts."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="file_count cannot be negative"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256=_VALID_SHA256,
            created_at=now,
            file_count=-1,
            total_size_bytes=1000,
        )


def test_asset_snapshot_negative_total_size() -> None:
    """Verifies rejection of negative file sizes."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="total_size_bytes cannot be negative"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256=_VALID_SHA256,
            created_at=now,
            file_count=10,
            total_size_bytes=-50,
        )


def test_asset_snapshot_naive_timestamp_rejection() -> None:
    """Verifies rejection of naive (timezone-unaware) datetime objects."""
    naive_dt = datetime.now()  # noqa: DTZ005 - purposefully testing naive rejection
    with pytest.raises(ValidationError, match="Datetime must be timezone-aware"):
        AssetSnapshot(
            snapshot_id="snap_123",
            asset_id="asset_123",
            asset_type=AssetType.DATASET_SNAPSHOT,
            manifest_sha256=_VALID_SHA256,
            created_at=naive_dt,
            file_count=10,
            total_size_bytes=1000,
        )


def test_asset_snapshot_immutability() -> None:
    """Verifies that AssetSnapshot is frozen and cannot be modified after creation."""
    now = datetime.now(UTC)
    snapshot = AssetSnapshot(
        snapshot_id="snap_123",
        asset_id="asset_123",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256=_VALID_SHA256,
        created_at=now,
        file_count=10,
        total_size_bytes=1000,
    )
    with pytest.raises(ValidationError):
        snapshot.file_count = 20  # type: ignore[misc]


def test_asset_snapshot_invalid_identifiers() -> None:
    """Verifies rejection of invalid identifiers containing path separators or invalid chars."""
    now = datetime.now(UTC)
    for bad_id in ["../evil", "snap/evil", "bad id", "s", ""]:
        with pytest.raises(ValidationError):
            AssetSnapshot(
                snapshot_id=bad_id,
                asset_id="asset_123",
                asset_type=AssetType.DATASET_SNAPSHOT,
                manifest_sha256=_VALID_SHA256,
                created_at=now,
                file_count=10,
                total_size_bytes=1000,
            )
