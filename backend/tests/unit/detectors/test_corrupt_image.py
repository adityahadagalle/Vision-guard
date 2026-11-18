"""Unit tests for CorruptImageDetector."""

import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.data_integrity.detectors.corrupt_image import (
    CorruptImageDetector,
)


def _create_snapshot_fixture(tmp_path: Path) -> tuple[Path, AssetSnapshot]:
    snap_dir = tmp_path / "snap_test"
    snap_dir.mkdir(parents=True, exist_ok=True)
    images_dir = snap_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Valid image
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(images_dir / "valid.jpg", format="JPEG")

    files = [{"relative_path": "images/valid.jpg", "sha256": "a" * 64, "size_bytes": 500}]
    manifest = {"files": files}
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    snapshot = AssetSnapshot(
        snapshot_id="snap_1234567890abcdef",
        asset_id="asset_test",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256="b" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=500,
        metadata={"snapshot_path": str(snap_dir)},
    )
    return snap_dir, snapshot


def test_corrupt_image_clean_snapshot(tmp_path: Path) -> None:
    """Verifies that an uncorrupted dataset snapshot produces zero corruption findings."""
    snap_dir, snapshot = _create_snapshot_fixture(tmp_path)
    detector = CorruptImageDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(snap_dir)},
    )

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert result.metadata["scanned_images"] == 1
    assert result.metadata["corrupted_count"] == 0


def test_corrupt_image_truncated_file_detection(tmp_path: Path) -> None:
    """Verifies that truncated or damaged image bytes produce a CORRUPTED_FILE finding."""
    snap_dir, snapshot = _create_snapshot_fixture(tmp_path)

    # Add truncated image
    corrupt_file = snap_dir / "images" / "corrupt.jpg"
    corrupt_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00truncated")

    # Update manifest
    manifest_path = snap_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["files"].append(
        {"relative_path": "images/corrupt.jpg", "sha256": "c" * 64, "size_bytes": 20}
    )
    manifest_path.write_text(json.dumps(manifest_data))

    detector = CorruptImageDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(snap_dir)},
    )

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1

    fnd = result.findings[0]
    assert fnd.category == FindingCategory.CORRUPTED_FILE
    assert fnd.severity == SeverityLevel.HIGH
    assert fnd.confidence == 1.0
    assert "images/corrupt.jpg" in fnd.title
    assert len(fnd.affected_assets) == 1
    assert fnd.affected_assets[0].sample_path == "images/corrupt.jpg"
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_id == fnd.evidence_ids[0]


def test_corrupt_image_empty_file_detection(tmp_path: Path) -> None:
    """Verifies that a 0-byte image file is flagged as corrupted."""
    snap_dir, snapshot = _create_snapshot_fixture(tmp_path)

    empty_file = snap_dir / "images" / "empty.png"
    empty_file.write_bytes(b"")

    manifest_path = snap_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["files"].append(
        {"relative_path": "images/empty.png", "sha256": "0" * 64, "size_bytes": 0}
    )
    manifest_path.write_text(json.dumps(manifest_data))

    detector = CorruptImageDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(snap_dir)},
    )

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert any("images/empty.png" in f.title for f in result.findings)


def test_corrupt_image_not_applicable_when_no_images(tmp_path: Path) -> None:
    """Verifies that snapshots containing zero images return NOT_APPLICABLE status."""
    snap_dir = tmp_path / "text_only_snap"
    snap_dir.mkdir()
    manifest = {"files": [{"relative_path": "notes.txt", "sha256": "f" * 64}]}
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    snapshot = AssetSnapshot(
        snapshot_id="snap_textonly000000",
        asset_id="asset_text",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256="d" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=10,
    )

    detector = CorruptImageDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(snap_dir)},
    )

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.NOT_APPLICABLE
    assert "No image files found" in (result.status_reason or "")
