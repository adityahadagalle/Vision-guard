"""Unit tests for ExactDuplicateDetector."""

import json
from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.data_integrity.detectors.exact_duplicate import (
    ExactDuplicateDetector,
)


def _create_snapshot(tmp_path: Path, files: list[dict]) -> tuple[Path, AssetSnapshot]:
    snap_dir = tmp_path / "snap_dup_test"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"files": files}
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    snapshot = AssetSnapshot(
        snapshot_id="snap_1122334455667788",
        asset_id="asset_dup_test",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256="e" * 64,
        created_at=datetime.now(UTC),
        file_count=len(files),
        total_size_bytes=1000,
        metadata={"snapshot_path": str(snap_dir)},
    )
    return snap_dir, snapshot


def test_exact_duplicate_clean_dataset(tmp_path: Path) -> None:
    """Verifies that datasets with strictly unique image hashes produce zero findings."""
    files = [
        {"relative_path": "images/train/001.jpg", "sha256": "1" * 64},
        {"relative_path": "images/train/002.jpg", "sha256": "2" * 64},
        {"relative_path": "images/val/001.jpg", "sha256": "3" * 64},
    ]
    snap_dir, snapshot = _create_snapshot(tmp_path, files)
    detector = ExactDuplicateDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert result.metadata["duplicate_groups_count"] == 0


def test_exact_duplicate_intra_split(tmp_path: Path) -> None:
    """Verifies that duplicates within the same split are flagged with DUPLICATE_SAMPLE and MEDIUM severity."""
    shared_hash = "a" * 64
    files = [
        {"relative_path": "images/train/001.jpg", "sha256": shared_hash},
        {"relative_path": "images/train/002_copy.jpg", "sha256": shared_hash},
        {"relative_path": "images/train/003.jpg", "sha256": "b" * 64},
    ]
    snap_dir, snapshot = _create_snapshot(tmp_path, files)
    detector = ExactDuplicateDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1

    fnd = result.findings[0]
    assert fnd.category == FindingCategory.DUPLICATE_SAMPLE
    assert fnd.severity == SeverityLevel.MEDIUM
    assert fnd.suggested_disposition == SuggestedDisposition.REVIEW
    assert "Intra-Split" in fnd.title
    assert len(fnd.affected_assets) == 2
    paths = [a.sample_path for a in fnd.affected_assets]
    assert "images/train/001.jpg" in paths
    assert "images/train/002_copy.jpg" in paths


def test_exact_duplicate_critical_cross_split_leakage(tmp_path: Path) -> None:
    """Verifies that duplicates spanning distinct splits are elevated to SPLIT_LEAKAGE and CRITICAL severity."""
    shared_hash = "f" * 64
    files = [
        {"relative_path": "images/train/target.png", "sha256": shared_hash},
        {"relative_path": "images/val/target.png", "sha256": shared_hash},
        {"relative_path": "images/test/target.png", "sha256": shared_hash},
        {"relative_path": "images/train/unique.png", "sha256": "1" * 64},
    ]
    snap_dir, snapshot = _create_snapshot(tmp_path, files)
    detector = ExactDuplicateDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1

    fnd = result.findings[0]
    assert fnd.category == FindingCategory.SPLIT_LEAKAGE
    assert fnd.severity == SeverityLevel.CRITICAL
    assert fnd.suggested_disposition == SuggestedDisposition.QUARANTINE
    assert "Cross-Split" in fnd.title
    assert len(fnd.affected_assets) == 3
    assert result.metadata["cross_split_leakage_groups"] == 1


def test_exact_duplicate_not_applicable_when_no_images(tmp_path: Path) -> None:
    """Verifies NOT_APPLICABLE status when snapshot has zero images."""
    files = [{"relative_path": "data.csv", "sha256": "9" * 64}]
    snap_dir, snapshot = _create_snapshot(tmp_path, files)
    detector = ExactDuplicateDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.NOT_APPLICABLE
    assert "No image files found" in (result.status_reason or "")
