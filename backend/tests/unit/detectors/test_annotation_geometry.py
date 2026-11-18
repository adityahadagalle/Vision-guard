"""Unit tests for AnnotationGeometryValidator."""

import json
from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.data_integrity.detectors.annotation_geometry import (
    AnnotationGeometryValidator,
)


def _create_snapshot(
    tmp_path: Path, format_name: str, files: list[dict]
) -> tuple[Path, AssetSnapshot]:
    snap_dir = tmp_path / "snap_geom_test"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"files": files}
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    meta = {"original_format": format_name}
    (snap_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    snapshot = AssetSnapshot(
        snapshot_id="snap_aabbccddeeff0011",
        asset_id="asset_geom_test",
        asset_type=AssetType.DATASET_SNAPSHOT,
        manifest_sha256="c" * 64,
        created_at=datetime.now(UTC),
        file_count=len(files),
        total_size_bytes=1000,
        metadata={"snapshot_path": str(snap_dir), "format": format_name},
    )
    return snap_dir, snapshot


def test_coco_geometry_clean_dataset(tmp_path: Path) -> None:
    """Verifies that COCO dataset with valid contained bounding boxes produces zero findings."""
    snap_dir = tmp_path / "coco_clean"
    snap_dir.mkdir()
    anno_dir = snap_dir / "annotations"
    anno_dir.mkdir()

    coco_data = {
        "images": [{"id": 1, "file_name": "001.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 200.0, 150.0]}
        ],
    }
    (anno_dir / "instances.json").write_text(json.dumps(coco_data))

    files = [
        {"relative_path": "images/001.jpg", "sha256": "1" * 64},
        {"relative_path": "annotations/instances.json", "sha256": "2" * 64},
    ]
    _, snapshot = _create_snapshot(tmp_path, "COCO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0


def test_coco_geometry_box_exceeds_image_canvas(tmp_path: Path) -> None:
    """Verifies that a COCO bounding box extending beyond the image width/height is flagged."""
    snap_dir = tmp_path / "coco_out_of_bounds"
    snap_dir.mkdir()
    anno_dir = snap_dir / "annotations"
    anno_dir.mkdir()

    # Image is 640x480, but box x + w = 500 + 200 = 700 > 640
    coco_data = {
        "images": [{"id": 1, "file_name": "001.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [500.0, 50.0, 200.0, 100.0]}
        ],
    }
    (anno_dir / "instances.json").write_text(json.dumps(coco_data))

    files = [{"relative_path": "annotations/instances.json", "sha256": "2" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "COCO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1

    fnd = result.findings[0]
    assert fnd.category == FindingCategory.STRUCTURAL_DEFECT
    assert fnd.severity == SeverityLevel.HIGH
    assert "Exceeds Image Canvas" in fnd.title
    assert fnd.affected_assets[0].sample_path == "001.jpg"


def test_coco_geometry_non_positive_dimension(tmp_path: Path) -> None:
    """Verifies that non-positive bounding box width or height is detected."""
    snap_dir = tmp_path / "coco_zero_dim"
    snap_dir.mkdir()
    anno_dir = snap_dir / "annotations"
    anno_dir.mkdir()

    coco_data = {
        "images": [{"id": 1, "file_name": "001.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 0.0, 50.0]}
        ],
    }
    (anno_dir / "instances.json").write_text(json.dumps(coco_data))

    files = [{"relative_path": "annotations/instances.json", "sha256": "2" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "COCO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1
    assert "Non-Positive" in result.findings[0].title


def test_yolo_geometry_clean_dataset(tmp_path: Path) -> None:
    """Verifies that YOLO dataset with valid normalized bounding boxes produces zero findings."""
    snap_dir = tmp_path / "yolo_clean"
    snap_dir.mkdir()
    lbl_dir = snap_dir / "labels"
    lbl_dir.mkdir()

    # Center (0.5, 0.5), w=0.2, h=0.3 -> corners in [0.4, 0.6] -> well inside [0, 1]
    (lbl_dir / "001.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    files = [{"relative_path": "labels/001.txt", "sha256": "3" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "YOLO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0


def test_yolo_geometry_extent_exceeds_unit_interval(tmp_path: Path) -> None:
    """Verifies that YOLO box with center inside [0, 1] but extent corner > 1.0 is flagged."""
    snap_dir = tmp_path / "yolo_extent_overflow"
    snap_dir.mkdir()
    lbl_dir = snap_dir / "labels"
    lbl_dir.mkdir()

    # Center is 0.95 (inside [0, 1]), but width is 0.2 -> x_max = 0.95 + 0.1 = 1.05 > 1.0
    (lbl_dir / "001.txt").write_text("0 0.95 0.5 0.2 0.2\n")

    files = [{"relative_path": "labels/001.txt", "sha256": "3" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "YOLO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1

    fnd = result.findings[0]
    assert fnd.category == FindingCategory.STRUCTURAL_DEFECT
    assert "Extent Outside [0, 1]" in fnd.title


def test_yolo_geometry_center_out_of_range(tmp_path: Path) -> None:
    """Verifies that YOLO box with center outside [0, 1] is flagged."""
    snap_dir = tmp_path / "yolo_bad_center"
    snap_dir.mkdir()
    lbl_dir = snap_dir / "labels"
    lbl_dir.mkdir()

    # Center is 1.5 > 1.0
    (lbl_dir / "001.txt").write_text("0 1.5 0.5 0.2 0.2\n")

    files = [{"relative_path": "labels/001.txt", "sha256": "3" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "YOLO_DETECTION", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.COMPLETED
    assert any("Center Coordinate Out of [0, 1]" in f.title for f in result.findings)


def test_annotation_geometry_not_applicable_for_classification(tmp_path: Path) -> None:
    """Verifies that classification datasets return NOT_APPLICABLE status."""
    snap_dir = tmp_path / "classification"
    snap_dir.mkdir()
    files = [{"relative_path": "cats/cat1.jpg", "sha256": "4" * 64}]
    _, snapshot = _create_snapshot(tmp_path, "FOLDER_PER_CLASS", files)
    detector = AnnotationGeometryValidator()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    result = detector.evaluate(context)
    assert result.status == CapabilityStatus.NOT_APPLICABLE
    assert "not applicable to classification datasets" in (result.status_reason or "")
