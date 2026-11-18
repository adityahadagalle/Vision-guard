"""Unit tests for dataset format detection and parsing adapters."""

import json
from pathlib import Path

import pytest
from PIL import Image

from backend.app.modules.data_integrity.ingestion.exceptions import (
    UnsupportedFormatError,
)
from backend.app.modules.data_integrity.ingestion.formats.detector import DatasetFormatDetector


def _create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(0, 255, 0))
    img.save(path, format="JPEG")


def test_coco_format_detection_and_parsing(tmp_path: Path) -> None:
    """Verifies that a valid COCO dataset is detected and parsed with absolute bounding boxes."""
    root = tmp_path / "coco_dataset"
    root.mkdir()

    # Images
    _create_dummy_image(root / "images" / "0001.jpg")
    _create_dummy_image(root / "images" / "0002.jpg")

    # Annotations
    anno_data = {
        "images": [
            {"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "0002.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "vehicle"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 50.0, 80.0],
                "area": 4000.0,
            },
            {
                "id": 102,
                "image_id": 2,
                "category_id": 2,
                "bbox": [100.0, 150.0, 200.0, 120.0],
                "area": 24000.0,
            },
        ],
    }
    anno_file = root / "annotations" / "instances.json"
    anno_file.parent.mkdir()
    anno_file.write_text(json.dumps(anno_data), encoding="utf-8")

    detector = DatasetFormatDetector()
    adapter = detector.detect(root)
    assert adapter.format_name == "COCO_DETECTION"

    parsed = adapter.parse_and_validate(root)
    assert len(parsed.samples) == 2
    assert "person" in parsed.categories
    assert "vehicle" in parsed.categories
    assert len(parsed.samples[0].bounding_boxes) == 1
    assert parsed.samples[0].bounding_boxes[0].coordinate_system == "ABSOLUTE_PIXELS"
    assert parsed.samples[0].bounding_boxes[0].bbox == [10.0, 20.0, 50.0, 80.0]


def test_yolo_format_detection_and_parsing(tmp_path: Path) -> None:
    """Verifies that a YOLO dataset with paired images/ and labels/ is detected and parsed."""
    root = tmp_path / "yolo_dataset"
    root.mkdir()

    # Images
    _create_dummy_image(root / "images" / "img_01.jpg")
    _create_dummy_image(root / "images" / "img_02.jpg")

    # Labels
    lbl_dir = root / "labels"
    lbl_dir.mkdir()
    # img_01 has 1 bounding box: class 0, center (0.5, 0.5), w 0.2, h 0.3
    (lbl_dir / "img_01.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    # img_02 is empty (valid negative sample)
    (lbl_dir / "img_02.txt").write_text("", encoding="utf-8")

    # data.yaml
    (root / "data.yaml").write_text("names: ['traffic_light', 'stop_sign']\n", encoding="utf-8")

    detector = DatasetFormatDetector()
    adapter = detector.detect(root)
    assert adapter.format_name == "YOLO_DETECTION"

    parsed = adapter.parse_and_validate(root)
    assert len(parsed.samples) == 2
    assert "traffic_light" in parsed.categories

    # Check sample 1 has box
    s1 = next(s for s in parsed.samples if "img_01" in s.sample_id)
    assert len(s1.bounding_boxes) == 1
    assert s1.bounding_boxes[0].coordinate_system == "NORMALIZED_0_1"
    assert s1.bounding_boxes[0].bbox == [0.5, 0.5, 0.2, 0.3]

    # Check sample 2 has 0 boxes (clean negative sample)
    s2 = next(s for s in parsed.samples if "img_02" in s.sample_id)
    assert len(s2.bounding_boxes) == 0


def test_folder_per_class_detection_and_parsing(tmp_path: Path) -> None:
    """Verifies that a folder-per-class classification directory is detected and parsed."""
    root = tmp_path / "classification_dataset"
    root.mkdir()

    # Create 2 class folders with images
    _create_dummy_image(root / "cats" / "cat1.jpg")
    _create_dummy_image(root / "cats" / "cat2.jpg")
    _create_dummy_image(root / "dogs" / "dog1.jpg")

    detector = DatasetFormatDetector()
    adapter = detector.detect(root)
    assert adapter.format_name == "FOLDER_PER_CLASS"

    parsed = adapter.parse_and_validate(root)
    assert len(parsed.samples) == 3
    assert parsed.categories == ["cats", "dogs"]
    for sample in parsed.samples:
        assert len(sample.bounding_boxes) == 0
        assert len(sample.labels) == 1


def test_unsupported_format_rejection(tmp_path: Path) -> None:
    """Verifies rejection of arbitrary or unrecognized directory layouts."""
    root = tmp_path / "random_files"
    root.mkdir()
    (root / "notes.txt").write_text("random text notes")
    (root / "script.py").write_text("print('hello')")

    detector = DatasetFormatDetector()
    with pytest.raises(UnsupportedFormatError, match="does not match any supported format"):
        detector.detect(root)


def test_coco_malformed_annotation_defect_tracking(tmp_path: Path) -> None:
    """Verifies that malformed COCO bounding boxes are recorded as validation errors rather than swallowed."""
    root = tmp_path / "bad_coco"
    root.mkdir()
    _create_dummy_image(root / "images" / "001.jpg")

    # Non-positive width/height
    anno_data = {
        "images": [{"id": 1, "file_name": "001.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "defect"}],
        "annotations": [
            {"id": 99, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, -5.0, 20.0]}
        ],
    }
    anno_file = root / "annotations" / "instances.json"
    anno_file.parent.mkdir()
    anno_file.write_text(json.dumps(anno_data), encoding="utf-8")

    detector = DatasetFormatDetector()
    adapter = detector.detect(root)
    parsed = adapter.parse_and_validate(root)

    # Defect must be documented in validation_errors
    assert any("invalid non-positive bbox" in err for err in parsed.validation_errors)


def test_yolo_multi_split_distinct_sample_ids(tmp_path: Path) -> None:
    """Verifies that YOLO datasets with identical filenames across splits receive distinct sample IDs."""
    root = tmp_path / "yolo_multi_split"
    root.mkdir()

    # Create images/train/0001.jpg, images/val/0001.jpg, images/test/0001.jpg
    _create_dummy_image(root / "images" / "train" / "0001.jpg")
    _create_dummy_image(root / "images" / "val" / "0001.jpg")
    _create_dummy_image(root / "images" / "test" / "0001.jpg")

    # Labels
    for split in ["train", "val", "test"]:
        lbl = root / "labels" / split / "0001.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (root / "data.yaml").write_text("names: ['object']\n", encoding="utf-8")

    detector = DatasetFormatDetector()
    adapter = detector.detect(root)
    assert adapter.format_name == "YOLO_DETECTION"

    parsed = adapter.parse_and_validate(root)
    assert len(parsed.samples) == 3

    sample_ids = [s.sample_id for s in parsed.samples]
    # Assert all sample IDs are completely distinct (no stem collisions)
    assert len(set(sample_ids)) == 3
    assert "yolo_images_train_0001" in sample_ids
    assert "yolo_images_val_0001" in sample_ids
    assert "yolo_images_test_0001" in sample_ids

    # Assert split information is preserved in metadata
    train_sample = next(s for s in parsed.samples if "train" in s.sample_id)
    val_sample = next(s for s in parsed.samples if "val" in s.sample_id)
    test_sample = next(s for s in parsed.samples if "test" in s.sample_id)

    assert train_sample.metadata.get("split") == "train"
    assert val_sample.metadata.get("split") == "val"
    assert test_sample.metadata.get("split") == "test"
