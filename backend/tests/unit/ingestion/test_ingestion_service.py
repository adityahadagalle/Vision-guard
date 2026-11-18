"""Integration and end-to-end unit tests for IngestionService."""

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    PathTraversalError,
)
from backend.app.modules.data_integrity.ingestion.service import IngestionService


def _create_real_jpeg_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(0, 128, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_coco_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "valid_coco.zip"
    img_bytes = _create_real_jpeg_bytes((64, 64))

    anno_data = {
        "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "pedestrian"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5.0, 5.0, 20.0, 30.0],
                "area": 600.0,
            }
        ],
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("images/0001.jpg", img_bytes)
        zf.writestr("annotations/instances.json", json.dumps(anno_data))

    return zip_path


def test_end_to_end_successful_coco_ingestion(tmp_path: Path) -> None:
    """Verifies complete ingestion pipeline from raw archive to sealed immutable snapshot."""
    config = IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        snapshots_dir=tmp_path / "snapshots",
    )
    service = IngestionService(config)
    coco_zip = _create_coco_zip(tmp_path)

    result = service.ingest_archive(coco_zip, asset_id="asset_coco_test")

    assert result.status == IngestionStatus.ACCEPTED
    assert result.snapshot.snapshot_id.startswith("snap_")
    assert result.snapshot.file_count >= 2
    assert result.normalized_dataset.original_format == "COCO_DETECTION"
    assert "pedestrian" in result.normalized_dataset.categories
    assert result.snapshot_path.exists()
    assert (result.snapshot_path / "manifest.json").exists()

    # Ephemeral staging directory must be cleaned up
    assert not any(config.staging_dir.glob("stage_*"))


def test_end_to_end_rejection_of_zip_slip_attack(tmp_path: Path) -> None:
    """Verifies that an archive with path traversal (ZIP Slip) is rejected and cleaned up."""
    config = IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        snapshots_dir=tmp_path / "snapshots",
    )
    service = IngestionService(config)

    malicious_zip = tmp_path / "zip_slip.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../evil_script.sh", b"rm -rf /")

    with pytest.raises(PathTraversalError, match="Path traversal component"):
        service.ingest_archive(malicious_zip)

    # Ephemeral staging directory must not linger
    assert not any(config.staging_dir.glob("stage_*"))
