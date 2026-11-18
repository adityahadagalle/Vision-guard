"""Unit tests for SafeImageValidator."""

from pathlib import Path

import pytest
from PIL import Image

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    ImageDecompressionBombError,
    InvalidImageError,
)
from backend.app.modules.data_integrity.ingestion.image_validator import SafeImageValidator


def _create_real_image(
    tmp_path: Path, filename: str, size: tuple[int, int] = (64, 64), img_format: str = "JPEG"
) -> Path:
    img_path = tmp_path / filename
    img = Image.new("RGB", size, color=(255, 0, 0))
    img.save(img_path, format=img_format)
    return img_path


def test_valid_jpeg_and_png_validation(tmp_path: Path) -> None:
    """Verifies that authentic JPEG and PNG images pass validation."""
    validator = SafeImageValidator()

    jpg = _create_real_image(tmp_path, "sample.jpg", size=(100, 80), img_format="JPEG")
    res_jpg = validator.validate_image_file(jpg)
    assert res_jpg.is_valid is True
    assert res_jpg.format == "JPEG"
    assert res_jpg.width == 100
    assert res_jpg.height == 80

    png = _create_real_image(tmp_path, "sample.png", size=(50, 50), img_format="PNG")
    res_png = validator.validate_image_file(png)
    assert res_png.is_valid is True
    assert res_png.format == "PNG"


def test_magic_bytes_mismatch_rejection(tmp_path: Path) -> None:
    """Verifies that non-image text files disguised with image extensions are rejected."""
    validator = SafeImageValidator()
    disguised = tmp_path / "fake.jpg"
    disguised.write_text("Hello, I am actually a text script masquerading as JPEG!")

    with pytest.raises(InvalidImageError, match="does not match any recognized image magic bytes"):
        validator.validate_image_file(disguised)


def test_empty_image_rejection(tmp_path: Path) -> None:
    """Verifies rejection of 0-byte image files."""
    validator = SafeImageValidator()
    empty = tmp_path / "empty.png"
    empty.touch()

    with pytest.raises(InvalidImageError, match="contains 0 bytes"):
        validator.validate_image_file(empty)


def test_image_decompression_bomb_rejection(tmp_path: Path) -> None:
    """Verifies rejection when image pixel count exceeds max_image_pixels."""
    # Configure tight pixel threshold (e.g. 500 pixels)
    config = IngestionConfig(max_image_pixels=500)
    validator = SafeImageValidator(config=config)

    # Create image with 100x100 = 10,000 pixels (> 500)
    large_img = _create_real_image(tmp_path, "bomb.png", size=(100, 100), img_format="PNG")

    with pytest.raises(ImageDecompressionBombError, match="decompression bomb"):
        validator.validate_image_file(large_img)


def test_excessive_dimension_rejection(tmp_path: Path) -> None:
    """Verifies rejection when width or height exceeds max_image_dimension."""
    config = IngestionConfig(max_image_dimension=50)
    validator = SafeImageValidator(config=config)

    # Create 60x10 image (width 60 > 50)
    wide_img = _create_real_image(tmp_path, "wide.jpg", size=(60, 10), img_format="JPEG")

    with pytest.raises(InvalidImageError, match="dimensions"):
        validator.validate_image_file(wide_img)
