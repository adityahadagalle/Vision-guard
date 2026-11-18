"""Defensive image file validation and decoding safety.

Enforces magic byte verification prior to parsing, Pillow decompression bomb limits,
pixel threshold checks, and structural verification without executing unconstrained I/O.
Note: The validator inspects headers, dimensions, and decoding integrity; it does not
re-encode image files or strip EXIF metadata.
"""

from pathlib import Path

from PIL import Image, ImageFile
from pydantic import BaseModel, ConfigDict

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    ImageDecompressionBombError,
    InvalidImageError,
)

# Prevent Pillow from loading truncated images as clean passes
ImageFile.LOAD_TRUNCATED_IMAGES = False

_MAGIC_BYTES: dict[str, bytes] = {
    "JPEG": b"\xff\xd8\xff",
    "PNG": b"\x89PNG\r\n\x1a\n",
    "BMP": b"BM",
}


class ImageValidationResult(BaseModel):
    """Structured report detailing image format, dimensions, and integrity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    is_valid: bool
    format: str | None = None
    width: int | None = None
    height: int | None = None
    pixel_count: int | None = None
    error: str | None = None


class SafeImageValidator:
    """Validates extracted image files defensively using magic bytes and Pillow bounds."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        # Enforce decompression bomb threshold globally on Pillow
        Image.MAX_IMAGE_PIXELS = self.config.max_image_pixels

    def validate_image_file(
        self, file_path: Path, relative_path: str | None = None
    ) -> ImageValidationResult:
        """Validates an image file against magic bytes, structural integrity, and pixel limits.

        Args:
            file_path: Absolute or relative filesystem path to the image file.
            relative_path: Optional display path inside dataset manifest.

        Returns:
            ImageValidationResult with image attributes or error details.

        Raises:
            InvalidImageError: If the file is not a valid or supported image.
            ImageDecompressionBombError: If pixel count exceeds safety threshold.
        """
        rel_path = relative_path or file_path.name
        if not file_path.exists() or not file_path.is_file():
            raise InvalidImageError(
                f"Image file does not exist: '{rel_path}'.",
                details={"relative_path": rel_path},
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise InvalidImageError(
                f"Image file contains 0 bytes: '{rel_path}'.",
                details={"relative_path": rel_path},
            )

        # 1. Inspect initial magic bytes
        with file_path.open("rb") as f:
            header = f.read(16)

        detected_magic = self._detect_magic_bytes(header)
        if detected_magic is None:
            raise InvalidImageError(
                f"File '{rel_path}' does not match any recognized image magic bytes (JPEG/PNG/BMP/WebP).",
                details={"relative_path": rel_path, "header_hex": header.hex()[:32]},
            )

        # 2. Open with Pillow in read mode and verify
        try:
            with Image.open(file_path) as img:
                # Pillow format check
                img_format = img.format
                width, height = img.size
                pixel_count = width * height

                # Pixel bomb check
                if pixel_count > self.config.max_image_pixels:
                    raise ImageDecompressionBombError(
                        f"Image '{rel_path}' pixel count ({pixel_count:,}) exceeds safety threshold "
                        f"of {self.config.max_image_pixels:,} pixels (decompression bomb protection).",
                        details={
                            "relative_path": rel_path,
                            "pixel_count": pixel_count,
                            "max_pixels": self.config.max_image_pixels,
                        },
                    )

                # Dimension check
                if (
                    width > self.config.max_image_dimension
                    or height > self.config.max_image_dimension
                ):
                    raise InvalidImageError(
                        f"Image '{rel_path}' dimensions ({width}x{height}) exceed maximum limit of "
                        f"{self.config.max_image_dimension}px.",
                        details={"relative_path": rel_path, "width": width, "height": height},
                    )

                # Structural integrity verification
                img.verify()

                return ImageValidationResult(
                    relative_path=rel_path,
                    is_valid=True,
                    format=img_format,
                    width=width,
                    height=height,
                    pixel_count=pixel_count,
                )

        except (ImageDecompressionBombError, InvalidImageError):
            raise
        except Image.DecompressionBombError as err:
            raise ImageDecompressionBombError(
                f"Pillow detected decompression bomb in '{rel_path}': {err}",
                details={"relative_path": rel_path, "error": str(err)},
            ) from err
        except Exception as err:
            raise InvalidImageError(
                f"Image decoding verification failed for '{rel_path}': {err}",
                details={"relative_path": rel_path, "error": str(err)},
            ) from err

    @staticmethod
    def _detect_magic_bytes(header: bytes) -> str | None:
        """Identifies image format from header magic bytes."""
        if header.startswith(_MAGIC_BYTES["JPEG"]):
            return "JPEG"
        if header.startswith(_MAGIC_BYTES["PNG"]):
            return "PNG"
        if header.startswith(_MAGIC_BYTES["BMP"]):
            return "BMP"
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
            return "WEBP"
        return None
