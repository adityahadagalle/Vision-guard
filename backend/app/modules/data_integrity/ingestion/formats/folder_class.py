"""Folder-per-Class image classification dataset format adapter.

Recognizes standard classification hierarchies where top-level subdirectories
represent distinct class labels (e.g. root/cats/*.jpg, root/dogs/*.jpg).
"""

from pathlib import Path

from backend.app.modules.data_integrity.ingestion.exceptions import MalformedAnnotationError
from backend.app.modules.data_integrity.ingestion.formats.base import (
    BaseFormatAdapter,
    NormalizedSample,
    ParsedDataset,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderPerClassDatasetAdapter(BaseFormatAdapter):
    """Adapter for directory-per-class image classification datasets."""

    @property
    def format_name(self) -> str:
        return "FOLDER_PER_CLASS"

    def can_handle(self, root_dir: Path) -> bool:
        """Determines if root_dir possesses a folder-per-class taxonomy."""
        # Exclude directories that look like COCO or YOLO
        if (root_dir / "annotations").exists() or (root_dir / "labels").exists():
            return False

        class_dirs = self._find_class_directories(root_dir)
        if len(class_dirs) < 2:
            return False

        # Verify that class directories contain actual image files
        return all(self._has_images(cdir) for cdir in class_dirs)

    def parse_and_validate(self, root_dir: Path) -> ParsedDataset:
        """Parses folder-per-class structure and creates classification samples."""
        class_dirs = self._find_class_directories(root_dir)
        if len(class_dirs) < 2:
            raise MalformedAnnotationError(
                f"Folder-per-Class requires at least 2 class subdirectories, found {len(class_dirs)} in '{root_dir}'.",
                details={"root_dir": str(root_dir), "found_classes": len(class_dirs)},
            )

        samples: list[NormalizedSample] = []
        categories: list[str] = sorted([cdir.name for cdir in class_dirs])
        warnings: list[str] = []
        errors: list[str] = []

        for cdir in class_dirs:
            class_name = cdir.name
            img_count = 0

            for ext in _IMAGE_EXTS:
                for img_path in sorted(cdir.glob(f"*{ext}")):
                    img_count += 1
                    rel_path = str(img_path.relative_to(root_dir)).replace("\\", "/")
                    samples.append(
                        NormalizedSample(
                            sample_id=f"cls_{class_name}_{img_path.stem}",
                            image_relative_path=rel_path,
                            labels=[class_name],
                            bounding_boxes=[],  # Pure classification, zero bboxes
                            metadata={"class_name": class_name},
                        )
                    )

            if img_count == 0:
                warnings.append(
                    f"Class directory '{class_name}' contains zero recognized image files."
                )

        if not samples:
            raise MalformedAnnotationError(
                f"Zero valid images found across class folders in '{root_dir}'.",
                details={"root_dir": str(root_dir)},
            )

        return ParsedDataset(
            format_name=self.format_name,
            samples=samples,
            categories=categories,
            metadata={
                "total_classes": len(categories),
                "total_images": len(samples),
                "classes": categories,
            },
            validation_warnings=warnings,
            validation_errors=errors,
        )

    @classmethod
    def _find_class_directories(cls, root_dir: Path) -> list[Path]:
        """Locates candidate class directories directly or under train/."""
        # 1. Direct subdirectories
        direct_subdirs = [
            p for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]

        # If subdirectories are split names (train, val, test), look inside train
        if any(p.name.lower() in ("train", "training") for p in direct_subdirs):
            for split_name in ("train", "training"):
                split_dir = root_dir / split_name
                if split_dir.exists() and split_dir.is_dir():
                    return [
                        p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
                    ]

        return direct_subdirs

    @classmethod
    def _has_images(cls, directory: Path) -> bool:
        """Returns True if directory contains at least one recognized image file."""
        for ext in _IMAGE_EXTS:
            if any(directory.glob(f"*{ext}")):
                return True
            if any(directory.glob(f"*{ext.upper()}")):
                return True
        return False
