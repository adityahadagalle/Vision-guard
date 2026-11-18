"""Structural dataset format detector.

Inspects extracted dataset directory layouts against registered format adapters,
enforcing unambiguous format identification and rejecting unrecognized layouts.
"""

from pathlib import Path

from backend.app.modules.data_integrity.ingestion.exceptions import UnsupportedFormatError
from backend.app.modules.data_integrity.ingestion.formats.base import BaseFormatAdapter
from backend.app.modules.data_integrity.ingestion.formats.coco import COCODatasetAdapter
from backend.app.modules.data_integrity.ingestion.formats.folder_class import (
    FolderPerClassDatasetAdapter,
)
from backend.app.modules.data_integrity.ingestion.formats.yolo import YOLODatasetAdapter


class DatasetFormatDetector:
    """Dispatches extracted dataset directories to matching format adapters."""

    def __init__(self, adapters: list[BaseFormatAdapter] | None = None) -> None:
        self.adapters: list[BaseFormatAdapter] = adapters or [
            COCODatasetAdapter(),
            YOLODatasetAdapter(),
            FolderPerClassDatasetAdapter(),
        ]

    def detect(self, extracted_root: Path) -> BaseFormatAdapter:
        """Identifies the unique dataset format adapter capable of parsing extracted_root.

        Args:
            extracted_root: Path to the root of the safely extracted dataset.

        Returns:
            The unique matching BaseFormatAdapter instance.

        Raises:
            UnsupportedFormatError: If zero adapters match or multiple conflicting adapters match.
        """
        if not extracted_root.exists() or not extracted_root.is_dir():
            raise UnsupportedFormatError(
                f"Extracted directory does not exist or is not a directory: '{extracted_root}'.",
                details={"path": str(extracted_root)},
            )

        matching_adapters: list[BaseFormatAdapter] = []
        for adapter in self.adapters:
            if adapter.can_handle(extracted_root):
                matching_adapters.append(adapter)

        if not matching_adapters:
            raise UnsupportedFormatError(
                f"Dataset structure in '{extracted_root.name}' does not match any supported format "
                f"(supported: {', '.join([a.format_name for a in self.adapters])}).",
                details={"path": str(extracted_root)},
            )

        if len(matching_adapters) > 1:
            matched_names = [a.format_name for a in matching_adapters]
            raise UnsupportedFormatError(
                f"Ambiguous dataset structure: directory matched multiple format adapters: {matched_names}.",
                details={"matched_formats": matched_names, "path": str(extracted_root)},
            )

        return matching_adapters[0]
