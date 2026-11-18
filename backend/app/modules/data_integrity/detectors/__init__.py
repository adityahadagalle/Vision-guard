"""Tier-1 data integrity detectors.

Provides deterministic structural and baseline integrity checks:
1. CorruptImageDetector: Verifies image decode safety, magic bytes, dimensions, and truncation.
2. ExactDuplicateDetector: Flags identical SHA-256 images within and across dataset splits.
3. AnnotationGeometryValidator: Validates COCO and YOLO bounding box geometry and bounds.
"""

from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.data_integrity.detectors.annotation_geometry import (
    AnnotationGeometryValidator,
)
from backend.app.modules.data_integrity.detectors.corrupt_image import (
    CorruptImageDetector,
)
from backend.app.modules.data_integrity.detectors.exact_duplicate import (
    ExactDuplicateDetector,
)

__all__ = [
    "CorruptImageDetector",
    "ExactDuplicateDetector",
    "AnnotationGeometryValidator",
    "register_tier1_detectors",
]


def register_tier1_detectors(registry: DetectorRegistry) -> None:
    """Registers all Tier-1 integrity detectors in the provided registry."""
    registry.register(CorruptImageDetector())
    registry.register(ExactDuplicateDetector())
    registry.register(AnnotationGeometryValidator())
