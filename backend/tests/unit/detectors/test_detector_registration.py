"""Unit tests for Tier-1 detector registration and protocol compliance."""

from backend.app.domain.ports.detector import BaseDetector
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.data_integrity.detectors import (
    AnnotationGeometryValidator,
    CorruptImageDetector,
    ExactDuplicateDetector,
    register_tier1_detectors,
)


def test_register_tier1_detectors() -> None:
    """Verifies that register_tier1_detectors successfully catalogs all three Tier-1 detectors."""
    registry = DetectorRegistry()
    register_tier1_detectors(registry)

    assert len(registry) == 3
    assert registry.has("data_integrity.corrupt_image")
    assert registry.has("data_integrity.exact_duplicate")
    assert registry.has("data_integrity.annotation_geometry")


def test_tier1_detector_protocol_compliance() -> None:
    """Verifies that all Tier-1 detectors derive from BaseDetector and declare valid requirements."""
    detectors = [
        CorruptImageDetector(),
        ExactDuplicateDetector(),
        AnnotationGeometryValidator(),
    ]

    for det in detectors:
        assert isinstance(det, BaseDetector)
        assert isinstance(det.detector_id, str) and det.detector_id.startswith("data_integrity.")
        assert det.version == "1.0.0"
        assert det.requirements is not None


def test_detector_metadata_extraction() -> None:
    """Verifies that metadata can be extracted from registered Tier-1 detectors."""
    registry = DetectorRegistry()
    register_tier1_detectors(registry)

    geom_meta = registry.get_metadata("data_integrity.annotation_geometry")
    assert geom_meta["detector_id"] == "data_integrity.annotation_geometry"
    assert geom_meta["requirements"]["requires_labels"] is True

    corrupt_meta = registry.get_metadata("data_integrity.corrupt_image")
    assert corrupt_meta["requirements"]["requires_labels"] is False
