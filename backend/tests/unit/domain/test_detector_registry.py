"""Unit tests for DetectorRegistry."""

import pytest

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)
from backend.app.domain.registry.detector_registry import (
    DetectorNotFoundError,
    DetectorRegistry,
    DuplicateDetectorError,
)


class MockDetector(BaseDetector):
    """Mock detector used for unit testing registry operations."""

    def __init__(
        self,
        detector_id: str = "data_integrity.mock_detector",
        version: str = "1.0.0",
        requires_labels: bool = False,
    ) -> None:
        self._detector_id = detector_id
        self._version = version
        self._requirements = DetectorRequirements(requires_labels=requires_labels)
        self.execution_count = 0

    @property
    def detector_id(self) -> str:
        return self._detector_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def requirements(self) -> DetectorRequirements:
        return self._requirements

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        self.execution_count += 1
        return DetectorResult(
            detector_id=self._detector_id,
            version=self._version,
            status=CapabilityStatus.COMPLETED,
        )


def test_registry_successful_registration_and_lookup() -> None:
    """Verifies registering a detector and retrieving it by stable ID."""
    registry = DetectorRegistry()
    detector = MockDetector(detector_id="data_integrity.corrupt_image")

    registry.register(detector)

    assert registry.has("data_integrity.corrupt_image")
    assert "data_integrity.corrupt_image" in registry
    assert len(registry) == 1

    retrieved = registry.get("data_integrity.corrupt_image")
    assert retrieved is detector
    assert retrieved.detector_id == "data_integrity.corrupt_image"
    assert retrieved.version == "1.0.0"


def test_registry_duplicate_registration_rejection() -> None:
    """Verifies that registering a duplicate detector ID raises DuplicateDetectorError."""
    registry = DetectorRegistry()
    d1 = MockDetector(detector_id="data_integrity.duplicate_check")
    d2 = MockDetector(detector_id="data_integrity.duplicate_check")

    registry.register(d1)

    with pytest.raises(DuplicateDetectorError, match="is already registered"):
        registry.register(d2)


def test_registry_unknown_detector_lookup() -> None:
    """Verifies that querying an unknown detector ID raises DetectorNotFoundError."""
    registry = DetectorRegistry()
    assert not registry.has("unknown.detector")

    with pytest.raises(DetectorNotFoundError, match="is not registered"):
        registry.get("unknown.detector")


def test_registry_list_ids_and_all() -> None:
    """Verifies listing all registered detector IDs and instances in sorted order."""
    registry = DetectorRegistry()
    d1 = MockDetector(detector_id="model_integrity.trojan_scanner")
    d2 = MockDetector(detector_id="data_integrity.exact_duplicate")
    d3 = MockDetector(detector_id="distribution_shift.wasserstein")

    registry.register(d1)
    registry.register(d2)
    registry.register(d3)

    expected_ids = [
        "data_integrity.exact_duplicate",
        "distribution_shift.wasserstein",
        "model_integrity.trojan_scanner",
    ]
    assert registry.list_ids() == expected_ids

    instances = registry.list_all()
    assert [inst.detector_id for inst in instances] == expected_ids


def test_registry_get_metadata() -> None:
    """Verifies metadata extraction for registered detectors."""
    registry = DetectorRegistry()
    detector = MockDetector(
        detector_id="data_integrity.label_noise",
        version="2.1.0",
        requires_labels=True,
    )
    registry.register(detector)

    meta = registry.get_metadata("data_integrity.label_noise")
    assert meta["detector_id"] == "data_integrity.label_noise"
    assert meta["version"] == "2.1.0"
    assert meta["requirements"]["requires_labels"] is True
    assert meta["requirements"]["requires_embeddings"] is False


def test_registry_does_not_execute_detectors() -> None:
    """Verifies that registering and inspecting detectors does not invoke their evaluate() method."""
    registry = DetectorRegistry()
    detector = MockDetector()

    registry.register(detector)
    _ = registry.get(detector.detector_id)
    _ = registry.get_metadata(detector.detector_id)

    assert detector.execution_count == 0


def test_registry_rejection_of_invalid_types() -> None:
    """Verifies rejection of objects not implementing BaseDetector."""
    registry = DetectorRegistry()
    with pytest.raises(TypeError, match="Expected an instance of BaseDetector"):
        registry.register("not_a_detector")  # type: ignore[arg-type]


def test_registry_clear() -> None:
    """Verifies that clear() removes all registered detectors."""
    registry = DetectorRegistry()
    registry.register(MockDetector())
    assert len(registry) == 1

    registry.clear()
    assert len(registry) == 0
    assert not registry.has("data_integrity.mock_detector")
