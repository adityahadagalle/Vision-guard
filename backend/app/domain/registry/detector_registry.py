"""In-memory detector registry for VisionGuard.

Enforces explicit, in-memory runtime detector registration, prevents duplicate
registrations, and provides metadata querying without dynamic code execution.
"""

from typing import Any

from backend.app.domain.ports.detector import BaseDetector, DetectorRequirements


class DetectorRegistryError(Exception):
    """Base exception for detector registry errors."""

    pass


class DuplicateDetectorError(DetectorRegistryError):
    """Raised when attempting to register a detector with an already registered ID."""

    pass


class DetectorNotFoundError(DetectorRegistryError):
    """Raised when attempting to retrieve an unregistered detector ID."""

    pass


class DetectorRegistry:
    """Central catalog for validated, statically registered detector implementations.

    Invariants:
    1. Detectors are explicitly registered; unsafe runtime dynamic imports are forbidden.
    2. Duplicate detector IDs are strictly rejected.
    3. Registration never executes detector algorithms.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector) -> None:
        """Registers a detector instance.

        Args:
            detector: Concrete implementation conforming to the BaseDetector interface.

        Raises:
            TypeError: If the object does not derive from BaseDetector.
            DuplicateDetectorError: If the detector_id has already been registered.
            ValueError: If detector_id or version are empty.
        """
        if not isinstance(detector, BaseDetector):
            raise TypeError(f"Expected an instance of BaseDetector, got {type(detector).__name__}.")

        det_id = detector.detector_id
        if not isinstance(det_id, str) or not det_id.strip():
            raise ValueError("Detector must provide a non-empty string detector_id.")

        if not isinstance(detector.version, str) or not detector.version.strip():
            raise ValueError(f"Detector '{det_id}' must provide a non-empty string version.")

        if not isinstance(detector.requirements, DetectorRequirements):
            raise TypeError(
                f"Detector '{det_id}' must provide requirements as a DetectorRequirements instance."
            )

        if det_id in self._detectors:
            raise DuplicateDetectorError(f"Detector with ID '{det_id}' is already registered.")

        self._detectors[det_id] = detector

    def get(self, detector_id: str) -> BaseDetector:
        """Retrieves a detector by its stable identifier.

        Args:
            detector_id: The dot-delimited detector identifier.

        Returns:
            The registered BaseDetector instance.

        Raises:
            DetectorNotFoundError: If no detector is registered with the given ID.
        """
        if detector_id not in self._detectors:
            raise DetectorNotFoundError(f"Detector with ID '{detector_id}' is not registered.")
        return self._detectors[detector_id]

    def has(self, detector_id: str) -> bool:
        """Returns True if a detector with the specified ID is registered."""
        return detector_id in self._detectors

    def list_ids(self) -> list[str]:
        """Returns a lexicographically sorted list of all registered detector IDs."""
        return sorted(self._detectors.keys())

    def list_all(self) -> list[BaseDetector]:
        """Returns a list of all registered detector instances."""
        return [self._detectors[k] for k in sorted(self._detectors.keys())]

    def get_metadata(self, detector_id: str) -> dict[str, Any]:
        """Extracts declared metadata and execution prerequisites for a detector."""
        detector = self.get(detector_id)
        reqs = detector.requirements
        return {
            "detector_id": detector.detector_id,
            "version": detector.version,
            "requirements": {
                "requires_labels": reqs.requires_labels,
                "requires_embeddings": reqs.requires_embeddings,
                "requires_model_checkpoint": reqs.requires_model_checkpoint,
                "requires_reference_set": reqs.requires_reference_set,
                "requires_provenance": reqs.requires_provenance,
            },
        }

    def clear(self) -> None:
        """Clears all registered detectors (used for test isolation)."""
        self._detectors.clear()

    def __len__(self) -> int:
        return len(self._detectors)

    def __contains__(self, detector_id: str) -> bool:
        return self.has(detector_id)
