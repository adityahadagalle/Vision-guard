"""Detector registry module for VisionGuard."""

from backend.app.domain.registry.detector_registry import (
    DetectorNotFoundError,
    DetectorRegistry,
    DetectorRegistryError,
    DuplicateDetectorError,
)

__all__ = [
    "DetectorNotFoundError",
    "DetectorRegistry",
    "DetectorRegistryError",
    "DuplicateDetectorError",
]
