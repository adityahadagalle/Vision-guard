"""Domain port interfaces for VisionGuard architecture boundaries."""

from backend.app.domain.ports.artifact_storage import ArtifactStoragePort
from backend.app.domain.ports.assessment_repository import AssessmentRepositoryPort
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)
from backend.app.domain.ports.job_queue import JobQueuePort

__all__ = [
    "ArtifactStoragePort",
    "AssessmentContext",
    "AssessmentRepositoryPort",
    "BaseDetector",
    "DetectorRequirements",
    "DetectorResult",
    "JobQueuePort",
]
