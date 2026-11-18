"""Domain ports for detector plugin interfaces and execution results.

Defines the minimal common detector protocol adhering to the eight-state
capability model and zero infrastructure dependency rule.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import UniversalFinding


@dataclass(frozen=True)
class DetectorRequirements:
    """Defines prerequisite assets and inputs necessary for detector execution."""

    requires_labels: bool = False
    requires_embeddings: bool = False
    requires_model_checkpoint: bool = False
    requires_reference_set: bool = False
    requires_provenance: bool = False


@dataclass(frozen=True)
class AssessmentContext:
    """Input context provided to a detector during assessment execution.

    Decouples detectors from file storage, network services, and databases.
    """

    snapshot: AssetSnapshot
    embeddings: Any | None = None
    reference_embeddings: Any | None = None
    model_checkpoint: Any | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorResult:
    """Standardized execution output returned by every VisionGuard detector.

    Supports the eight formal capability states:
    COMPLETED, NOT_APPLICABLE, UNAVAILABLE, SKIPPED, FAILED,
    TIMED_OUT, INCONCLUSIVE, and CONFLICTED.
    """

    detector_id: str
    version: str
    status: CapabilityStatus
    status_reason: str | None = None
    findings: list[UniversalFinding] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status in (CapabilityStatus.UNAVAILABLE, CapabilityStatus.FAILED) and (
            not self.status_reason or not self.status_reason.strip()
        ):
            raise ValueError(
                f"A non-empty 'status_reason' is mandatory when status is {self.status.value}."
            )


class BaseDetector(ABC):
    """Universal plugin interface for VisionGuard integrity detectors.

    Detectors are stateless algorithms that inspect input snapshots and return
    standardized DetectorResults containing findings and evidence records.
    """

    @property
    @abstractmethod
    def detector_id(self) -> str:
        """Stable dot-delimited identifier (e.g., 'data_integrity.corrupt_image')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of the detector implementation."""
        ...

    @property
    @abstractmethod
    def requirements(self) -> DetectorRequirements:
        """Prerequisites required to evaluate the detector."""
        ...

    @abstractmethod
    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Executes the detection algorithm and returns structured results."""
        ...
