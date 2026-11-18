"""Domain port for assessment persistence and query operations.

Defines the boundary interface for managing assessment lifecycles, states,
and associated integrity findings without exposing database technologies.
"""

from abc import ABC, abstractmethod
from typing import Any

from backend.app.domain.entities.enums import AssessmentStatus
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import UniversalFinding


class AssessmentRepositoryPort(ABC):
    """Port interface for storing and querying assessment lifecycle states and findings."""

    @abstractmethod
    def create_assessment(
        self,
        assessment_id: str,
        name: str,
        snapshot_id: str,
        config: dict[str, Any],
    ) -> None:
        """Initializes a new assessment record."""
        ...

    @abstractmethod
    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        """Retrieves assessment metadata and status by identifier."""
        ...

    @abstractmethod
    def update_status(
        self,
        assessment_id: str,
        status: AssessmentStatus,
        status_reason: str | None = None,
    ) -> None:
        """Updates the operational state of an assessment."""
        ...

    @abstractmethod
    def record_findings(
        self,
        assessment_id: str,
        findings: list[UniversalFinding],
    ) -> None:
        """Stores integrity findings produced during an assessment."""
        ...

    @abstractmethod
    def get_findings(self, assessment_id: str) -> list[UniversalFinding]:
        """Retrieves all findings associated with a specific assessment."""
        ...

    @abstractmethod
    def publish_assessment_results(
        self,
        assessment_id: str,
        detector_runs: list[dict[str, Any]],
        findings: list[UniversalFinding],
        evidence_records: list[EvidenceRecord],
        final_status: AssessmentStatus = AssessmentStatus.COMPLETED,
        status_reason: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        lease_token: str | None = None,
        lease_generation: int | None = None,
    ) -> None:
        """Atomically commits detector runs, findings, evidence, and final assessment status under lease fencing.

        Requires all lease credentials (job_id, worker_id, lease_token, lease_generation).
        Unauthenticated publication is strictly rejected to preserve audit integrity.
        """
        ...
