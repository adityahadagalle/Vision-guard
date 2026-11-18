"""Domain port for asynchronous job submission and task leasing.

Defines the boundary interface for background task orchestration without
depending on specific message brokers or queue implementations.
"""

from abc import ABC, abstractmethod
from typing import Any


class JobQueuePort(ABC):
    """Port interface for submitting, leasing, and resolving asynchronous tasks."""

    @abstractmethod
    def submit_job(self, task_type: str, payload: dict[str, Any]) -> str:
        """Enqueues a new background task.

        Returns:
            The unique identifier of the created job.
        """
        ...

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieves details, parameters, and status of a specific job."""
        ...

    @abstractmethod
    def claim_next_job(
        self,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Atomically leases the next available task for processing.

        Returns:
            Job dictionary if claimed, or None if the queue is empty.
        """
        ...

    @abstractmethod
    def complete_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Marks a claimed job as successfully completed."""
        ...

    @abstractmethod
    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str | None = None,
        error_message: str = "",
    ) -> None:
        """Marks a claimed job as failed with structured error information."""
        ...

    @abstractmethod
    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        additional_seconds: int = 300,
    ) -> Any:
        """Extends the lease duration for an active job holding a valid lease token."""
        ...
