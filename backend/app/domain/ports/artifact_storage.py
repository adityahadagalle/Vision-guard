"""Domain port for content-addressed artifact storage.

Defines the boundary interface for storing and retrieving immutable binary
payloads addressed by their cryptographic SHA-256 digests.
"""

from abc import ABC, abstractmethod


class ArtifactStoragePort(ABC):
    """Port interface for content-addressed binary artifact storage."""

    @abstractmethod
    def store(self, data: bytes) -> str:
        """Stores binary data content-addressed by its SHA-256 digest.

        Returns:
            The 64-character lowercase hexadecimal SHA-256 digest of the stored data.
        """
        ...

    @abstractmethod
    def retrieve(self, digest: str) -> bytes | None:
        """Retrieves binary data associated with the given SHA-256 digest.

        Returns:
            The raw bytes if present, or None if the digest does not exist.
        """
        ...

    @abstractmethod
    def exists(self, digest: str) -> bool:
        """Verifies whether an artifact with the given SHA-256 digest exists."""
        ...

    @abstractmethod
    def delete(self, digest: str) -> bool:
        """Removes the artifact associated with the given digest if permitted.

        Returns:
            True if the artifact was deleted, False if it was not found or deletion failed.
        """
        ...
