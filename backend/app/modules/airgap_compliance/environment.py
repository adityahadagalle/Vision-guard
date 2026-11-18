"""Air-gap compliance environment validation module.

Validates that runtime execution occurs strictly within an isolated,
air-gapped environment with offline library flags strictly enforced.
"""

from __future__ import annotations

import os
from typing import ClassVar


class AirgapEnvironmentValidator:
    """Validates host and container runtime environment variables for strict air-gap compliance."""

    MANDATORY_OFFLINE_FLAGS: ClassVar[list[str]] = [
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ]

    @classmethod
    def validate_offline_flags(cls) -> bool:
        """Validate that all mandatory offline flags are strictly set to '1'.

        Returns True only if both HF_DATASETS_OFFLINE and TRANSFORMERS_OFFLINE are
        present in os.environ and strictly equal to '1'.
        """
        return all(os.environ.get(flag) == "1" for flag in cls.MANDATORY_OFFLINE_FLAGS)

    @classmethod
    def get_offline_status(cls) -> dict[str, bool]:
        """Return detailed per-flag validation status for audit reporting.

        Returns:
            Dictionary mapping flag name to boolean compliance status.
        """
        return {flag: os.environ.get(flag) == "1" for flag in cls.MANDATORY_OFFLINE_FLAGS}
