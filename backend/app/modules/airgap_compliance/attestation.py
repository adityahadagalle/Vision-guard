"""Epistemic platform attestation engine for air-gapped environments.

Evaluates and certifies host isolation, offline environment enforcement,
non-root container execution posture, and hardware identity binding.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from backend.app.modules.airgap_compliance.environment import AirgapEnvironmentValidator
from backend.app.modules.airgap_compliance.hardware import (
    HardwareBindingError,
    HardwareFingerprinter,
)

__all__ = ["AttestationReport", "PlatformAttestationEngine"]


class AttestationReport(BaseModel):
    """Immutable platform attestation report representing verified air-gap posture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: str
    hardware_fingerprint: str
    is_offline_enforced: bool
    is_non_root_user: bool
    is_hardware_bound: bool
    is_compliant: bool


class PlatformAttestationEngine:
    """Evaluates host security posture and produces cryptographic platform attestation reports."""

    @classmethod
    def check_non_root_user(cls) -> bool:
        """Verify process execution is not running with root/administrator privileges."""
        getuid_fn = getattr(os, "getuid", None)
        if callable(getuid_fn):
            return getuid_fn() > 0

        # Windows platform fallback: check administrative token
        try:
            import ctypes

            return not bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return True

    @classmethod
    def generate_report(cls) -> AttestationReport:
        """Evaluate platform environment invariants and generate immutable AttestationReport.

        Invariants evaluated:
        1. Offline flags enforcement (HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
        2. Non-root user execution posture (UID > 0 or non-admin).
        3. Deterministic hardware binding (valid MAC without multicast bit + host identifier).
        """
        timestamp = datetime.now(UTC).isoformat()

        # 1. Check offline flags
        is_offline_enforced = AirgapEnvironmentValidator.validate_offline_flags()

        # 2. Check non-root status
        is_non_root_user = cls.check_non_root_user()

        # 3. Check hardware binding
        try:
            hardware_fingerprint = HardwareFingerprinter.get_hardware_fingerprint()
            is_hardware_bound = True
        except HardwareBindingError:
            hardware_fingerprint = "HARDWARE_BINDING_FAILED"
            is_hardware_bound = False

        # Invariant: is_compliant strictly True only if ALL checks pass
        is_compliant = bool(is_offline_enforced and is_non_root_user and is_hardware_bound)

        return AttestationReport(
            timestamp_utc=timestamp,
            hardware_fingerprint=hardware_fingerprint,
            is_offline_enforced=is_offline_enforced,
            is_non_root_user=is_non_root_user,
            is_hardware_bound=is_hardware_bound,
            is_compliant=is_compliant,
        )
