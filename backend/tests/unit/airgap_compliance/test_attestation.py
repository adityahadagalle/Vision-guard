"""Unit tests for Platform Attestation Engine.

Verifies air-gap security posture reporting, non-root user checks, offline
environment flag enforcement, hardware identity binding, and fail-closed compliance gating.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.app.modules.airgap_compliance.attestation import (
    AttestationReport,
    PlatformAttestationEngine,
)
from backend.app.modules.airgap_compliance.environment import AirgapEnvironmentValidator
from backend.app.modules.airgap_compliance.hardware import (
    HardwareBindingError,
    HardwareFingerprinter,
)


class TestPlatformAttestationEngine:
    """Test suite for PlatformAttestationEngine security assertions."""

    def test_fully_compliant_environment(self) -> None:
        """Returns is_compliant=True when all offline, non-root, and hardware invariants hold."""
        with (
            patch.object(AirgapEnvironmentValidator, "validate_offline_flags", return_value=True),
            patch.object(PlatformAttestationEngine, "check_non_root_user", return_value=True),
            patch.object(
                HardwareFingerprinter,
                "get_hardware_fingerprint",
                return_value="a" * 64,
            ),
        ):
            report = PlatformAttestationEngine.generate_report()

            assert isinstance(report, AttestationReport)
            assert report.is_offline_enforced is True
            assert report.is_non_root_user is True
            assert report.is_hardware_bound is True
            assert report.is_compliant is True
            assert report.hardware_fingerprint == "a" * 64
            assert report.timestamp_utc is not None

    def test_missing_offline_flags_results_in_non_compliant(self) -> None:
        """Sets is_compliant=False when offline enforcement flags fail."""
        with (
            patch.object(AirgapEnvironmentValidator, "validate_offline_flags", return_value=False),
            patch.object(PlatformAttestationEngine, "check_non_root_user", return_value=True),
            patch.object(
                HardwareFingerprinter,
                "get_hardware_fingerprint",
                return_value="b" * 64,
            ),
        ):
            report = PlatformAttestationEngine.generate_report()

            assert report.is_offline_enforced is False
            assert report.is_compliant is False

    def test_root_user_execution_results_in_non_compliant(self) -> None:
        """Sets is_compliant=False when executing as root (UID 0)."""
        with (
            patch.object(AirgapEnvironmentValidator, "validate_offline_flags", return_value=True),
            patch("os.getuid", return_value=0, create=True),
            patch.object(
                HardwareFingerprinter,
                "get_hardware_fingerprint",
                return_value="c" * 64,
            ),
        ):
            report = PlatformAttestationEngine.generate_report()

            assert report.is_non_root_user is False
            assert report.is_compliant is False

    def test_hardware_binding_error_results_in_non_compliant(self) -> None:
        """Sets is_compliant=False and is_hardware_bound=False when hardware binding fails."""
        with (
            patch.object(AirgapEnvironmentValidator, "validate_offline_flags", return_value=True),
            patch.object(PlatformAttestationEngine, "check_non_root_user", return_value=True),
            patch.object(
                HardwareFingerprinter,
                "get_hardware_fingerprint",
                side_effect=HardwareBindingError("Randomized dummy MAC"),
            ),
        ):
            report = PlatformAttestationEngine.generate_report()

            assert report.is_hardware_bound is False
            assert report.hardware_fingerprint == "HARDWARE_BINDING_FAILED"
            assert report.is_compliant is False

    def test_multiple_failures_fail_closed(self) -> None:
        """Fails closed when all three security invariants fail simultaneously."""
        with (
            patch.object(AirgapEnvironmentValidator, "validate_offline_flags", return_value=False),
            patch.object(PlatformAttestationEngine, "check_non_root_user", return_value=False),
            patch.object(
                HardwareFingerprinter,
                "get_hardware_fingerprint",
                side_effect=HardwareBindingError("MAC error"),
            ),
        ):
            report = PlatformAttestationEngine.generate_report()

            assert report.is_offline_enforced is False
            assert report.is_non_root_user is False
            assert report.is_hardware_bound is False
            assert report.is_compliant is False

    def test_attestation_report_immutability(self) -> None:
        """Asserts AttestationReport is frozen and forbids extra attributes."""
        report = AttestationReport(
            timestamp_utc="2026-11-17T00:00:00Z",
            hardware_fingerprint="abc123",
            is_offline_enforced=True,
            is_non_root_user=True,
            is_hardware_bound=True,
            is_compliant=True,
        )

        with pytest.raises(ValidationError):
            # Frozen model rejects field mutation
            report.is_compliant = False  # type: ignore[misc]

        with pytest.raises(ValidationError):
            # extra="forbid" rejects arbitrary properties
            AttestationReport(
                timestamp_utc="2026-11-17T00:00:00Z",
                hardware_fingerprint="abc123",
                is_offline_enforced=True,
                is_non_root_user=True,
                is_hardware_bound=True,
                is_compliant=True,
                extra_field="disallowed",  # type: ignore[call-arg]
            )
