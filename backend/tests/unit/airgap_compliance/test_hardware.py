"""Unit tests for Hardware Fingerprinter.

Verifies deterministic SHA-256 hardware fingerprint generation, MAC address
multicast bit gating per RFC 4122, and failure when running on randomized dummy MACs.
"""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from backend.app.modules.airgap_compliance.hardware import (
    HardwareBindingError,
    HardwareFingerprinter,
)


class TestHardwareFingerprinter:
    """Test suite for HardwareFingerprinter device binding and verification."""

    def test_get_mac_address_genuine_success(self) -> None:
        """Returns integer MAC when uuid.getnode() provides genuine hardware MAC."""
        # 0x001122334455 has multicast bit (bit 40) = 0
        with patch("uuid.getnode", return_value=0x001122334455):
            mac = HardwareFingerprinter.get_mac_address()
            assert mac == 0x001122334455

    def test_get_mac_address_multicast_dummy_raises_hardware_binding_error(self) -> None:
        """Raises HardwareBindingError when uuid.getnode() returns randomized dummy MAC."""
        # 0x011122334455 has bit 40 = 1 (multicast bit set per RFC 4122)
        with (
            patch("uuid.getnode", return_value=0x011122334455),
            pytest.raises(HardwareBindingError, match="multicast bit set"),
        ):
            HardwareFingerprinter.get_mac_address()

    @pytest.mark.parametrize("invalid_node", [0, -1, 1 << 48, 1 << 50, "not-an-int"])
    def test_get_mac_address_invalid_values_raise_hardware_binding_error(
        self, invalid_node: object
    ) -> None:
        """Raises HardwareBindingError for out-of-range, non-int, or non-positive values."""
        with (
            patch("uuid.getnode", return_value=invalid_node),
            pytest.raises(HardwareBindingError),
        ):
            HardwareFingerprinter.get_mac_address()

    def test_get_stable_host_id_linux_mock(self) -> None:
        """Reads linux machine-id when present."""
        with (
            patch("sys.platform", "linux"),
            patch("os.path.exists", side_effect=lambda p: p == "/etc/machine-id"),
            patch("builtins.open", mock_open(read_data="mock-machine-id-12345\n")),
        ):
            host_id = HardwareFingerprinter.get_stable_host_id()
            assert host_id == "mock-machine-id-12345"

    def test_get_stable_host_id_cpu_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to CPU_SERIAL or PROCESSOR_IDENTIFIER when OS inspection fails."""
        monkeypatch.setenv("CPU_SERIAL", "MOCKED-CPU-SERIAL-XYZ")
        with (
            patch("sys.platform", "linux"),
            patch("os.path.exists", return_value=False),
        ):
            host_id = HardwareFingerprinter.get_stable_host_id()
            assert host_id == "MOCKED-CPU-SERIAL-XYZ"

    def test_get_hardware_fingerprint_deterministic(self) -> None:
        """Produces identical SHA-256 fingerprint for identical hardware attributes."""
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455),
            patch.object(
                HardwareFingerprinter, "get_stable_host_id", return_value="STATIC-HOST-ID"
            ),
        ):
            fp1 = HardwareFingerprinter.get_hardware_fingerprint()
            fp2 = HardwareFingerprinter.get_hardware_fingerprint()
            assert fp1 == fp2
            assert len(fp1) == 64
            # Verify instance call works identically
            assert HardwareFingerprinter().get_hardware_fingerprint() == fp1

    def test_get_hardware_fingerprint_changes_on_mac_mutation(self) -> None:
        """Produces different SHA-256 fingerprint when MAC address changes."""
        with patch.object(
            HardwareFingerprinter, "get_stable_host_id", return_value="STATIC-HOST-ID"
        ):
            with patch.object(
                HardwareFingerprinter, "get_mac_address", return_value=0x001122334455
            ):
                fp1 = HardwareFingerprinter.get_hardware_fingerprint()

            with patch.object(
                HardwareFingerprinter, "get_mac_address", return_value=0x001122334466
            ):
                fp2 = HardwareFingerprinter.get_hardware_fingerprint()

            assert fp1 != fp2

    def test_get_hardware_fingerprint_changes_on_host_mutation(self) -> None:
        """Produces different SHA-256 fingerprint when host identifier changes."""
        with patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455):
            with patch.object(HardwareFingerprinter, "get_stable_host_id", return_value="HOST-A"):
                fp1 = HardwareFingerprinter.get_hardware_fingerprint()

            with patch.object(HardwareFingerprinter, "get_stable_host_id", return_value="HOST-B"):
                fp2 = HardwareFingerprinter.get_hardware_fingerprint()

            assert fp1 != fp2

    def test_get_hardware_fingerprint_fails_closed_on_invalid_mac(self) -> None:
        """Fails closed and raises HardwareBindingError if MAC address check fails."""
        with (
            patch("uuid.getnode", return_value=0x010000000000),  # multicast bit set
            pytest.raises(HardwareBindingError),
        ):
            HardwareFingerprinter.get_hardware_fingerprint()
