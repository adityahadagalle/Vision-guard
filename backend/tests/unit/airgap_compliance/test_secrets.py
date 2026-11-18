"""Unit tests for Air-Gap Secrets Manager.

Verifies hardware-bound secret encryption/decryption, anti-cloning cryptographic
protection via HKDF-SHA256 and Fernet, and fail-closed behavior on hardware mismatch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from cryptography.fernet import InvalidToken

from backend.app.modules.airgap_compliance.hardware import (
    HardwareBindingError,
    HardwareFingerprinter,
)
from backend.app.modules.airgap_compliance.secrets import AirgapSecretsManager


class TestAirgapSecretsManager:
    """Test suite for AirgapSecretsManager hardware-bound encryption."""

    def test_successful_encryption_and_decryption_on_same_hardware(self) -> None:
        """Encrypts and decrypts secret successfully when executed on the same hardware."""
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455),
            patch.object(
                HardwareFingerprinter, "get_stable_host_id", return_value="SECURE-NODE-01"
            ),
        ):
            manager = AirgapSecretsManager()
            plaintext = "super-secret-offline-token-2026"
            ciphertext = manager.encrypt_secret(plaintext)

            assert isinstance(ciphertext, str)
            assert ciphertext != plaintext

            decrypted = manager.decrypt_secret(ciphertext)
            assert decrypted == plaintext

    def test_decryption_failure_when_hardware_fingerprint_changes_cloned_drive(self) -> None:
        """Raises InvalidToken when attempting to decrypt secret on different hardware (cloned disk scenario)."""
        # Machine A: Source hardware where secret was encrypted
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455),
            patch.object(
                HardwareFingerprinter, "get_stable_host_id", return_value="SOURCE-MACHINE-A"
            ),
        ):
            manager_a = AirgapSecretsManager()
            ciphertext = manager_a.encrypt_secret("sensitive-model-signing-key")

        # Machine B: Foreign hardware (e.g. image clone or stolen drive)
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x00AABBCCDDEE),
            patch.object(
                HardwareFingerprinter, "get_stable_host_id", return_value="FOREIGN-MACHINE-B"
            ),
            pytest.raises(InvalidToken),
        ):
            manager_b = AirgapSecretsManager()
            manager_b.decrypt_secret(ciphertext)

    def test_decryption_failure_when_only_mac_changes(self) -> None:
        """Fails decryption if MAC address differs even if host ID matches."""
        with patch.object(HardwareFingerprinter, "get_stable_host_id", return_value="STATIC-HOST"):
            with patch.object(
                HardwareFingerprinter, "get_mac_address", return_value=0x001122334455
            ):
                ciphertext = AirgapSecretsManager().encrypt_secret("vault-passphrase")

            with (
                patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334466),
                pytest.raises(InvalidToken),
            ):
                AirgapSecretsManager().decrypt_secret(ciphertext)

    def test_decryption_failure_when_only_host_id_changes(self) -> None:
        """Fails decryption if host ID differs even if MAC matches."""
        with patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455):
            with patch.object(
                HardwareFingerprinter, "get_stable_host_id", return_value="HOST-ORIGINAL"
            ):
                ciphertext = AirgapSecretsManager().encrypt_secret("database-encryption-key")

            with (
                patch.object(
                    HardwareFingerprinter, "get_stable_host_id", return_value="HOST-MUTATED"
                ),
                pytest.raises(InvalidToken),
            ):
                AirgapSecretsManager().decrypt_secret(ciphertext)

    def test_fails_closed_when_mac_fails_multicast_check(self) -> None:
        """Raises HardwareBindingError when MAC is invalid or multicast dummy, preventing decryption."""
        with patch("uuid.getnode", return_value=0x010000000000):  # multicast dummy
            manager = AirgapSecretsManager()
            with pytest.raises(HardwareBindingError):
                manager.encrypt_secret("should-fail-immediately")

            with pytest.raises(HardwareBindingError):
                manager.decrypt_secret("dummy-ciphertext")

    def test_tampered_ciphertext_fails_decryption(self) -> None:
        """Raises InvalidToken if ciphertext payload is corrupted or modified."""
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455),
            patch.object(HardwareFingerprinter, "get_stable_host_id", return_value="HOST-SECURE"),
        ):
            manager = AirgapSecretsManager()
            ciphertext = manager.encrypt_secret("unaltered-plaintext")

            # Corrupt ciphertext bytes
            tampered = ciphertext[:-4] + ("AAAA" if ciphertext[-4:] != "AAAA" else "BBBB")
            with pytest.raises(InvalidToken):
                manager.decrypt_secret(tampered)

    def test_custom_fingerprint_support(self) -> None:
        """Supports explicit fingerprint injection for custom isolation contexts."""
        fp1 = "custom-static-fingerprint-001"
        fp2 = "custom-static-fingerprint-002"

        mgr1 = AirgapSecretsManager(custom_fingerprint=fp1)
        mgr2 = AirgapSecretsManager(custom_fingerprint=fp2)

        ciphertext = mgr1.encrypt_secret("custom-fp-data")
        assert mgr1.decrypt_secret(ciphertext) == "custom-fp-data"

        with pytest.raises(InvalidToken):
            mgr2.decrypt_secret(ciphertext)

    def test_classmethod_convenience_wrappers(self) -> None:
        """Verifies encrypt and decrypt classmethods operate correctly."""
        with (
            patch.object(HardwareFingerprinter, "get_mac_address", return_value=0x001122334455),
            patch.object(HardwareFingerprinter, "get_stable_host_id", return_value="CLASS-HOST"),
        ):
            ciphertext = AirgapSecretsManager.encrypt("classmethod-payload")
            decrypted = AirgapSecretsManager.decrypt(ciphertext)
            assert decrypted == "classmethod-payload"
