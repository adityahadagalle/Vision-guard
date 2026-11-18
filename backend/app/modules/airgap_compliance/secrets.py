"""Air-gap hardware-bound cryptographic secrets manager.

Derives authenticated symmetric encryption keys from local hardware fingerprints
(MAC address, machine GUID, CPU identifier) using HKDF-SHA256, binding secrets
cryptographically to the underlying physical or virtual host.
"""

from __future__ import annotations

import base64
from typing import ClassVar

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from backend.app.modules.airgap_compliance.hardware import (
    HardwareBindingError,
    HardwareFingerprinter,
)

__all__ = [
    "AirgapSecretsManager",
    "HardwareBindingError",
    "InvalidToken",
    "SecretDecryptionError",
]


class SecretDecryptionError(Exception):
    """Raised when hardware-bound secret decryption fails due to tamper or host mismatch."""


class AirgapSecretsManager:
    """Cryptographic secrets manager enforcing host hardware binding for air-gapped deployments."""

    KDF_SALT: ClassVar[bytes] = b"visionguard-airgap-salt-2026"
    KDF_INFO: ClassVar[bytes] = b"visionguard-hardware-binding-v1"

    def __init__(
        self,
        fingerprinter: type[HardwareFingerprinter] | None = None,
        custom_fingerprint: str | None = None,
    ) -> None:
        self._fingerprinter = fingerprinter or HardwareFingerprinter
        self._custom_fingerprint = custom_fingerprint

    def _get_fingerprint(self) -> str:
        """Retrieve hardware fingerprint, failing closed on invalid hardware attributes."""
        if self._custom_fingerprint is not None:
            return self._custom_fingerprint
        return self._fingerprinter.get_hardware_fingerprint()

    def _derive_fernet_key(self, fingerprint: str) -> bytes:
        """Derive a 256-bit URL-safe base64 Fernet key from the hardware fingerprint via HKDF-SHA256."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.KDF_SALT,
            info=self.KDF_INFO,
        )
        derived = hkdf.derive(fingerprint.encode("utf-8"))
        return base64.urlsafe_b64encode(derived)

    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt plaintext secret bound to the current hardware fingerprint.

        Raises:
            HardwareBindingError: If genuine hardware identification fails.
        """
        fingerprint = self._get_fingerprint()
        key = self._derive_fernet_key(fingerprint)
        fernet = Fernet(key)
        return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt_secret(self, ciphertext: str) -> str:
        """Decrypt ciphertext secret enforcing hardware binding.

        Raises:
            HardwareBindingError: If genuine hardware identification fails.
            InvalidToken: If the hardware fingerprint does not match or ciphertext was tampered with.
        """
        fingerprint = self._get_fingerprint()
        key = self._derive_fernet_key(fingerprint)
        fernet = Fernet(key)
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    @classmethod
    def encrypt(cls, plaintext: str) -> str:
        """Class method convenience wrapper for encrypt_secret."""
        return cls().encrypt_secret(plaintext)

    @classmethod
    def decrypt(cls, ciphertext: str) -> str:
        """Class method convenience wrapper for decrypt_secret."""
        return cls().decrypt_secret(ciphertext)
