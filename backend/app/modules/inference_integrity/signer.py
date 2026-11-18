"""Ed25519 cryptographic signer and offline key custody for inference receipts.

Implements high-assurance Ed25519 digital signatures (RFC 8032) for InferenceReceiptPayload
instances. Operates strictly offline without external key-management dependencies or cloud HSMs.
Binds canonicalized payload bytes from RFC 8785 to tamper-evident cryptographic signatures.
"""

import base64
import contextlib
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from backend.app.core.config import get_settings
from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import canonicalize_payload


class InferenceSigner:
    """Ed25519 signer and verifier for inference provenance receipts.

    Provides deterministic signing, verification against registered public keys,
    and offline filesystem-based key custody with strict permission isolation.
    """

    def __init__(self, keys_dir: Path | None = None) -> None:
        self.keys_dir = Path(keys_dir) if keys_dir else get_settings().keys_dir

    @classmethod
    def generate_keypair(cls, device_id: str) -> tuple[bytes, bytes]:
        """Generates an Ed25519 private/public keypair in PEM format.

        Args:
            device_id: Identifier for the edge hardware node (for tracking context).

        Returns:
            A tuple of (private_key_pem, public_key_pem) as bytes.
        """
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string.")

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return private_pem, public_pem

    @classmethod
    def sign_payload(
        cls,
        payload: InferenceReceiptPayload,
        private_key_pem: bytes,
    ) -> str:
        """Serializes an inference payload via RFC 8785 and generates an Ed25519 signature.

        Args:
            payload: Validated InferenceReceiptPayload domain entity.
            private_key_pem: Ed25519 private key in PKCS8 PEM format.

        Returns:
            A standard Base64-encoded signature string (64 bytes raw).

        Raises:
            ValueError: If the private key cannot be parsed or is not an Ed25519 key.
        """
        try:
            loaded_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load Ed25519 private key PEM: {exc}") from exc

        if not isinstance(loaded_key, ed25519.Ed25519PrivateKey):
            raise ValueError(f"Expected Ed25519PrivateKey, got {type(loaded_key).__name__}")

        canonical_bytes = canonicalize_payload(payload)
        signature_bytes = loaded_key.sign(canonical_bytes)
        return base64.b64encode(signature_bytes).decode("ascii")

    @classmethod
    def verify_signature(
        cls,
        payload: InferenceReceiptPayload,
        signature_b64: str,
        public_key_pem: bytes,
    ) -> bool:
        """Verifies an Ed25519 signature against an InferenceReceiptPayload and public key.

        Args:
            payload: Validated InferenceReceiptPayload domain entity.
            signature_b64: Base64-encoded 64-byte Ed25519 signature.
            public_key_pem: Ed25519 public key in SubjectPublicKeyInfo PEM format.

        Returns:
            True if the signature is cryptographically valid for the payload; False otherwise.
        """
        try:
            loaded_key = serialization.load_pem_public_key(public_key_pem)
            if not isinstance(loaded_key, ed25519.Ed25519PublicKey):
                return False

            signature_bytes = base64.b64decode(signature_b64, validate=True)
            if len(signature_bytes) != 64:
                return False

            canonical_bytes = canonicalize_payload(payload)
            loaded_key.verify(signature_bytes, canonical_bytes)
            return True
        except (InvalidSignature, ValueError, TypeError, Exception):
            return False

    def save_keypair(
        self,
        device_id: str,
        private_key_pem: bytes,
        public_key_pem: bytes,
        target_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """Saves a generated keypair to offline disk custody.

        Enforces restrictive filesystem permissions on the private key (0o600 on POSIX).

        Args:
            device_id: Edge device identifier used as key filename base.
            private_key_pem: Private key PEM bytes.
            public_key_pem: Public key PEM bytes.
            target_dir: Optional override directory; defaults to self.keys_dir.

        Returns:
            Tuple of (private_key_path, public_key_path).
        """
        dest_dir = Path(target_dir) if target_dir else self.keys_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        priv_path = dest_dir / f"{device_id}.key"
        pub_path = dest_dir / f"{device_id}.pub"

        priv_path.write_bytes(private_key_pem)
        with contextlib.suppress(OSError, NotImplementedError):
            priv_path.chmod(0o600)

        pub_path.write_bytes(public_key_pem)
        with contextlib.suppress(OSError, NotImplementedError):
            pub_path.chmod(0o644)

        return priv_path, pub_path

    def load_keypair(
        self,
        device_id: str,
        target_dir: Path | None = None,
    ) -> tuple[bytes, bytes]:
        """Loads the private and public PEM keys for a device from disk custody."""
        src_dir = Path(target_dir) if target_dir else self.keys_dir
        priv_path = src_dir / f"{device_id}.key"
        pub_path = src_dir / f"{device_id}.pub"

        if not priv_path.exists():
            raise FileNotFoundError(f"Private key not found: {priv_path}")
        if not pub_path.exists():
            raise FileNotFoundError(f"Public key not found: {pub_path}")

        return priv_path.read_bytes(), pub_path.read_bytes()

    def load_public_key(
        self,
        device_id: str,
        target_dir: Path | None = None,
    ) -> bytes:
        """Loads only the public key PEM for a device from disk custody."""
        src_dir = Path(target_dir) if target_dir else self.keys_dir
        pub_path = src_dir / f"{device_id}.pub"
        if not pub_path.exists():
            raise FileNotFoundError(f"Public key not found: {pub_path}")
        return pub_path.read_bytes()


__all__ = ["InferenceSigner"]
