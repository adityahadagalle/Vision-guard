"""Monotonic hash-chaining engine for cryptographic inference receipts.

Constructs and verifies tamper-evident, strictly monotonic hash-linked chains of
inference receipts per edge device. Enforces genesis block invariants, continuous
sequence numbers (preventing replay and drop attacks), and bit-for-bit RFC 8785
SHA-256 linkage across consecutive inference events.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.signer import InferenceSigner


class ChainIntegrityError(ValueError):
    """Base class for all hash-chain integrity violations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = "CHAIN_INTEGRITY_ERROR"
        self.details = details or {}


class ChainSequenceError(ChainIntegrityError):
    """Raised when sequence numbers skip, regress, or violate genesis constraints."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.code = "CHAIN_SEQUENCE_ERROR"


class ChainLinkError(ChainIntegrityError):
    """Raised when previous_receipt_hash does not match the previous receipt digest."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.code = "CHAIN_LINK_ERROR"


class ChainDeviceMismatchError(ChainIntegrityError):
    """Raised when receipts from different edge devices are linked together."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.code = "CHAIN_DEVICE_MISMATCH_ERROR"


class ChainSignatureVerificationError(ChainIntegrityError):
    """Raised when an Ed25519 signature fails cryptographic verification."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.code = "CHAIN_SIGNATURE_VERIFICATION_ERROR"


class MonotonicHashChainEngine:
    """Deterministic hash-chain engine for offline inference integrity receipts."""

    @classmethod
    def construct_next_receipt(
        cls,
        device_id: str,
        model_sha256: str,
        input_hash: str,
        output: dict[str, Any],
        private_key: bytes,
        public_key: bytes,
        previous_receipt: InferenceReceiptPayload | None = None,
        *,
        timestamp_iso: str | None = None,
    ) -> tuple[InferenceReceiptPayload, str, str]:
        """Constructs, hashes, and cryptographically signs the next monotonic receipt in the chain.

        Enforces:
        - Rule 1 (Genesis): If previous_receipt is None, sequence_number is 0 and
          previous_receipt_hash is None.
        - Rule 2 (Monotonicity): If previous_receipt is present, sequence_number is
          strictly previous_receipt.sequence_number + 1.
        - Rule 3 (Cryptographic Linkage): previous_receipt_hash is the SHA-256 hex digest
          of the RFC 8785 canonical serialization of previous_receipt.
        - Immediate Verification: The generated Ed25519 signature is verified against public_key
          before returning to ensure cryptographic consistency.

        Args:
            device_id: Unique hardware identifier of the capturing edge node.
            model_sha256: 64-character SHA-256 digest of the model checkpoint.
            input_hash: 64-character SHA-256 digest of the input data.
            output: Inference output dictionary (classifications, boxes, scores).
            private_key: Ed25519 private key in PKCS8 PEM format.
            public_key: Ed25519 public key in SubjectPublicKeyInfo PEM format.
            previous_receipt: Preceding receipt in the chain, or None if genesis.
            timestamp_iso: Optional ISO 8601 timestamp string (generated in UTC if omitted).

        Returns:
            A tuple of (payload, payload_hash, signature_b64):
                - payload: Validated InferenceReceiptPayload domain entity.
                - payload_hash: 64-character SHA-256 digest of canonicalized payload.
                - signature_b64: Base64-encoded Ed25519 digital signature.

        Raises:
            ChainDeviceMismatchError: If previous_receipt belongs to a different device.
            ChainSignatureVerificationError: If the signature fails immediate verification.
        """
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string.")

        if previous_receipt is not None:
            if previous_receipt.edge_device_id != device_id:
                raise ChainDeviceMismatchError(
                    f"Device mismatch: previous receipt belongs to '{previous_receipt.edge_device_id}', "
                    f"but requested device is '{device_id}'."
                )
            sequence_number = previous_receipt.sequence_number + 1
            previous_receipt_hash = compute_payload_hash(previous_receipt)
        else:
            sequence_number = 0
            previous_receipt_hash = None

        ts = timestamp_iso or datetime.now(UTC).isoformat()

        payload = InferenceReceiptPayload(
            edge_device_id=device_id,
            timestamp_iso=ts,
            sequence_number=sequence_number,
            model_sha256=model_sha256,
            input_hash=input_hash,
            inference_output=output,
            previous_receipt_hash=previous_receipt_hash,
        )

        payload_hash = compute_payload_hash(payload)
        signature_b64 = InferenceSigner.sign_payload(payload, private_key)

        # Immediate self-verification
        is_valid = InferenceSigner.verify_signature(payload, signature_b64, public_key)
        if not is_valid:
            raise ChainSignatureVerificationError(
                "Generated signature failed immediate cryptographic verification with provided public key."
            )

        return payload, payload_hash, signature_b64

    @classmethod
    def verify_receipt_link(
        cls,
        current_receipt: InferenceReceiptPayload,
        previous_receipt: InferenceReceiptPayload | None = None,
    ) -> None:
        """Verifies cryptographic and sequence continuity between consecutive receipts.

        Args:
            current_receipt: The receipt being validated.
            previous_receipt: The immediately preceding receipt, or None if current is genesis.

        Raises:
            ChainSequenceError: If sequence numbers are discontinuous or genesis is non-zero.
            ChainLinkError: If previous_receipt_hash does not match expected SHA-256 digest.
            ChainDeviceMismatchError: If receipts belong to different edge devices.
        """
        if previous_receipt is None:
            if current_receipt.sequence_number != 0:
                raise ChainSequenceError(
                    f"Genesis receipt must have sequence_number 0, got {current_receipt.sequence_number}."
                )
            if current_receipt.previous_receipt_hash is not None:
                raise ChainLinkError(
                    f"Genesis receipt must have previous_receipt_hash None, "
                    f"got '{current_receipt.previous_receipt_hash}'."
                )
            return

        if current_receipt.edge_device_id != previous_receipt.edge_device_id:
            raise ChainDeviceMismatchError(
                f"Edge device mismatch in receipt chain: current='{current_receipt.edge_device_id}', "
                f"previous='{previous_receipt.edge_device_id}'."
            )

        expected_sequence = previous_receipt.sequence_number + 1
        if current_receipt.sequence_number != expected_sequence:
            raise ChainSequenceError(
                f"Discontinuous sequence detected for device '{current_receipt.edge_device_id}': "
                f"expected sequence {expected_sequence}, got {current_receipt.sequence_number}."
            )

        expected_hash = compute_payload_hash(previous_receipt)
        if current_receipt.previous_receipt_hash != expected_hash:
            raise ChainLinkError(
                f"Broken hash chain link for device '{current_receipt.edge_device_id}' sequence "
                f"{current_receipt.sequence_number}: expected previous_receipt_hash '{expected_hash}', "
                f"got '{current_receipt.previous_receipt_hash}'."
            )

    @classmethod
    def verify_chain(
        cls,
        receipts: Sequence[InferenceReceiptPayload],
    ) -> None:
        """Verifies that an entire ordered sequence of receipts maintains strict cryptographic chain continuity.

        Args:
            receipts: Sequence of InferenceReceiptPayload instances starting at Genesis (sequence 0).

        Raises:
            ChainSequenceError: On sequence discontinuity or non-zero start.
            ChainLinkError: On broken hash linkage.
            ChainDeviceMismatchError: If any receipt device_id differs.
        """
        if not receipts:
            return

        cls.verify_receipt_link(receipts[0], None)
        for i in range(1, len(receipts)):
            cls.verify_receipt_link(receipts[i], receipts[i - 1])

    @classmethod
    def verify_subchain(
        cls,
        receipts: Sequence[InferenceReceiptPayload],
    ) -> None:
        """Verifies contiguous linkage between a subchain of receipts without requiring genesis.

        Args:
            receipts: Sequence of contiguous InferenceReceiptPayload instances.

        Raises:
            ChainSequenceError: On sequence discontinuity.
            ChainLinkError: On broken hash linkage.
            ChainDeviceMismatchError: If any receipt device_id differs.
        """
        if len(receipts) <= 1:
            return

        for i in range(1, len(receipts)):
            cls.verify_receipt_link(receipts[i], receipts[i - 1])


__all__ = [
    "ChainDeviceMismatchError",
    "ChainIntegrityError",
    "ChainLinkError",
    "ChainSequenceError",
    "ChainSignatureVerificationError",
    "MonotonicHashChainEngine",
]
