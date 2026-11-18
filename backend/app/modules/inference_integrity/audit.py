"""Offline chain audit verification engine and tamper detection suite.

Performs forensic verification of cryptographic inference receipt chains stored
in SQLite. Detects data mutation (signature mismatch), record deletion or dropped
telemetry (sequence gaps), and unauthorized chain re-branching (hash link breaks).
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.models import InferenceReceiptModel
from backend.app.db.repositories.inference_repository import SqliteInferenceRepository
from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.signer import InferenceSigner


class AuditResult(BaseModel):
    """Forensic verification result for an edge device receipt chain."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    device_id: str
    verified_count: int
    break_sequence_number: int | None = None
    reason: str | None = None
    error_code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def IS_VALID(self) -> bool:  # noqa: N802
        """Uppercase alias for compatibility with task invariants."""
        return self.is_valid

    def __bool__(self) -> bool:
        """Enables boolean evaluation of the audit result."""
        return self.is_valid


class ChainAuditEngine:
    """Offline cryptographic audit verification engine for inference receipt chains."""

    @classmethod
    def verify_device_chain(
        cls,
        device_id: str,
        repository: SqliteInferenceRepository,
    ) -> AuditResult:
        """Retrieves and cryptographically audits the full receipt chain for a device.

        Execution checks per receipt:
        a. Signature Check: signature_b64 must be valid for the payload and public_key_pem.
        b. Hash Link Check: previous_receipt_hash must equal preceding receipt payload_hash
           (Genesis must have None).
        c. Sequence Check: sequence_number must strictly equal n_prev + 1
           (Genesis must be 0).
        d. Public Key Consistency: public_key_pem must remain unchanged across the chain.

        Args:
            device_id: Unique hardware identifier for the edge device to audit.
            repository: Active SqliteInferenceRepository connected to the SQLite database.

        Returns:
            An AuditResult detailing validity, verified record count, and failure context.
        """
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string.")

        receipts = repository.get_all_receipts_for_device_ascending(device_id)
        return cls.verify_receipt_sequence(device_id, receipts)

    @classmethod
    def verify_receipt_sequence(
        cls,
        device_id: str,
        receipts: Sequence[InferenceReceiptModel],
    ) -> AuditResult:
        """Verifies an in-memory sequence of InferenceReceiptModel records.

        Halts immediately upon detecting the first integrity violation.

        Args:
            device_id: The expected edge hardware identifier.
            receipts: Sequence of InferenceReceiptModel records ordered by sequence_number ascending.

        Returns:
            An AuditResult summarizing forensic findings.
        """
        if not receipts:
            return AuditResult(
                is_valid=True,
                device_id=device_id,
                verified_count=0,
                break_sequence_number=None,
                reason=None,
                error_code=None,
                details={"message": "No receipts recorded for device."},
            )

        initial_public_key: str | None = None

        for idx, current_record in enumerate(receipts):
            # 1. Device match verification
            if current_record.edge_device_id != device_id:
                return AuditResult(
                    is_valid=False,
                    device_id=device_id,
                    verified_count=idx,
                    break_sequence_number=current_record.sequence_number,
                    reason="Device mismatch",
                    error_code="DEVICE_MISMATCH",
                    details={
                        "expected_device_id": device_id,
                        "actual_device_id": current_record.edge_device_id,
                        "receipt_id": current_record.id,
                    },
                )

            # Reconstruct domain payload for cryptographic verification
            try:
                payload = InferenceReceiptPayload(
                    edge_device_id=current_record.edge_device_id,
                    timestamp_iso=current_record.timestamp_iso,
                    sequence_number=current_record.sequence_number,
                    model_sha256=current_record.model_sha256,
                    input_hash=current_record.input_hash,
                    inference_output=current_record.inference_output,
                    previous_receipt_hash=current_record.previous_receipt_hash,
                )
            except Exception as exc:
                return AuditResult(
                    is_valid=False,
                    device_id=device_id,
                    verified_count=idx,
                    break_sequence_number=current_record.sequence_number,
                    reason=f"Payload schema validation error: {exc}",
                    error_code="SCHEMA_VALIDATION_ERROR",
                    details={"receipt_id": current_record.id, "error": str(exc)},
                )

            # 2. Public key consistency across the device chain
            if initial_public_key is None:
                initial_public_key = current_record.public_key_pem
            elif current_record.public_key_pem != initial_public_key:
                return AuditResult(
                    is_valid=False,
                    device_id=device_id,
                    verified_count=idx,
                    break_sequence_number=current_record.sequence_number,
                    reason="Public key mismatch across chain",
                    error_code="PUBLIC_KEY_MISMATCH",
                    details={
                        "sequence_number": current_record.sequence_number,
                        "receipt_id": current_record.id,
                    },
                )

            # 3. Payload hash integrity verification
            computed_payload_hash = compute_payload_hash(payload)
            if current_record.payload_hash != computed_payload_hash:
                return AuditResult(
                    is_valid=False,
                    device_id=device_id,
                    verified_count=idx,
                    break_sequence_number=current_record.sequence_number,
                    reason="Payload hash mismatch",
                    error_code="PAYLOAD_HASH_MISMATCH",
                    details={
                        "expected_hash": computed_payload_hash,
                        "stored_hash": current_record.payload_hash,
                        "receipt_id": current_record.id,
                    },
                )

            # 4. Signature verification (RFC 8032 Ed25519)
            pub_bytes = current_record.public_key_pem.encode("utf-8")
            try:
                is_sig_valid = InferenceSigner.verify_signature(
                    payload,
                    current_record.signature_b64,
                    pub_bytes,
                )
            except Exception:
                is_sig_valid = False

            if not is_sig_valid:
                return AuditResult(
                    is_valid=False,
                    device_id=device_id,
                    verified_count=idx,
                    break_sequence_number=current_record.sequence_number,
                    reason="Signature mismatch",
                    error_code="SIGNATURE_MISMATCH",
                    details={
                        "sequence_number": current_record.sequence_number,
                        "receipt_id": current_record.id,
                    },
                )

            # 5. Genesis vs. Chaining Linkage
            if idx == 0:
                # Genesis receipt constraints
                if current_record.sequence_number != 0:
                    return AuditResult(
                        is_valid=False,
                        device_id=device_id,
                        verified_count=idx,
                        break_sequence_number=current_record.sequence_number,
                        reason="Sequence gap: genesis sequence must be 0",
                        error_code="GENESIS_SEQUENCE_ERROR",
                        details={
                            "actual_sequence": current_record.sequence_number,
                            "expected_sequence": 0,
                            "receipt_id": current_record.id,
                        },
                    )
                if current_record.previous_receipt_hash is not None:
                    return AuditResult(
                        is_valid=False,
                        device_id=device_id,
                        verified_count=idx,
                        break_sequence_number=current_record.sequence_number,
                        reason="Hash link broken: genesis previous_receipt_hash must be None",
                        error_code="GENESIS_LINK_ERROR",
                        details={
                            "previous_receipt_hash": current_record.previous_receipt_hash,
                            "receipt_id": current_record.id,
                        },
                    )
            else:
                prev_record = receipts[idx - 1]

                # Check strict monotonic sequence continuity: n = n_prev + 1
                expected_seq = prev_record.sequence_number + 1
                if current_record.sequence_number != expected_seq:
                    return AuditResult(
                        is_valid=False,
                        device_id=device_id,
                        verified_count=idx,
                        break_sequence_number=current_record.sequence_number,
                        reason="Sequence gap",
                        error_code="SEQUENCE_GAP",
                        details={
                            "expected_sequence": expected_seq,
                            "actual_sequence": current_record.sequence_number,
                            "previous_sequence": prev_record.sequence_number,
                            "receipt_id": current_record.id,
                        },
                    )

                # Check cryptographic hash linkage: previous_receipt_hash == prev_record.payload_hash
                if current_record.previous_receipt_hash != prev_record.payload_hash:
                    return AuditResult(
                        is_valid=False,
                        device_id=device_id,
                        verified_count=idx,
                        break_sequence_number=current_record.sequence_number,
                        reason="Hash link broken",
                        error_code="HASH_LINK_BROKEN",
                        details={
                            "expected_previous_receipt_hash": prev_record.payload_hash,
                            "actual_previous_receipt_hash": current_record.previous_receipt_hash,
                            "receipt_id": current_record.id,
                        },
                    )

        # All receipts cryptographically verified and unbroken
        return AuditResult(
            is_valid=True,
            device_id=device_id,
            verified_count=len(receipts),
            break_sequence_number=None,
            reason=None,
            error_code=None,
            details={"status": "CHAIN_VERIFIED_INTACT"},
        )


__all__ = [
    "AuditResult",
    "ChainAuditEngine",
]
