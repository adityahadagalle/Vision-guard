"""SQLite repository adapter for cryptographic inference receipts and hash-chain persistence."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.models import InferenceReceiptModel


class DuplicateReceiptError(ValueError):
    """Raised when an inference receipt violates unique sequence number or payload hash constraints."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = "DUPLICATE_RECEIPT_ERROR"
        self.details = details or {}


class SqliteInferenceRepository:
    """Repository managing immutable cryptographic inference receipts in SQLite."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_receipt(
        self,
        *,
        edge_device_id: str,
        timestamp_iso: str,
        sequence_number: int,
        model_sha256: str,
        input_hash: str,
        inference_output: dict[str, Any],
        previous_receipt_hash: str | None,
        payload_hash: str,
        signature_b64: str,
        public_key_pem: str,
        receipt_id: str | None = None,
    ) -> InferenceReceiptModel:
        """Persists a new cryptographic inference receipt record.

        Args:
            edge_device_id: Unique hardware identifier of the capturing node.
            timestamp_iso: ISO 8601 creation timestamp string.
            sequence_number: Strictly monotonic sequence number for this edge device.
            model_sha256: SHA-256 digest of the model checkpoint.
            input_hash: SHA-256 digest of the input data.
            inference_output: Model inference output dictionary.
            previous_receipt_hash: SHA-256 digest of previous receipt in chain, or None if genesis.
            payload_hash: Unique SHA-256 digest of canonicalized payload bytes.
            signature_b64: Ed25519 signature encoded in Base64.
            public_key_pem: Ed25519 public key in SubjectPublicKeyInfo PEM format.
            receipt_id: Optional UUID string identifier. If omitted, a UUIDv4 string is generated.

        Returns:
            The persisted and refreshed InferenceReceiptModel instance.

        Raises:
            DuplicateReceiptError: If (edge_device_id, sequence_number) or payload_hash collides.
        """
        record_id = receipt_id or str(uuid.uuid4())
        model = InferenceReceiptModel(
            id=record_id,
            edge_device_id=edge_device_id,
            timestamp_iso=timestamp_iso,
            sequence_number=sequence_number,
            model_sha256=model_sha256,
            input_hash=input_hash,
            inference_output=inference_output,
            previous_receipt_hash=previous_receipt_hash,
            payload_hash=payload_hash,
            signature_b64=signature_b64,
            public_key_pem=public_key_pem,
        )
        try:
            self._session.add(model)
            self._session.commit()
            self._session.refresh(model)
            return model
        except IntegrityError as exc:
            self._session.rollback()
            raise DuplicateReceiptError(
                f"Receipt constraint violation for device '{edge_device_id}' "
                f"sequence {sequence_number} or payload_hash '{payload_hash}': {exc}",
                details={
                    "edge_device_id": edge_device_id,
                    "sequence_number": sequence_number,
                    "payload_hash": payload_hash,
                },
            ) from exc

    def get_latest_receipt(self, edge_device_id: str) -> InferenceReceiptModel | None:
        """Retrieves the latest receipt for a specific edge device ordered by sequence_number descending."""
        stmt = (
            select(InferenceReceiptModel)
            .where(InferenceReceiptModel.edge_device_id == edge_device_id)
            .order_by(InferenceReceiptModel.sequence_number.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_receipt_by_id(self, receipt_id: str) -> InferenceReceiptModel | None:
        """Retrieves an inference receipt by its primary key UUID."""
        stmt = select(InferenceReceiptModel).where(InferenceReceiptModel.id == receipt_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_receipt_by_payload_hash(self, payload_hash: str) -> InferenceReceiptModel | None:
        """Retrieves an inference receipt by its unique payload hash."""
        stmt = select(InferenceReceiptModel).where(
            InferenceReceiptModel.payload_hash == payload_hash
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_receipt_by_sequence(
        self, edge_device_id: str, sequence_number: int
    ) -> InferenceReceiptModel | None:
        """Retrieves an inference receipt for a given edge device and sequence number."""
        stmt = select(InferenceReceiptModel).where(
            InferenceReceiptModel.edge_device_id == edge_device_id,
            InferenceReceiptModel.sequence_number == sequence_number,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_receipts_for_device(
        self,
        edge_device_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[InferenceReceiptModel]:
        """Lists receipts for a device ordered chronologically by sequence_number ascending."""
        stmt = (
            select(InferenceReceiptModel)
            .where(InferenceReceiptModel.edge_device_id == edge_device_id)
            .order_by(InferenceReceiptModel.sequence_number.asc())
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_all_receipts_for_device_ascending(
        self, edge_device_id: str
    ) -> list[InferenceReceiptModel]:
        """Retrieves all inference receipts for a device ordered chronologically by sequence_number ascending."""
        stmt = (
            select(InferenceReceiptModel)
            .where(InferenceReceiptModel.edge_device_id == edge_device_id)
            .order_by(InferenceReceiptModel.sequence_number.asc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def count_receipts_for_device(self, edge_device_id: str) -> int:
        """Counts total persisted receipts for a given edge device."""
        stmt = select(InferenceReceiptModel).where(
            InferenceReceiptModel.edge_device_id == edge_device_id
        )
        return len(list(self._session.execute(stmt).scalars().all()))


InferenceRepository = SqliteInferenceRepository

__all__ = [
    "DuplicateReceiptError",
    "InferenceRepository",
    "SqliteInferenceRepository",
]
