"""API endpoints for secure edge inference receipt ingestion, verification, and hash-chain persistence.

Implements the 4-step Epistemic Verification Pipeline:
1. Strict Ed25519 cryptographic signature verification against the provided public key (RFC 8032).
2. Genesis invariant vs. continuous hash-chain linkage check (RFC 8785 SHA-256).
3. Replay attack and sequence gap detection (monotonically enforced sequence numbers).
4. Relational persistence with SQLite compound unique constraints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from backend.app.db.repositories.inference_repository import (
    DuplicateReceiptError,
    SqliteInferenceRepository,
)
from backend.app.db.session import get_db
from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.signer import InferenceSigner


class InferenceReceiptRequest(BaseModel):
    """Request payload containing an inference receipt and cryptographic proof."""

    model_config = ConfigDict(extra="forbid")

    payload: InferenceReceiptPayload
    signature: str = Field(
        ..., min_length=1, description="Base64-encoded Ed25519 digital signature."
    )
    public_key_pem: str = Field(
        ..., min_length=1, description="SubjectPublicKeyInfo Ed25519 public key in PEM format."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalizes either nested {'payload': {...}, 'signature': ..., 'public_key_pem': ...}
        or flat structure for ease of edge ingestion."""
        if isinstance(data, dict):
            # Normalize signature_b64 -> signature if provided
            if "signature_b64" in data and "signature" not in data:
                data = dict(data)
                data["signature"] = data.pop("signature_b64")

            if "payload" not in data and "edge_device_id" in data:
                payload_fields = {
                    "edge_device_id",
                    "timestamp_iso",
                    "sequence_number",
                    "model_sha256",
                    "input_hash",
                    "inference_output",
                    "previous_receipt_hash",
                }
                payload_dict = {k: data[k] for k in payload_fields if k in data}
                remaining = {k: v for k, v in data.items() if k not in payload_fields}
                remaining["payload"] = payload_dict
                return remaining
        return data


class InferenceReceiptResponse(BaseModel):
    """Response returned upon successful receipt ingestion."""

    model_config = ConfigDict(extra="forbid")

    receipt_id: str
    edge_device_id: str
    sequence_number: int
    payload_hash: str
    signature: str
    status: str = "ACCEPTED"


router = APIRouter(prefix="/inference", tags=["Inference Integrity"])


@router.post(
    "/receipts",
    response_model=InferenceReceiptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and cryptographically verify an edge inference receipt",
)
def ingest_receipt(
    request: InferenceReceiptRequest,
    db: Session = Depends(get_db),
) -> InferenceReceiptResponse:
    """Ingests an inference receipt adhering to the Epistemic Verification Pipeline.

    Execution order:
    1. Signature Check: Verify signature using InferenceSigner.verify_signature.
       Returns HTTP 401 Unauthorized immediately if invalid.
    2. Genesis vs. Chain State Check:
       - If no receipts exist: sequence MUST be 0 and previous_receipt_hash MUST be None.
       - If receipts exist: sequence MUST be latest.sequence_number + 1, and
         previous_receipt_hash MUST equal latest.payload_hash.
    3. Replay / Sequence Gap Detection:
       - If sequence <= latest.sequence_number: Replay Attack (HTTP 409 Conflict).
       - If sequence > latest.sequence_number + 1: Sequence Gap (HTTP 409 Conflict).
    4. Persistence: Write to SQLite via SqliteInferenceRepository.
    """
    # 1. Signature Check (Execution order: Must precede any database lookup)
    pub_bytes = request.public_key_pem.encode("utf-8")
    try:
        is_valid = InferenceSigner.verify_signature(
            request.payload,
            request.signature,
            pub_bytes,
        )
    except Exception:
        is_valid = False

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "INVALID_SIGNATURE",
                "message": "Cryptographic signature verification failed against provided public key.",
                "details": {"edge_device_id": request.payload.edge_device_id},
            },
        )

    # 2. Genesis vs. Chain State & Replay / Gap Detection
    repo = SqliteInferenceRepository(db)
    latest = repo.get_latest_receipt(request.payload.edge_device_id)

    if latest is None:
        # Genesis block constraints
        if request.payload.sequence_number != 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "SEQUENCE_ERROR",
                    "message": (
                        f"Genesis receipt for device '{request.payload.edge_device_id}' "
                        f"must have sequence_number 0, but received {request.payload.sequence_number}."
                    ),
                    "details": {
                        "edge_device_id": request.payload.edge_device_id,
                        "expected_sequence": 0,
                        "received_sequence": request.payload.sequence_number,
                    },
                },
            )
        if request.payload.previous_receipt_hash is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "CHAIN_LINK_ERROR",
                    "message": (
                        f"Genesis receipt for device '{request.payload.edge_device_id}' "
                        "must have previous_receipt_hash=None."
                    ),
                    "details": {
                        "edge_device_id": request.payload.edge_device_id,
                        "previous_receipt_hash": request.payload.previous_receipt_hash,
                    },
                },
            )
    else:
        # Replay detection: n <= latest.sequence_number
        if request.payload.sequence_number <= latest.sequence_number:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "REPLAY_DETECTED",
                    "message": (
                        f"Replay attack detected for device '{request.payload.edge_device_id}': "
                        f"received sequence {request.payload.sequence_number} <= "
                        f"latest recorded sequence {latest.sequence_number}."
                    ),
                    "details": {
                        "edge_device_id": request.payload.edge_device_id,
                        "latest_sequence": latest.sequence_number,
                        "received_sequence": request.payload.sequence_number,
                    },
                },
            )

        # Sequence gap detection: n > latest.sequence_number + 1
        if request.payload.sequence_number > latest.sequence_number + 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "SEQUENCE_GAP_DETECTED",
                    "message": (
                        f"Sequence gap detected for device '{request.payload.edge_device_id}': "
                        f"received sequence {request.payload.sequence_number}, "
                        f"expected {latest.sequence_number + 1}."
                    ),
                    "details": {
                        "edge_device_id": request.payload.edge_device_id,
                        "expected_sequence": latest.sequence_number + 1,
                        "received_sequence": request.payload.sequence_number,
                    },
                },
            )

        # Hash link verification: previous_receipt_hash == latest.payload_hash
        if request.payload.previous_receipt_hash != latest.payload_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "HASH_LINK_MISMATCH",
                    "message": (
                        f"Hash link mismatch for device '{request.payload.edge_device_id}': "
                        f"expected previous_receipt_hash '{latest.payload_hash}', "
                        f"got '{request.payload.previous_receipt_hash}'."
                    ),
                    "details": {
                        "edge_device_id": request.payload.edge_device_id,
                        "expected_previous_receipt_hash": latest.payload_hash,
                        "received_previous_receipt_hash": request.payload.previous_receipt_hash,
                    },
                },
            )

    # 4. Persistence
    payload_hash = compute_payload_hash(request.payload)

    try:
        receipt = repo.create_receipt(
            edge_device_id=request.payload.edge_device_id,
            timestamp_iso=request.payload.timestamp_iso,
            sequence_number=request.payload.sequence_number,
            model_sha256=request.payload.model_sha256,
            input_hash=request.payload.input_hash,
            inference_output=request.payload.inference_output,
            previous_receipt_hash=request.payload.previous_receipt_hash,
            payload_hash=payload_hash,
            signature_b64=request.signature,
            public_key_pem=request.public_key_pem,
        )
    except DuplicateReceiptError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "DUPLICATE_RECEIPT",
                "message": str(exc),
                "details": exc.details,
            },
        ) from exc

    return InferenceReceiptResponse(
        receipt_id=receipt.id,
        edge_device_id=receipt.edge_device_id,
        sequence_number=receipt.sequence_number,
        payload_hash=receipt.payload_hash,
        signature=receipt.signature_b64,
        status="ACCEPTED",
    )


@router.get(
    "/receipts/{edge_device_id}/latest",
    response_model=InferenceReceiptResponse,
    summary="Get latest inference receipt for a specific edge device",
)
def get_latest_receipt(
    edge_device_id: str,
    db: Session = Depends(get_db),
) -> InferenceReceiptResponse:
    """Retrieves the latest verified receipt for an edge hardware node."""
    repo = SqliteInferenceRepository(db)
    receipt = repo.get_latest_receipt(edge_device_id)
    if not receipt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "RECEIPT_NOT_FOUND",
                "message": f"No inference receipts found for edge device '{edge_device_id}'.",
                "details": {"edge_device_id": edge_device_id},
            },
        )
    return InferenceReceiptResponse(
        receipt_id=receipt.id,
        edge_device_id=receipt.edge_device_id,
        sequence_number=receipt.sequence_number,
        payload_hash=receipt.payload_hash,
        signature=receipt.signature_b64,
        status="ACCEPTED",
    )
