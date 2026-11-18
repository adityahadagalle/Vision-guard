"""Domain models for inference integrity receipts and provenance verification.

Defines the core immutable payload structure representing an individual inference event
at the edge. Enforces strict cryptographic digest formats, timestamp validity, and
forbids unauthorized schema mutations.
"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256_HEX_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


def _validate_sha256_digest(value: str, field_name: str) -> str:
    """Validates and normalizes a 64-character hexadecimal SHA-256 digest."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not _SHA256_HEX_PATTERN.match(value):
        raise ValueError(f"Invalid {field_name}: must be exactly 64 hexadecimal characters.")
    return value.lower()


class InferenceReceiptPayload(BaseModel):
    """Immutable, validated payload representing a single inference event.

    This payload serves as the canonical domain representation that is serialized
    via RFC 8785 JSON Canonicalization Scheme (JCS) before being digitally signed
    and linked into the tamper-evident hash chain.

    Attributes:
        edge_device_id: Unique identifier for the capturing edge device.
        timestamp_iso: Strict ISO 8601 timestamp string representing event time.
        sequence_number: Non-negative integer strictly incremented per edge device.
        model_sha256: 64-character hex digest of the model checkpoint used for inference.
        input_hash: 64-character hex digest of the input image/sensor payload.
        inference_output: Arbitrary JSON-serializable output dictionary from the model.
        previous_receipt_hash: 64-character hex digest of prior receipt (null for genesis).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    edge_device_id: str = Field(
        ...,
        description="Unique identifier for the recording edge hardware node.",
    )
    timestamp_iso: str = Field(
        ...,
        description="Event creation timestamp in strict ISO 8601 format.",
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Strictly monotonic sequence counter (0 for genesis receipt).",
    )
    model_sha256: str = Field(
        ...,
        description="64-character hexadecimal SHA-256 digest of active model weights.",
    )
    input_hash: str = Field(
        ...,
        description="64-character hexadecimal SHA-256 digest of raw input tensor/bytes.",
    )
    inference_output: dict[str, Any] = Field(
        ...,
        description="Arbitrary model prediction output (logits, boxes, classifications).",
    )
    previous_receipt_hash: str | None = Field(
        default=None,
        description="64-character SHA-256 of preceding receipt in chain, or None if genesis.",
    )

    @field_validator("edge_device_id")
    @classmethod
    def _validate_edge_device_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("edge_device_id must be a non-empty string.")
        return v.strip()

    @field_validator("timestamp_iso")
    @classmethod
    def _validate_timestamp_iso(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("timestamp_iso must be a non-empty string.")
        try:
            # Handle ISO 8601 representation including 'Z' or standard offsets
            normalized = v.replace("Z", "+00:00") if v.endswith("Z") else v
            datetime.fromisoformat(normalized)
        except Exception as exc:
            raise ValueError(
                f"timestamp_iso must be a valid ISO 8601 formatted string: {v}"
            ) from exc
        return v

    @field_validator("model_sha256")
    @classmethod
    def _validate_model_sha256(cls, v: str) -> str:
        return _validate_sha256_digest(v, "model_sha256")

    @field_validator("input_hash")
    @classmethod
    def _validate_input_hash(cls, v: str) -> str:
        return _validate_sha256_digest(v, "input_hash")

    @field_validator("previous_receipt_hash")
    @classmethod
    def _validate_previous_receipt_hash(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_sha256_digest(v, "previous_receipt_hash")
