"""RFC 8785 JSON Canonicalization Scheme (JCS) and cryptographic hashing.

Provides deterministic, bit-for-bit reproducible canonical serialization of
InferenceReceiptPayload instances. Guarantees key sorting, whitespace elimination,
and strict UTF-8 byte stream generation for Ed25519 signature computation and
SHA-256 hash chaining.
"""

import hashlib
import json
from typing import Any

from backend.app.domain.entities.inference import InferenceReceiptPayload


def canonicalize_payload(payload: InferenceReceiptPayload | dict[str, Any]) -> bytes:
    """Serializes an inference receipt payload into canonical JSON bytes per RFC 8785.

    Rules enforced:
    1. Keys at all hierarchy levels are sorted lexicographically by Unicode code point.
    2. Whitespace between separators is completely eliminated (separators=(',', ':')).
    3. Strings are encoded as unescaped UTF-8 without byte-order marks (BOM).
    4. Deterministic scalar and null representations.

    Args:
        payload: An InferenceReceiptPayload entity or dictionary conforming to the schema.

    Returns:
        Exact bytes representing the canonicalized JSON serialization.

    Raises:
        TypeError: If the input object is not an InferenceReceiptPayload or dict.
        ValueError: If the payload contains values that cannot be serialized to JSON.
    """
    if isinstance(payload, InferenceReceiptPayload):
        raw_data = payload.model_dump(mode="json")
    elif isinstance(payload, dict):
        raw_data = payload
    else:
        raise TypeError(f"Expected InferenceReceiptPayload or dict, got {type(payload).__name__}")

    try:
        canonical_str = json.dumps(
            raw_data,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Inference receipt payload cannot be serialized to JSON: {exc}") from exc

    return canonical_str.encode("utf-8")


def compute_payload_hash(payload: InferenceReceiptPayload | dict[str, Any]) -> str:
    """Computes the SHA-256 hexadecimal digest of the RFC 8785 canonicalized payload.

    Args:
        payload: An InferenceReceiptPayload entity or dictionary.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest string.
    """
    canonical_bytes = canonicalize_payload(payload)
    return hashlib.sha256(canonical_bytes).hexdigest()


__all__ = [
    "canonicalize_payload",
    "compute_payload_hash",
]
