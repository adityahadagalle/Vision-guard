"""Unit tests for inference receipt entity validation and RFC 8785 JCS canonicalization.

Verifies bit-for-bit serialization determinism, key sorting invariance across
nested dictionary structures, strict Pydantic validation boundaries, and
SHA-256 cryptographic digest computation.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import (
    canonicalize_payload,
    compute_payload_hash,
)

_VALID_SHA256_A = "a" * 64
_VALID_SHA256_B = "b" * 64
_VALID_SHA256_C = "c" * 64


def _create_sample_payload(
    edge_device_id: str = "edge-jetson-xavier-01",
    timestamp_iso: str = "2026-11-15T10:30:00Z",
    sequence_number: int = 0,
    model_sha256: str = _VALID_SHA256_A,
    input_hash: str = _VALID_SHA256_B,
    inference_output: dict[str, Any] | None = None,
    previous_receipt_hash: str | None = None,
) -> InferenceReceiptPayload:
    """Helper to build a valid InferenceReceiptPayload for testing."""
    if inference_output is None:
        inference_output = {
            "predictions": [
                {"class": "pedestrian", "confidence": 0.945, "box": [12, 45, 100, 220]},
                {"class": "vehicle", "confidence": 0.812, "box": [150, 80, 400, 300]},
            ],
            "metadata": {"frame_id": 1042, "inference_time_ms": 14.2},
        }
    return InferenceReceiptPayload(
        edge_device_id=edge_device_id,
        timestamp_iso=timestamp_iso,
        sequence_number=sequence_number,
        model_sha256=model_sha256,
        input_hash=input_hash,
        inference_output=inference_output,
        previous_receipt_hash=previous_receipt_hash,
    )


class TestInferenceReceiptPayloadValidation:
    """Tests Pydantic validation invariants on InferenceReceiptPayload."""

    def test_valid_payload_creation(self) -> None:
        payload = _create_sample_payload()
        assert payload.edge_device_id == "edge-jetson-xavier-01"
        assert payload.sequence_number == 0
        assert payload.model_sha256 == _VALID_SHA256_A
        assert payload.input_hash == _VALID_SHA256_B
        assert payload.previous_receipt_hash is None

    def test_uppercase_sha256_normalized_to_lowercase(self) -> None:
        upper_sha = "A" * 64
        payload = _create_sample_payload(
            model_sha256=upper_sha,
            input_hash=upper_sha,
            previous_receipt_hash=upper_sha,
        )
        assert payload.model_sha256 == _VALID_SHA256_A
        assert payload.input_hash == _VALID_SHA256_A
        assert payload.previous_receipt_hash == _VALID_SHA256_A

    @pytest.mark.parametrize(
        "invalid_hash",
        [
            "",
            "abc",
            "a" * 63,
            "a" * 65,
            "g" * 64,  # 'g' is not hex
            "zz" + "a" * 62,
            12345,
            None,
        ],
    )
    def test_reject_invalid_model_sha256(self, invalid_hash: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(model_sha256=invalid_hash)
        assert "model_sha256" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_hash",
        [
            "",
            "12345",
            "f" * 63,
            "f" * 65,
            "x" * 64,
        ],
    )
    def test_reject_invalid_input_hash(self, invalid_hash: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(input_hash=invalid_hash)
        assert "input_hash" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_hash",
        [
            "invalid_hash",
            "e" * 63,
            "e" * 65,
            "not_hex" * 8,
        ],
    )
    def test_reject_invalid_previous_receipt_hash(self, invalid_hash: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(previous_receipt_hash=invalid_hash)
        assert "previous_receipt_hash" in str(exc_info.value)

    def test_accept_null_or_valid_previous_receipt_hash(self) -> None:
        # Genesis receipt has null previous_receipt_hash
        payload_genesis = _create_sample_payload(previous_receipt_hash=None)
        assert payload_genesis.previous_receipt_hash is None

        # Chained receipt has valid 64-hex string
        payload_chained = _create_sample_payload(previous_receipt_hash=_VALID_SHA256_C)
        assert payload_chained.previous_receipt_hash == _VALID_SHA256_C

    def test_sequence_number_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(sequence_number=-1)
        assert "sequence_number" in str(exc_info.value)

        # 0 (genesis) and positive integers are valid
        payload_zero = _create_sample_payload(sequence_number=0)
        assert payload_zero.sequence_number == 0

        payload_large = _create_sample_payload(sequence_number=100_000)
        assert payload_large.sequence_number == 100_000

    @pytest.mark.parametrize(
        "invalid_ts",
        [
            "",
            "   ",
            "not-a-timestamp",
            "2026/11/15 10:30:00",
            "yesterday",
        ],
    )
    def test_reject_invalid_timestamp_iso(self, invalid_ts: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(timestamp_iso=invalid_ts)
        assert "timestamp_iso" in str(exc_info.value)

    @pytest.mark.parametrize(
        "valid_ts",
        [
            "2026-11-15T10:30:00Z",
            "2026-11-15T10:30:00+00:00",
            "2026-11-15T10:30:00.123456+05:30",
        ],
    )
    def test_accept_valid_timestamp_iso(self, valid_ts: str) -> None:
        payload = _create_sample_payload(timestamp_iso=valid_ts)
        assert payload.timestamp_iso == valid_ts

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t\n"])
    def test_reject_empty_edge_device_id(self, empty_id: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _create_sample_payload(edge_device_id=empty_id)
        assert "edge_device_id" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            InferenceReceiptPayload(
                edge_device_id="edge-01",
                timestamp_iso="2026-11-15T10:30:00Z",
                sequence_number=1,
                model_sha256=_VALID_SHA256_A,
                input_hash=_VALID_SHA256_B,
                inference_output={"class": "test"},
                unauthorized_extra_field="tampering_attempt",
            )
        assert "extra" in str(exc_info.value).lower()

    def test_payload_is_frozen_immutable(self) -> None:
        payload = _create_sample_payload()
        with pytest.raises(ValidationError):
            payload.sequence_number = 99  # type: ignore[misc]


class TestRFC8785CanonicalizationAndHashing:
    """Tests RFC 8785 JCS canonicalization and SHA-256 cryptographic determinism."""

    def test_key_ordering_invariance(self) -> None:
        """Paylists with identical content but reversed key ordering produce identical bytes and hashes."""
        # Payload 1: keys in one order
        output1 = {
            "zebra": 1,
            "apple": 2,
            "nested": {"z": 100, "a": 200},
        }
        # Payload 2: keys in reversed order
        output2 = {
            "apple": 2,
            "nested": {"a": 200, "z": 100},
            "zebra": 1,
        }

        p1 = _create_sample_payload(inference_output=output1)
        p2 = _create_sample_payload(inference_output=output2)

        bytes1 = canonicalize_payload(p1)
        bytes2 = canonicalize_payload(p2)

        assert bytes1 == bytes2
        assert compute_payload_hash(p1) == compute_payload_hash(p2)

    def test_whitespace_elimination_and_strict_sorting(self) -> None:
        """Ensures zero whitespace around separators and strict lexicographical key ordering."""
        payload = _create_sample_payload(
            edge_device_id="dev-1",
            timestamp_iso="2026-11-15T10:30:00Z",
            sequence_number=0,
            inference_output={"b": 2, "a": 1},
        )
        canonical_bytes = canonicalize_payload(payload)
        canonical_text = canonical_bytes.decode("utf-8")

        # Whitespace invariant: no spaces after colons or commas
        assert ": " not in canonical_text
        assert ", " not in canonical_text
        assert "\n" not in canonical_text
        assert "\r" not in canonical_text
        assert "\t" not in canonical_text

        # Top-level key order must be strictly sorted
        keys = [
            "edge_device_id",
            "inference_output",
            "input_hash",
            "model_sha256",
            "previous_receipt_hash",
            "sequence_number",
            "timestamp_iso",
        ]
        # Check positions appear in order
        positions = [canonical_text.index(f'"{k}":') for k in keys]
        assert positions == sorted(positions)

    def test_utf8_encoding_preserves_characters_without_escape(self) -> None:
        """RFC 8785 requires characters > 0x7F to be encoded directly as UTF-8, not \\uXXXX."""
        unicode_output = {
            "detection": "車両 (vehicle)",
            "description": "Schöne Grüße aus München",
            "emoji": "🛡️",
        }
        payload = _create_sample_payload(inference_output=unicode_output)
        canonical_bytes = canonicalize_payload(payload)

        # Must not contain escaped unicode
        assert b"\\u" not in canonical_bytes

        # Must contain exact UTF-8 byte representations
        assert "車両 (vehicle)".encode() in canonical_bytes
        assert "Schöne Grüße aus München".encode() in canonical_bytes
        assert "🛡️".encode() in canonical_bytes

    def test_genesis_receipt_null_previous_hash_serialization(self) -> None:
        payload = _create_sample_payload(previous_receipt_hash=None)
        canonical_text = canonicalize_payload(payload).decode("utf-8")

        assert '"previous_receipt_hash":null' in canonical_text
        digest = compute_payload_hash(payload)
        assert len(digest) == 64
        assert int(digest, 16) > 0

    def test_compute_payload_hash_determinism_across_multiple_invocations(self) -> None:
        payload = _create_sample_payload()
        first_hash = compute_payload_hash(payload)

        # Hash must be strictly identical across 50 iterations
        for _ in range(50):
            assert compute_payload_hash(payload) == first_hash

    def test_canonicalize_accepts_dict_input(self) -> None:
        raw_dict = {
            "edge_device_id": "edge-01",
            "timestamp_iso": "2026-11-15T10:30:00Z",
            "sequence_number": 0,
            "model_sha256": _VALID_SHA256_A,
            "input_hash": _VALID_SHA256_B,
            "inference_output": {"cat": 0.99},
            "previous_receipt_hash": None,
        }
        bytes_from_dict = canonicalize_payload(raw_dict)
        hash_from_dict = compute_payload_hash(raw_dict)

        payload = InferenceReceiptPayload(**raw_dict)
        assert bytes_from_dict == canonicalize_payload(payload)
        assert hash_from_dict == compute_payload_hash(payload)

    def test_reject_non_serializable_values(self) -> None:
        class UnserializableObject:
            pass

        bad_dict = {"unserializable": UnserializableObject()}
        with pytest.raises(ValueError) as exc_info:
            canonicalize_payload(bad_dict)
        assert "cannot be serialized to JSON" in str(exc_info.value)

    def test_reject_invalid_payload_type(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            canonicalize_payload(["not", "a", "dict"])  # type: ignore[arg-type]
        assert "Expected InferenceReceiptPayload or dict" in str(exc_info.value)
