"""Unit tests for Ed25519 cryptographic signer and offline key custody.

Validates RFC 8032 signature generation, bit-for-bit tamper rejection across all
payload attributes, key custody serialization, and signature verification failures.
"""

import base64
from pathlib import Path

import pytest

from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.signer import InferenceSigner

_SHA256_A = "a" * 64
_SHA256_B = "b" * 64
_SHA256_C = "c" * 64


def _create_sample_payload(
    edge_device_id: str = "edge-xavier-01",
    timestamp_iso: str = "2026-11-15T10:30:00Z",
    sequence_number: int = 0,
    model_sha256: str = _SHA256_A,
    input_hash: str = _SHA256_B,
    inference_output: dict | None = None,
    previous_receipt_hash: str | None = None,
) -> InferenceReceiptPayload:
    """Helper to construct an immutable InferenceReceiptPayload."""
    if inference_output is None:
        inference_output = {
            "predictions": [
                {"label": "pedestrian", "score": 0.98},
                {"label": "traffic_light", "score": 0.85},
            ],
            "fps": 30.5,
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


class TestEd25519KeyGeneration:
    """Tests Ed25519 keypair generation and format invariants."""

    def test_generate_keypair_returns_valid_pem(self) -> None:
        priv_pem, pub_pem = InferenceSigner.generate_keypair("edge-device-01")

        assert isinstance(priv_pem, bytes)
        assert isinstance(pub_pem, bytes)
        assert b"-----BEGIN PRIVATE KEY-----" in priv_pem
        assert b"-----END PRIVATE KEY-----" in priv_pem
        assert b"-----BEGIN PUBLIC KEY-----" in pub_pem
        assert b"-----END PUBLIC KEY-----" in pub_pem

    @pytest.mark.parametrize("invalid_id", ["", "   ", "\t\n"])
    def test_generate_keypair_rejects_empty_device_id(self, invalid_id: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            InferenceSigner.generate_keypair(invalid_id)
        assert "device_id must be a non-empty string" in str(exc_info.value)


class TestEd25519SigningAndVerification:
    """Tests signing and verification lifecycle on InferenceReceiptPayload."""

    def test_sign_and_verify_roundtrip_success(self) -> None:
        priv_pem, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()

        signature_b64 = InferenceSigner.sign_payload(payload, priv_pem)

        assert isinstance(signature_b64, str)
        # 64 raw bytes encoded in base64 is 88 characters (with padding) or 86 without
        raw_sig = base64.b64decode(signature_b64)
        assert len(raw_sig) == 64

        is_valid = InferenceSigner.verify_signature(payload, signature_b64, pub_pem)
        assert is_valid is True

    def test_deterministic_signing_rfc8032(self) -> None:
        """Ed25519 is deterministic under RFC 8032; signing identical payload yields identical sig."""
        priv_pem, _ = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()

        sig1 = InferenceSigner.sign_payload(payload, priv_pem)
        sig2 = InferenceSigner.sign_payload(payload, priv_pem)
        assert sig1 == sig2

    def test_chained_payload_with_previous_receipt_hash(self) -> None:
        priv_pem, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload(
            sequence_number=42,
            previous_receipt_hash=_SHA256_C,
        )

        signature_b64 = InferenceSigner.sign_payload(payload, priv_pem)
        assert InferenceSigner.verify_signature(payload, signature_b64, pub_pem) is True


class TestTamperDetection:
    """Verifies that altering ANY attribute in the payload invalidates the signature."""

    @pytest.fixture
    def signed_setup(self) -> tuple[InferenceReceiptPayload, str, bytes]:
        priv_pem, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload(
            sequence_number=1,
            previous_receipt_hash=_SHA256_C,
        )
        signature_b64 = InferenceSigner.sign_payload(payload, priv_pem)
        return payload, signature_b64, pub_pem

    def test_tamper_sequence_number(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            sequence_number=2,  # Modified sequence number
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_edge_device_id(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            edge_device_id="edge-xavier-02",  # Modified device ID
            sequence_number=payload.sequence_number,
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_timestamp_iso(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            timestamp_iso="2026-11-15T10:30:01Z",  # 1-second drift
            sequence_number=payload.sequence_number,
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_model_sha256(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            model_sha256="d" * 64,  # Modified model digest
            sequence_number=payload.sequence_number,
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_input_hash(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            input_hash="e" * 64,  # Modified input digest
            sequence_number=payload.sequence_number,
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_inference_output(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            sequence_number=payload.sequence_number,
            previous_receipt_hash=payload.previous_receipt_hash,
            inference_output={"predictions": [{"label": "pedestrian", "score": 0.50}]},
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False

    def test_tamper_previous_receipt_hash(
        self, signed_setup: tuple[InferenceReceiptPayload, str, bytes]
    ) -> None:
        payload, signature_b64, pub_pem = signed_setup
        tampered = _create_sample_payload(
            sequence_number=payload.sequence_number,
            previous_receipt_hash="f" * 64,  # Modified prior hash
            inference_output=payload.inference_output,
        )
        assert InferenceSigner.verify_signature(tampered, signature_b64, pub_pem) is False


class TestSignatureVerificationFailures:
    """Tests robustness against invalid, corrupt, or forged signatures and keys."""

    def test_mismatched_public_key_rejects(self) -> None:
        priv_pem_a, _ = InferenceSigner.generate_keypair("device-a")
        _, pub_pem_b = InferenceSigner.generate_keypair("device-b")
        payload = _create_sample_payload()

        signature_b64 = InferenceSigner.sign_payload(payload, priv_pem_a)
        # Attempt to verify with device B's public key
        assert InferenceSigner.verify_signature(payload, signature_b64, pub_pem_b) is False

    def test_corrupted_signature_bytes_rejects(self) -> None:
        priv_pem, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()
        signature_b64 = InferenceSigner.sign_payload(payload, priv_pem)

        # Corrupt one character in the base64 string
        corrupted_b64 = "A" + signature_b64[1:]
        assert InferenceSigner.verify_signature(payload, corrupted_b64, pub_pem) is False

    def test_malformed_base64_signature_returns_false(self) -> None:
        _, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()
        assert InferenceSigner.verify_signature(payload, "not-valid-base64!@#$", pub_pem) is False

    def test_invalid_signature_length_returns_false(self) -> None:
        _, pub_pem = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()
        short_sig_b64 = base64.b64encode(b"too_short").decode("ascii")
        assert InferenceSigner.verify_signature(payload, short_sig_b64, pub_pem) is False

    def test_invalid_public_key_pem_returns_false(self) -> None:
        priv_pem, _ = InferenceSigner.generate_keypair("device-01")
        payload = _create_sample_payload()
        sig_b64 = InferenceSigner.sign_payload(payload, priv_pem)
        assert InferenceSigner.verify_signature(payload, sig_b64, b"not_a_pem_key") is False

    def test_invalid_private_key_pem_raises_value_error(self) -> None:
        payload = _create_sample_payload()
        with pytest.raises(ValueError) as exc_info:
            InferenceSigner.sign_payload(payload, b"invalid_pem_data")
        assert "Failed to load Ed25519 private key PEM" in str(exc_info.value)


class TestKeyCustodyFilesystemOperations:
    """Tests saving and loading Ed25519 keypairs to/from local disk storage."""

    def test_save_and_load_keypair_roundtrip(self, tmp_path: Path) -> None:
        signer = InferenceSigner(keys_dir=tmp_path)
        priv_pem, pub_pem = signer.generate_keypair("node-alpha")

        priv_path, pub_path = signer.save_keypair("node-alpha", priv_pem, pub_pem)
        assert priv_path.exists()
        assert pub_path.exists()
        assert priv_path.name == "node-alpha.key"
        assert pub_path.name == "node-alpha.pub"

        loaded_priv, loaded_pub = signer.load_keypair("node-alpha")
        assert loaded_priv == priv_pem
        assert loaded_pub == pub_pem

        loaded_pub_only = signer.load_public_key("node-alpha")
        assert loaded_pub_only == pub_pem

    def test_load_nonexistent_key_raises_file_not_found(self, tmp_path: Path) -> None:
        signer = InferenceSigner(keys_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            signer.load_keypair("missing-device")

        with pytest.raises(FileNotFoundError):
            signer.load_public_key("missing-device")
