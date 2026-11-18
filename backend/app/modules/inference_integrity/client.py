"""Lightweight Edge Runtime Client for VisionGuard Inference Integrity.

Designed for resource-constrained edge hardware (e.g., Jetson Nano, Raspberry Pi).
Maintains local monotonic sequence and hash-chaining state, signs inference receipts
locally using Ed25519 offline custody keys, and transmits receipts to the backend
ingestion REST API.

Enforces strict sequence locking: if transmission or server validation fails,
local sequence counters are NEVER advanced, preventing catastrophic state desync.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from backend.app.domain.entities.inference import InferenceReceiptPayload
from backend.app.modules.inference_integrity.canonicalizer import compute_payload_hash
from backend.app.modules.inference_integrity.signer import InferenceSigner


class EdgeClientError(Exception):
    """Base exception for all edge client errors."""

    pass


class EdgeClientKeyError(EdgeClientError):
    """Raised when key loading, parsing, or file access fails."""

    pass


class EdgeClientTransmissionError(EdgeClientError):
    """Raised when receipt transmission fails or server rejects the payload."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class VisionGuardEdgeClient:
    """Lightweight edge client managing local hash-chain state and telemetry provenance.

    Attributes:
        device_id: Unique hardware identifier for this edge device.
        api_url: Base URL of the VisionGuard ingestion API (e.g., http://cloud:8000/api/v1).
        sequence_number: Monotonically incrementing sequence counter (starts at 0 for genesis).
        previous_receipt_hash: SHA-256 digest of previous receipt payload (None for genesis).
    """

    def __init__(
        self,
        device_id: str,
        api_url: str,
        *,
        private_key_path: str | Path | None = None,
        public_key_path: str | Path | None = None,
        private_key_pem: bytes | str | None = None,
        public_key_pem: bytes | str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not device_id or not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string.")
        if not api_url or not isinstance(api_url, str) or not api_url.strip():
            raise ValueError("api_url must be a non-empty string.")

        self.device_id = device_id.strip()
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

        # Local Monotonic State
        self.sequence_number: int = 0
        self.previous_receipt_hash: str | None = None

        # Load Keys
        self._private_key_bytes = self._resolve_private_key(private_key_path, private_key_pem)
        self._public_key_pem_str = self._resolve_public_key(public_key_path, public_key_pem)

        # HTTP Client management
        self._custom_client = http_client is not None
        self._client: httpx.Client = (
            http_client if http_client is not None else httpx.Client(timeout=self.timeout)
        )

    def _resolve_private_key(
        self,
        key_path: str | Path | None,
        key_pem: bytes | str | None,
    ) -> bytes:
        """Resolves private key bytes from either a file path or in-memory string/bytes."""
        if key_pem is not None:
            if isinstance(key_pem, str):
                return key_pem.encode("utf-8")
            if isinstance(key_pem, bytes):
                return key_pem
            raise EdgeClientKeyError("private_key_pem must be bytes or str.")

        if key_path is not None:
            p = Path(key_path)
            if not p.exists() or not p.is_file():
                raise EdgeClientKeyError(f"Private key file not found: {key_path}")
            try:
                return p.read_bytes()
            except OSError as exc:
                raise EdgeClientKeyError(f"Failed to read private key file: {exc}") from exc

        raise EdgeClientKeyError("Either private_key_path or private_key_pem must be provided.")

    def _resolve_public_key(
        self,
        key_path: str | Path | None,
        key_pem: bytes | str | None,
    ) -> str:
        """Resolves public key PEM string from either a file path or in-memory string/bytes."""
        if key_pem is not None:
            if isinstance(key_pem, bytes):
                return key_pem.decode("utf-8")
            if isinstance(key_pem, str):
                return key_pem
            raise EdgeClientKeyError("public_key_pem must be bytes or str.")

        if key_path is not None:
            p = Path(key_path)
            if not p.exists() or not p.is_file():
                raise EdgeClientKeyError(f"Public key file not found: {key_path}")
            try:
                return p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise EdgeClientKeyError(f"Failed to read public key file: {exc}") from exc

        raise EdgeClientKeyError("Either public_key_path or public_key_pem must be provided.")

    def record_inference(
        self,
        model_sha256: str,
        input_hash: str,
        inference_output: dict[str, Any],
        *,
        timestamp_iso: str | None = None,
    ) -> dict[str, Any]:
        """Constructs, signs, and ingests an inference receipt to the cloud API.

        Execution steps:
        1. Construct the canonical InferenceReceiptPayload using local sequence & hash state.
        2. Compute the RFC 8785 SHA-256 payload hash.
        3. Locally sign the canonical payload with the Ed25519 private key.
        4. Transmit payload, signature, and public key to POST /inference/receipts.
        5. If HTTP 201 Created:
           - Advance local sequence_number (n = n + 1)
           - Update previous_receipt_hash to the current payload hash
           - Return the server response payload.
        6. If the server rejects or fails (HTTP 4xx/5xx or network error):
           - DO NOT advance sequence_number or previous_receipt_hash.
           - Raise EdgeClientTransmissionError immediately.

        Args:
            model_sha256: 64-character hex digest of active model weights.
            input_hash: 64-character hex digest of the input image/sensor payload.
            inference_output: Model prediction dictionary (logits, boxes, classifications).
            timestamp_iso: Optional ISO 8601 timestamp string (generated if omitted).

        Returns:
            Dictionary response from the ingestion API.

        Raises:
            EdgeClientTransmissionError: On HTTP error status or network failure.
        """
        ts = timestamp_iso or datetime.now(UTC).isoformat()

        # Step 1: Construct domain payload
        payload = InferenceReceiptPayload(
            edge_device_id=self.device_id,
            timestamp_iso=ts,
            sequence_number=self.sequence_number,
            model_sha256=model_sha256,
            input_hash=input_hash,
            inference_output=inference_output,
            previous_receipt_hash=self.previous_receipt_hash,
        )

        # Step 2: Compute payload hash
        payload_hash = compute_payload_hash(payload)

        # Step 3: Local Ed25519 digital signature
        signature_b64 = InferenceSigner.sign_payload(payload, self._private_key_bytes)

        # Step 4: Prepare wire request
        request_body = {
            "payload": payload.model_dump(mode="json"),
            "signature": signature_b64,
            "public_key_pem": self._public_key_pem_str,
        }

        # Step 5: Transmit to ingestion endpoint
        endpoint = f"{self.api_url}/inference/receipts"
        try:
            response = self._client.post(endpoint, json=request_body)
        except httpx.RequestError as exc:
            # STOP INVARIANT: Do NOT advance sequence on transmission failure
            raise EdgeClientTransmissionError(
                f"Network failure transmitting inference receipt to {endpoint}: {exc}",
                status_code=None,
                response_data=None,
            ) from exc

        # Step 6: Evaluate response
        if response.status_code == 201:
            # Success: Atomically advance local hash-chain state
            self.sequence_number += 1
            self.previous_receipt_hash = payload_hash
            try:
                return response.json()
            except Exception:
                return {"status": "ACCEPTED", "payload_hash": payload_hash}

        # Failure: Raise error and keep sequence state locked
        try:
            err_data = response.json()
        except Exception:
            err_data = response.text

        raise EdgeClientTransmissionError(
            f"Ingestion API rejected receipt: HTTP {response.status_code} - {err_data}",
            status_code=response.status_code,
            response_data=err_data,
        )

    @property
    def state(self) -> dict[str, Any]:
        """Returns the current local hash-chain state."""
        return {
            "device_id": self.device_id,
            "sequence_number": self.sequence_number,
            "previous_receipt_hash": self.previous_receipt_hash,
        }

    def close(self) -> None:
        """Closes the underlying HTTP client if managed by this instance."""
        if not self._custom_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> VisionGuardEdgeClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
