# Module Specification: Inference Integrity Engine

**Document Version:** 1.0.0  
**Status:** Architectural Specification  
**Date:** November 2026  
**Scope:** Cryptographic inference receipts, Ed25519 signing, append-only SHA-256 hash chaining, replay enforcement, and audit verification.

---

## 1. Goal & Mission

The Inference Integrity Engine creates verifiable, tamper-evident cryptographic receipts for computer-vision inference predictions. It binds input image digests, model digests, inference configuration digests, output predictions, and sequence metadata into a canonically serialized payload signed with Ed25519 and chained via SHA-256 links.

---

## 2. Core Cryptographic Concepts & Operations

```text
Inference Event
       │
       ├── Step 1: Compute Input SHA-256 Digest (Raw Image Bytes)
       ├── Step 2: Compute Model SHA-256 Digest (Active Model Checkpoint)
       ├── Step 3: Compute Configuration Digest (Thresholds, Preprocessing)
       ├── Step 4: Construct Versioned Receipt Payload
       ├── Step 5: Canonical Serialization (RFC 8785 JCS)
       ├── Step 6: Ed25519 Digital Signature (Curve25519)
       └── Step 7: Hash Link to Prior Receipt: H_curr = SHA256(Payload || H_prev)
```

---

## 3. Cryptographic Verification & Audit Engine

During audit inspection, the engine traverses the stored sequence of inference receipts and evaluates:
1. **Signature Verification:** Re-serializes payload using canonical rules and verifies signature against the trusted Ed25519 public key.
2. **Chain Continuity:** Validates that `previous_receipt_hash` matches the SHA-256 digest of the immediately preceding receipt:
   $$\text{Chain Check: } H_{n-1} = \text{receipt}[n].\text{previous\_receipt\_hash}$$
3. **Monotonic Sequence Enforcement:** Validates that `sequence_id` increments by exactly 1 without gaps or duplicates.
4. **Replay Enforcement:** Verifies that `event_id` (UUIDv4) has not been previously registered.
5. **Periodic Checkpoints:** Compares the tip of the local chain against independently exported, signed checkpoints.

---

## 4. Key Management & Custody Policy

- **Offline Local Key Pair:** Generated using `cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey`.
- **Storage Protection:** Stored locally under `data/keys/inference_signer.key` with strict filesystem permissions (`chmod 600`).
- **No In-Memory Leaks:** Private key bytes are never returned via REST endpoints, logged in audit files, or committed to version control.
- **Limitation Note:** If the host OS is completely compromised by root malware, local signatures can be forged. Documented recommendation: Export periodic chain checkpoints to write-once-read-many (WORM) storage or an external audit host.

---

## 5. Evaluation Protocol & Metrics

- **Deterministic Tamper Tests:** 100% detection of single-bit field modifications, row deletions, sequence reordering, and replayed event IDs.
- **Signing Performance:** Throughput $> 1,000$ receipts/second on CPU; verification latency $< 0.8$ ms per receipt.
