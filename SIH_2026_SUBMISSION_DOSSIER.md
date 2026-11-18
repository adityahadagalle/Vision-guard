# VisionGuard — SIH 2026 Master Submission Dossier

**Project Identity:** VisionGuard — Offline-First AI Lifecycle Integrity & Hardware-Bound Security Assurance Platform for Computer-Vision Systems  
**Hackathon Track:** Smart India Hackathon (SIH) 2026 — AI Security, Trustworthy ML & Defense Systems  
**Status:** Certified Production-Ready Baseline (Phases 0–9 Complete)  
**Verification Baseline:** 625 Automated Security & Pipeline Tests Passed, 1 Skipped, 0 Failed  

---

## 1. Executive Summary

Computer-vision models deployed across sovereign, tactical, healthcare, and critical infrastructure environments face acute security, provenance, and data-integrity risks. In high-assurance deployments, reliance on continuous cloud connectivity or third-party validation APIs introduces attack vectors, data-exfiltration vulnerabilities, and single points of failure.

**VisionGuard** is an offline-first, cryptographic lifecycle assurance platform purpose-built to guarantee trust, data immutability, behavioral stability, and epistemic honesty across the four core phases of the computer-vision lifecycle without external network dependencies.

```
+----------------------------------------------------------------------------------------------------+
|                                    VISIONGUARD 4-PILLAR DEFENSE                                    |
+--------------------------+--------------------------+--------------------------+-------------------+
| 1. DATA INTEGRITY        | 2. MODEL INTEGRITY       | 3. INFERENCE INTEGRITY   | 4. SHIFT MONITOR  |
| Content-Addressed Store  | 11 Isolation Detectors   | Ed25519 Ledger Receipts  | Two-Sample KS &   |
| SHA-256 Inode Fencing    | Safe Checkpoint Loader   | Monotonic Hash Chaining  | Wasserstein Dist. |
| Ingestion Quarantine     | Conflict Resolution Eng. | Sequence Lock Guarantee  | ADR-012 Epistemic |
+--------------------------+--------------------------+--------------------------+-------------------+
                                           ▲
                                           │
+----------------------------------------------------------------------------------------------------+
|                       AIR-GAP COMPLIANCE & HARDWARE-BOUND ATTESTATION CORE                         |
|   Strict Offline Flags (HF/Transformers) • Non-Root Sandboxing (UID > 0) • MAC Hardware Binding   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. The Four Assurance Pillars

### Pillar 1: Data Integrity & Ingestion Hardening
- **Content-Addressed Storage (CAS):** Datasets and media archives are hashed using streaming chunk-buffered SHA-256 digests. Tamper attempts or single-bit mutations result in instant manifest mismatch and quarantine isolation.
- **Defensive Ingestion Limits:** Enforces strict limits against decompression bombs, path traversal attacks, and inode exhaustion ($250\text{ MB}$ max upload, $40\text{M}$ pixel decompression limits, streaming SHA-256 calculation).
- **Quarantine Fencing:** Malicious or malformed inputs are quarantined into isolated directory partitions with zero executable permissions.

### Pillar 2: Model Integrity & Backdoor Defense
- **Restricted Checkpoint Inspection:** PyTorch `.pt` / `.pth` deserialization is gated against arbitrary pickle gadget execution and memory corruptions.
- **11 Modular Forensic Detectors:** Analyzes architecture conformity, parameter bounds, weight divergence, NaN/Inf poisoning, and layer modifications.
- **Dual Backdoor Detectors:**
  - *Spectral Signatures (Tran et al., 2018):* SVD decomposition of feature representations flags latent clustering outliers.
  - *Activation Clustering (Chen et al., 2018):* ICA and silhouette scoring identify poisoned bimodal clusters in representation space.
- **Epistemological Conflict Engine:** Synthesizes higher-order findings (`BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER`) when behavioral stability appears clean but latent representations reveal stealth triggers.

### Pillar 3: Inference Integrity & Provenance Ledger
- **RFC 8785 Canonical Serialization:** Every edge inference payload is canonicalized with sorted keys and deterministic whitespace before hashing.
- **Ed25519 Cryptographic Receipts:** Edge devices digitally sign canonical receipt hashes using offline custody private keys (RFC 8032).
- **Continuous Monotonic Hash-Chaining:** Inferences form an append-only, tamper-evident cryptographic blockchain ledger:
  $$\text{ReceiptHash}_n = \mathcal{H}\Big(\text{Sequence}_n \,\|\, \text{ReceiptHash}_{n-1} \,\|\, \text{InputHash}_n \,\|\, \text{OutputHash}_n\Big)$$
- **Sequence Lock Invariant:** Replayed sequence numbers ($n \le \text{latest}$) or gaps ($n > \text{latest} + 1$) trigger immediate HTTP 409 Conflict rejection, preventing desynchronization.

### Pillar 4: Distribution Shift & OOD Drift Monitoring
- **Hypothesis Testing Without Hallucination:** Evaluates two-sample Kolmogorov-Smirnov (KS) statistic and Wasserstein distance metrics between baseline reference distributions and production telemetry.
- **ADR-012 Epistemic Honesty:** When sample density is insufficient ($N < 30$), the system strictly returns `CapabilityStatus.INCONCLUSIVE`. It refuses to guess statistical significance or report a false sense of clean passing health.

---

## 3. Air-Gap Architecture & Attestation Core

VisionGuard is architected from the bare metal up to operate without cloud telemetry, remote key-management, or external package repositories:

1. **Air-Gap Flags Enforcement:** Automated validators ensure `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `VISIONGUARD_AIRGAP=1` are exported in the environment.
2. **Non-Root Least Privilege:** The runtime drops administrative privileges (`UID > 0`), disabling kernel exploit propagation.
3. **Deterministic Hardware Binding:** Platform attestation binds execution identity to physical host attributes (non-multicast physical MAC address and CPU/host identifiers), producing immutable attestation certificates.

---

## 4. SIH 2026 Chaos Benchmark Engine Results

The Chaos Engine (`backend/tests/chaos/test_chaos_pipeline.py` & `scripts/run_chaos_benchmark.py`) subjects VisionGuard to extreme adversarial stress testing:

```
+--------------------------------------------------------------------------------------+
| SIH 2026 ADVERSARIAL THREAT VECTOR MATRIX & MITIGATION VERIFICATION                  |
+---+----------------------------+-----------------------------+-----------------------+
| # | Attack Vector              | Mitigation Mechanism        | Security Verdict      |
+---+----------------------------+-----------------------------+-----------------------+
| 1 | Poisoned Model Injection   | Dual Spectral Signatures &  | MITIGATED [QUARANTINE]|
|   | Stealth Latent Trigger     | Activation Clustering       | Conflict Triggered:   |
|   | (Chen / Tran et al.)       | + Epistemic Conflict Engine | SUSPECTED_POISON (4)  |
+---+----------------------------+-----------------------------+-----------------------+
| 2 | Telemetry Replay Attack    | Ed25519 Cryptographic Proof | MITIGATED [REJECTED]  |
|   | Sequence Spoofing          | & Monotonic Hash Linkage    | HTTP 409 Conflict:    |
|   | (Sequence gap / replay)    | + Sequence Lock Invariant   | REPLAY_DETECTED       |
+---+----------------------------+-----------------------------+-----------------------+
| 3 | Micro-Batch Hallucination  | ADR-012 Epistemic Gate      | MITIGATED [GATED]     |
|   | Small Sample Attack (N=29) | Density Cutoff (N >= 30)    | Epistemic Status:     |
|   | (Statistical hallucination)| Fail-Closed Defense         | INCONCLUSIVE (No Hal.)|
+---+----------------------------+-----------------------------+-----------------------+
| OVERALL RESILIENCE SCORE: 100% (Zero Hallucinations, Monotonic Ledger Preserved)     |
+--------------------------------------------------------------------------------------+
```

---

## 5. High-Agency Operator Dashboard

Phase 8 delivered an enterprise-grade, dark-mode operator frontend built with **React 18, TypeScript, and Vite**, communicating over loopback proxy `127.0.0.1:8000`:
- **Cryptographic Hash Chain Viewer:** Live sequence pipeline with interactive receipt verification, public key inspection, and broken-chain alarms.
- **Distribution Shift & OOD Monitor:** Real-time KS and Wasserstein metric displays with ADR-012 amber `INCONCLUSIVE` epistemic badges.
- **Dataset & Model Quarantine Inspector:** Forensic viewer inspecting quarantined weights, unapproved layers, and tamper triggers.
- **Air-Gap Attestation Panel:** Real-time host posture gauges for offline flags, process isolation, and physical hardware binding.
- **Fault-Tolerant Error Boundary:** Defensive isolation guarantees single-component faults never crash the operational console.

---

## 6. Verification & Audit Trail Summary

| Verification Suite | Command | Test Count | Result |
|---|---|---|---|
| Comprehensive Core Test Suite | `.venv\Scripts\python.exe -m pytest backend/tests -q` | 625 Tests | **625 Passed, 1 Skipped, 0 Failed** |
| Chaos Adversarial Benchmark | `.venv\Scripts\python.exe scripts/run_chaos_benchmark.py` | 3 Attacks | **3/3 Mitigated (100% Defense)** |
| Frontend Production Build | `npm --prefix frontend run build` | 1589 Modules | **Zero TS/Vite Errors (Exit 0)** |
| Static Analysis & Linter | `.venv\Scripts\ruff.exe check backend scripts` | All Source Files | **All Checks Passed (Exit 0)** |

---

*Certified by the VisionGuard Autonomous Security Engineering Team for SIH 2026 Submission.*
