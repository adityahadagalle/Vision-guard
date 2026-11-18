# Research Gaps, Hypotheses & Engineering vs. Research Novelty

**Document Version:** 1.1.0 (Audited)  
**Status:** Canonical Research Gap Analysis & Formal Hypotheses  
**Date:** November 2026  
**Context:** Identifying genuine whitespace in Computer Vision Lifecycle Integrity, Adversarial Robustness, and Data Provenance while rigorously distinguishing engineering synthesis from research novelty.

---

## 1. Disciplinary Demarcation: Engineering Synthesis vs. Research Novelty

> **CRITICAL SCIENTIFIC PRINCIPLE:**  
> **Integrating established algorithms into a unified platform is an ENGINEERING ACHIEVEMENT, not an academic research discovery or patentable invention.**  
> Combining pHash, ResNet embeddings, Spectral Signatures, and Ed25519 into a single FastAPI service provides immense architectural and operational value, but it does not constitute novel computer science theory. True research contributions must be formulated as testable scientific hypotheses evaluated against published baselines.

---

## 2. Rigorous Evaluation of Proposed Research Hypotheses

### Hypothesis A: Leave-One-Contributor-Out (LOCO) Empirical Bayes Contributor Risk Aggregation
- **Category:** **Primary Scientific / Algorithmic Hypothesis**.
- **Formal Statement:** *Aggregating sample-level anomaly and poisoning signals at the contributor/batch level using Leave-One-Contributor-Out (LOCO) Empirical Bayes Beta-Binomial shrinkage improves contaminated batch triage efficiency (higher batch-level AUPRC) compared to naive thresholding, while achieving an empirical False Attribution Rate $< 5\%$ on low-volume clean contributors.*
- **Existing Prior Art & Published Approaches:**
  - Empirical Bayes shrinkage was formulated by Robbins (1956) and expanded extensively in biostatistics (Efron, 2010).
  - Beta-Binomial hierarchical modeling is widely used in click-through rate (CTR) prediction, credit default scoring, and hospital mortality ranking.
  - In ML security, sample-level poisoning defenses (Tran et al., 2018; Chen et al., 2018) treat datasets as flat collections of $(x, y)$ tuples without modeling provenance.
- **What VisionGuard Actually Adds:**
  - Adapts LOCO shrinkage specifically to computer-vision sample anomaly aggregation across annotator queues and sensor batches.
  - Eliminates the self-influence of large contributors on the estimated global prior rate.
  - Couples the continuous posterior risk score with explicit credible intervals to represent small-sample uncertainty.
- **Evaluation Protocol & Metrics:**
  - Evaluate on synthetic and real multi-contributor benchmarks (CIFAR-10 / Pascal VOC partitioned across 20 synthetic contributors).
  - Metric: Batch-level AUPRC and False Attribution Rate (FAR) across variable sample volumes ($N_i \in [5, 2000]$).
- **Baseline It Must Outperform:**
  - Baseline 1: Naive empirical anomaly ratio ($\hat{p}_i = k_i / N_i$).
  - Baseline 2: Global-prior Empirical Bayes without LOCO adjustment.
- **Why It Might Fail (Potential Blind Spots & Counter-Strategies):**
  - **Sybil / Camouflage Attacks:** An adversary aware of the model distributes poisoned samples across hundreds of generated low-volume contributor IDs ($N_i \le 3$), staying within the prior shrinkage window.
  - **Class-Assignment Confounding:** If a contributor is assigned exclusively difficult, high-variance classes (e.g., night-time rainy scenes), their natural anomaly rate is elevated, triggering false attribution.

---

### Hypothesis B: Capability-Aware Multi-State Assurance Modeling
- **Category:** **System Architecture & Engineering Innovation** (Not an academic discovery).
- **Formal Statement:** *An explicit multi-state execution and capability model (`available`, `unavailable`, `failed`, `not_run`) eliminates false "all-clear" security interpretations in incomplete computer vision inspection pipelines compared to conventional binary pass/fail reporting.*
- **Existing Prior Art & Published Approaches:**
  - Hardware and safety systems (AUTOSAR, ISO 26262, POSIX errno) use explicit capability and degradation state modeling.
  - Security vulnerability scanners (Nessus, OpenVAS) record plugin execution states and skipped checks.
  - In ML tooling (Cleanlab, ART), missing prerequisites typically trigger runtime exceptions (`KeyError`, `ValueError`) or are silently omitted from reports.
- **What VisionGuard Actually Adds:**
  - Standardizes capability status across heterogeneous ML integrity layers (data, model, inference, shift).
  - Enforces mandatory reason strings for `unavailable` and `failed` checks, preventing unexecuted checks from presenting a clean bill of health.
- **Evaluation Protocol & Metrics:** Scenario-based usability evaluation across 10 asset-combination test suites; measurement of user misinterpretation rates.
- **Baseline It Must Outperform:** Conventional binary pass/fail or raw exception crashing.
- **Why It Might Fail:** If status messages are verbose or poorly categorized, operators may experience alert fatigue.

---

### Hypothesis C: Cross-Layer Correlative Verification
- **Category:** **Applied Research Hypothesis** (Candidate for Phase 4 / Phase 6).
- **Formal Statement:** *Correlating data-integrity findings (near-duplicate clusters, spectral anomalies) with model behavioral fingerprints provides measurable signal separation between deliberate backdoor injection and benign data collection artifacts.*
- **Existing Prior Art & Published Approaches:**
  - Multimodal event correlation in Security Information and Event Management (SIEM) systems (Splunk, Zeek).
  - Academic backdoor research typically evaluates data defenses (Tran et al., 2018) or model defenses (Wang et al., 2019) in isolation.
- **What VisionGuard Actually Adds:**
  - Bridges the data-model lifecycle boundary: evaluates whether high contributor anomaly risk correlates with specific trigger activations in model fingerprint suites.
- **Evaluation Protocol & Metrics:** Area Under ROC (AUROC) in separating deliberate Trojan models from clean models trained on naturally noisy data.
- **Baseline It Must Outperform:** Independent single-layer detectors operating in isolation.
- **Why It Might Fail:** Weak or noisy correlations can lead to false causal conclusions if statistical coupling is not rigorously calibrated.

---

### Hypothesis D: Offline Verifiable Assurance Manifests
- **Category:** **Applied Cryptographic Synthesis** (Engineering Innovation).
- **Formal Statement:** *A standalone, canonically serialized JSON manifest binding input SHA-256 digests, detector configurations, software versions, and Ed25519 digital signatures allows third-party verification of assessment authenticity and reproducibility without network access or database connection.*
- **Existing Prior Art & Published Approaches:**
  - The Update Framework (TUF), Sigstore / Cosign, in-toto software supply chain attestations.
  - RFC 8785 (JSON Canonicalization Scheme - JCS) and RFC 8032 (Ed25519).
- **What VisionGuard Actually Adds:**
  - Tailors supply-chain attestation standards directly to computer vision datasets, PyTorch checkpoints, and inference audit receipts in an air-gapped environment.
- **Evaluation Protocol & Metrics:** Third-party verification latency ($< 5$ ms), deterministic signature validation across platforms, 100% tamper detection on mutated manifests.
- **Baseline It Must Outperform:** Unsigned database dumps and unauthenticated JSON/CSV logs.
- **Why It Might Fail:** Loss or compromise of local private keys undermines verification authenticity.

---

## 3. Clear Summary: Engineering vs. Scientific Novelty

| Innovation Item | Classification | Justification |
|---|---|---|
| **9-Stage Unified Integrity Pipeline** | **Engineering Synthesis** | Coherent, modular integration of established computer-vision and security techniques. |
| **LOCO Contributor Risk Shrinkage** | **Primary Research Hypothesis** | Novel algorithmic formulation of leave-one-out Bayesian shrinkage for dataset contributor attribution. |
| **Capability-Aware State Machine** | **Engineering Architecture** | Structured software design pattern eliminating silent failures and false "all-clear" reports. |
| **Cross-Layer Integrity Correlation** | **Applied Research Hypothesis** | Investigates empirical signal separation combining data-level and model-level anomaly indicators. |
| **Signed Offline Inference Receipts** | **Applied Cryptographic Engineering** | Implementation of RFC 8785 and Ed25519 for PyTorch vision audit trails. |
