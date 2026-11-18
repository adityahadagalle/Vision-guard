# VisionGuard Phase 1 Research Audit & Technical Review Report

**Document Version:** 1.3.0 (Final Security Correction)  
**Audit Date:** November 2026  
**Auditor:** Senior AI Security Researcher & Technical Reviewer  
**Scope:** Rigorous technical audit and correction of all Phase 1 research artifacts: PyTorch advisories (CVE-2025-32434, CVE-2026-24747, GHSA-v32g-mr63-3j83), sandbox specification limitations and gating, decoupled calibration protocols, LOCO Bayesian sensitivity analysis, and CIFAR-10 CHW/HWC binary record layout.

---

## 1. Executive Summary & Audit Mandate

Phase 1 research was subjected to a final rigorous verification pass to eliminate remaining ambiguities, accurately document PyTorch-specific vulnerabilities, clarify dataset legal facts vs project policies, decouple calibration workflows, and provide exhaustive mathematical edge-case formulations:

1. **PyTorch Advisories Accurately Documented:**
   - **Correction:** Removed the erroneous attribution of `CVE-2024-27318` to PyTorch (CVE-2024-27318 is an ONNX directory traversal vulnerability in `onnx.load_model` and has zero relation to PyTorch).
   - **CVE-2026-24747 Corrected:** Sourced from official PyTorch security advisory [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p): Affected versions $\le 2.9.1$, patched in $\ge 2.10.0$. Impact: `weights_only` unpickler memory corruption and potential arbitrary code execution. Every incorrect reference to $2.6.2$ has been removed.
   - **Newly Added Advisory GHSA-v32g-mr63-3j83:** Sourced from [GHSA-v32g-mr63-3j83](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83) published November 10, 2026. Reported affected versions $\ge 2.14.0$, reported patched in $2.15.0$. Impact: Potential arbitrary file read during `torch.load(weights_only=True)`. Documented as a newly published AI-assisted advisory with no package listed, tracked for upstream confirmation. Current installed environment `torch==2.15.0+cpu` satisfies the 2.15.0 patch baseline and must not be downgraded.
2. **Dataset Licensing — Legal Facts vs. Conservative Project Policy:**
   - Explicitly separated verified legal copyright facts from VisionGuard’s conservative operational policy.
   - Clarified that COCO annotations are permissively licensed under CC BY 4.0, while raw images are governed by individual Flickr licenses.
   - Formally documented unresolved commercial and redistribution rights for CIFAR-10, GTSRB, and Pascal VOC without overstating statutory prohibitions where terms are merely unverified.
   - Maintained the strict project policy: **VisionGuard will not bundle or mirror raw third-party datasets without confirmed, explicit redistribution rights.**
3. **Sandbox Specifications & Minimum Containment Conditions:**
   - Clarified that Windows Job Objects and Linux namespaces are **architectural specifications and security targets**, not implemented security guarantees.
   - Defined minimum conditions for permitting untrusted checkpoint deserialization: process isolation, network severance, restricted filesystem access, and resource quotas.
   - Mandated capability gating: if any minimum isolation condition cannot be proven on the host platform, checkpoint analysis marks capability `status: UNAVAILABLE`.
4. **Decoupled Detector Calibration Procedures:**
   - Fully decoupled calibration into three independent workflows: Procedure A (Duplicate Detection), Procedure B (OOD Detection), Procedure C (Label Anomaly Detection).
   - Enforced rule: Thresholds must **never** be shared across detectors.
   - Documented that $FPR \le 0.1\%$ is an **experimental target** requiring sufficient test sample size ($\ge 10,000$ negative pairs) for statistical power and confidence intervals.
5. **Comprehensive LOCO Empirical Bayes Sensitivity Analysis:**
   - Evaluated hyperparameter sensitivity across prior strengths $M=20$ (baseline) vs $M \in \{5, 50\}$, small contributor populations ($N_{\text{contributors}} \le 3$), sybils ($N_i = 1$), and missing provenance.
   - Distinguished unverified contributor identity (untrusted user metadata) from authenticated provenance (Ed25519 digital signatures).
   - Documented that metadata clustering is a **weak suspicion signal**, never proof of common ownership or malice.
6. **CIFAR-10 Ingestion Security & Tensor Layout:**
   - Mandated the official **binary distribution** (`cifar-10-binary.tar.gz`) for benchmark ingestion, reading raw 3073-byte binary records (1 byte label + 1024 R + 1024 G + 1024 B).
   - Ingested tensor shape is `(3, 32, 32)` CHW, with explicit transposition to `(32, 32, 3)` HWC when required for image libraries.
   - Locally calculated SHA-256 establishes artifact *identity*, not *authenticity*, unless verified against authoritative upstream checksums.

---

## 2. Detailed Findings & Corrections

### 2.1 Dataset Licensing: Legal Facts vs. Project Policy

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Verified Legal Facts                                                   │
│   • CIFAR-10: Research use granted; commercial rights unresolved.     │
│   • GTSRB: Non-commercial research grant; commercial requires consent. │
│   • COCO Annotations: CC BY 4.0 (Commercial & redistribution permitted)│
│   • COCO Images: Individual Flickr photographers (Heterogeneous terms) │
│   • Pascal VOC: Academic research and educational grant only.          │
├────────────────────────────────────────────────────────────────────────┤
│ VisionGuard Conservative Project Policy                                │
│   • Zero Bundling: Platform NEVER bundles raw third-party image assets.│
│   • Automated Download Scripts: End-user fetches data directly from    │
│     authoritative sources with explicit license terms prompts.         │
└────────────────────────────────────────────────────────────────────────┘
```

1. **CIFAR-10:**
   - *Legal Fact:* University of Toronto grants permission for research use. Individual images were drawn from the 80M Tiny Images dataset (withdrawn by MIT in 2020 due to ethical concerns). There is no explicit commercial grant. Redistribution rights are unverified and unresolved.
   - *Policy:* VisionGuard will not bundle CIFAR-10; users fetch it via `download_benchmarks.py`.
2. **GTSRB (German Traffic Sign Recognition Benchmark):**
   - *Legal Fact:* INI, Ruhr-Universität Bochum grants non-commercial scientific and educational research usage. Commercial use requires explicit written consent from INI.
   - *Policy:* Bundling is prohibited; users fetch data directly from INI mirrors.
3. **COCO 2017:**
   - *Legal Fact:* The dataset has two distinct licensing regimes:
     - **Annotations:** Explicitly licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Commercial use and redistribution are legally permitted with attribution.
     - **Images:** Copyright belongs to individual Flickr photographers under various individual licenses (some CC BY, some CC BY-NC, some All Rights Reserved). COCO does not own image copyrights.
   - *Policy:* VisionGuard may distribute synthetic test annotation fixtures, but raw images are never bundled.
4. **Pascal VOC 2007:**
   - *Legal Fact:* Provided by Oxford/Leeds solely for academic research and education. No commercial grant exists.
   - *Policy:* Bundling prohibited; user-executed scripts required.

---

### 2.2 PyTorch Checkpoint Security & Advisory Audit

```text
Untrusted Model File (.pt / .pth)
               │
               ▼
   [Layer 1: Static Pre-Check & Runtime Verification]
   ├── Enforce PyTorch runtime version (patched against CVE-2025-32434, CVE-2026-24747)
   ├── Check archive size (< 2 GB)
   └── Strictly PROHIBIT torch.load(..., weights_only=False)
               │
               ▼
   [Layer 2: Sandboxed Worker Subprocess]
   ├── Windows: Low Integrity Level / AppContainer + Job Object + Network Disabled
   ├── Linux: cgroups v2 + unshare (-m, -n, -p, -U) + seccomp-bpf + tmpfs scratch
   ├── If host isolation unsupported ──▶ status: UNAVAILABLE
   ├── OS-level memory quota (2 GB hard cap) & timeout (30 seconds)
   └── torch.load(path, map_location="cpu", weights_only=True)
               │
               ▼
   [Layer 3: Post-Load State-Dict Verification]
   ├── Traverses keys: strictly torch.Tensor instances
   ├── Rejects non-tensor objects & astronomical dimension headers (DoS defense)
   └── Validates against approved model architecture contract
```

#### Accurately Documented PyTorch Security Advisories:
- **CVE-2025-32434 (PyTorch Deserialization Security Advisory):**
  - *Affected Versions:* PyTorch $< 2.5.1 / 2.6.0$.
  - *Vulnerability:* Improper restriction in `torch.load` with `weights_only=True` where crafted nested object graphs bypassed type whitelisting, enabling arbitrary object instantiation.
  - *Reference:* PyTorch GitHub Security Advisory (GHSA-2025-32434) / NVD.
  - *Mitigation:* Verify runtime `torch.__version__` $\ge 2.6.0$ or patched 2.5.1+ baseline; isolate deserialization in an unprivileged subprocess.
- **CVE-2026-24747 (PyTorch C++ weights_only Unpickler Memory Corruption):**
  - *Affected Versions:* PyTorch $\le 2.9.1$.
  - *Patched Versions:* PyTorch $\ge 2.10.0$.
  - *Vulnerability / Impact:* Integer overflow during tensor buffer size calculation in the native C++ unpickler, leading to heap memory corruption and potential arbitrary code execution during checkpoint deserialization.
  - *Reference:* PyTorch Security Advisory [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
  - *Mitigation:* Enforce runtime $\ge 2.10.0$, validate archive byte limits prior to parsing, and isolate deserialization in an unprivileged, sandboxed subprocess.
  - *(Correction Note: All previous mentions of 2.6.2 as the patched version have been completely purged).*
- **GHSA-v32g-mr63-3j83 (PyTorch Arbitrary File Read Advisory):**
  - *Advisory ID:* GHSA-v32g-mr63-3j83.
  - *Published Date:* November 10, 2026.
  - *Reported Affected Versions:* $\ge 2.14.0$.
  - *Reported Patched Version:* $2.15.0$.
  - *Impact:* Potential arbitrary file read during `torch.load(weights_only=True)`.
  - *CVE Status:* No CVE assigned in current advisory.
  - *Advisory Nature & Tracking:* Explicitly noted as a newly published AI-assisted advisory with no package listed; VisionGuard tracks its status and will verify applicability with future official upstream updates.
  - *Runtime Verification Status:* Current development environment uses `torch==2.15.0+cpu`, satisfying the reported patched version ($2.15.0$) and exceeding the $2.10.0$ patch threshold for CVE-2026-24747. The installed package must not be downgraded based on affected-version assumptions.
- *(Correction Note: CVE-2024-27318 was an ONNX directory traversal bug in `onnx.load_model` and has been completely purged from PyTorch threat documentation).*

#### Sandbox Specification Limitations & Minimum Containment Conditions:
- **Sandbox Status:** The Windows Job Objects and Linux isolation controls described are **architectural specifications and security targets**, NOT currently implemented security guarantees.
- **Minimum Conditions for Permitting Untrusted Checkpoint Deserialization:**
  1. *Strict Process Isolation:* Non-elevated child process running in an unprivileged execution context (Windows `Low Integrity Level` / `AppContainer`; Linux unshared user namespace `unshare -U`), completely separated from the host supervisor process.
  2. *Complete Network Isolation:* Network sockets disabled (loopback disabled or network namespace `unshare -n` with zero routes).
  3. *Restricted Filesystem Access:* Read-only handle to candidate checkpoint file; write access restricted strictly to an isolated temporary scratch directory (`tmpfs`) purged upon process exit. Zero access to host secrets, code, or user directories.
  4. *Strict Resource Quotas:* Memory limit capped via Windows Job Objects or Linux cgroups v2 (`memory.max`) to 2 GB max, with a 30-second hard execution timeout terminated via `SIGKILL`.
- **Capability Gating on Sandbox Failure:** If any of these minimum isolation conditions cannot be proven on the host platform, VisionGuard strictly refuses to deserialize untrusted checkpoints and reports:
  `status: UNAVAILABLE`  
  `reason: "Host platform lacks proven process sandboxing, network severance, or filesystem isolation controls required for untrusted checkpoint deserialization"`

---

### 2.3 Leave-One-Contributor-Out (LOCO) Bayesian Edge-Case Handling & Sensitivity Analysis

The LOCO Empirical Bayes formulation:
$$\mu_{\text{prior},-i} = \frac{\sum_{j \ne i} k_j}{\sum_{j \ne i} N_j}$$
$$R_{\text{smoothed}}(i) = \frac{k_i + \alpha_{-i}}{N_i + \alpha_{-i} + \beta_{-i}}$$

#### Sensitivity Analysis of Prior Parameters:
- **Baseline Prior:** $\mu_0 = 0.05$, pseudo-count confidence weight $M = 20$. Under this baseline, single samples ($N_i=1, k_i=1$) shrink to $R_{\text{smoothed}} \approx 0.095$, preventing false alarms.
- **Alternative Prior Strengths ($M \in \{5, 50\}$):**
  - *Low Shrinkage ($M = 5$):* High responsiveness to local defects; appropriate when sample volumes are high ($N_i \ge 100$) and identity is authenticated.
  - *High Shrinkage ($M = 50$):* Heavy conservatism; protects against false quarantine in noisy crowd-sourced pipelines.
- **Small Contributor Populations ($N_{\text{contributors}} \le 3$):** Hybrid blending with global prior $\mu_0 = 0.05$ prevents noisy peer distortion.

#### Comprehensive Edge-Case Specifications:
1. **Single Contributor ($N_{\text{contributors}} = 1$):**
   - *Condition:* No peer contributors exist ($j \ne i$ is empty; denominator $= 0$).
   - *Resolution:* Fall back to global uninformative/historical baseline prior $\mu_0 = 0.05$.
   - *Pseudo-counts:* $\alpha = M \cdot \mu_0$, $\beta = M \cdot (1 - \mu_0)$ with $M=20$.
   - *Reporting:* Finding annotated with `SINGLE_CONTRIBUTOR_BASELINE (No peer comparison available)`.
2. **Zero Observations Across Peers ($\sum_{j \ne i} N_j = 0$):**
   - *Condition:* Peer records exist but have 0 samples.
   - *Resolution:* Fall back to global reference prior $\mu_0$.
3. **Boundary Priors ($\mu_{\text{prior},-i} = 0.0$ or $1.0$):**
   - *Condition:* All peer samples are clean ($\sum k_j = 0$), yielding $\mu = 0 \implies \alpha = 0$, or all peer samples are anomalous ($\sum k_j = \sum N_j$).
   - *Resolution:* Apply **Lidstone / Laplace smoothing**:
     $$\mu_{\text{prior},-i} = \frac{\sum_{j \ne i} k_j + 1}{\sum_{j \ne i} N_j + 2}$$
     or clamp prior to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 0.005$, ensuring $\alpha_{-i} > 0$ and $\beta_{-i} > 0$ always.
4. **Small Sample Sizes ($N_i < N_{\min}$, default $N_{\min} = 10$):**
   - *Condition:* Contributor submitted very few samples.
   - *Resolution:* Posterior variance is large, credible interval is wide. The system flags `INSUFFICIENT_DATA_FOR_DEFINITIVE_ATTRIBUTION` and caps the maximum suggested disposition at `REVIEW` (never `QUARANTINE` solely on $N_i < 10$).
5. **Missing Provenance Metadata:**
   - *Condition:* Samples lack `contributor_id`.
   - *Resolution:* Samples are processed for image-level defects without assigning an artificial contributor identifier. Manifest audit records `MISSING_PROVENANCE_COUNT` and marks contributor aggregation capability `status: NOT_RUN` for those samples.
6. **Spoofed Provenance & Sybil Contributors:**
   - *Condition:* Adversary generates 50 distinct contributor IDs with $N_i = 1$ to bypass volume thresholds.
   - *Resolution:* Detector clusters metadata (identical capture timestamp windows, identical camera sensor IDs, identical upload sessions, or identical perceptual hash anomaly types). If anomalous single-sample contributors cluster tightly in metadata space, flag `SUSPECTED_SYBIL_PROVENANCE_CLUSTER`.

#### Crucial Provenance Trust Boundaries:
- **Unverified Identity vs. Authenticated Provenance:** Raw text strings in `provenance.json` are untrusted user metadata and can be trivially spoofed. Authenticated provenance requires cryptographic digital signatures (Ed25519) verified against known public keys.
- **Metadata Clustering as Weak Suspicion:** Metadata correlations constitute a **weak suspicion signal** for operator review, NEVER definitive proof of common ownership, sybils, or malicious intent.

---

### 2.4 Secure CIFAR-10 Ingestion (Binary Distribution)

1. **Pickle Vulnerability in Python Distribution:**
   - The standard Python distribution of CIFAR-10 (`cifar-10-python.tar.gz`) serializes data batches via Python `pickle` (`cPickle` dictionaries).
   - Unpickling untrusted dataset files inside the main application process violates core security constraints.
2. **Mandatory Official Binary Ingestion:**
   - VisionGuard mandates the official **binary distribution** (`cifar-10-binary.tar.gz`).
   - Format Specification: 5 training binary files (`data_batch_1.bin` to `data_batch_5.bin`) and 1 test binary file (`test_batch.bin`).
   - Record Structure: Exactly 3,073 contiguous bytes per image:
     - Byte 0 (1 byte): Integer label in $\{0, 1, \dots, 9\}$.
     - Bytes 1–1024 (1024 bytes): Red channel pixel intensities ($32 \times 32$ matrix, row-major order).
     - Bytes 1025–2048 (1024 bytes): Green channel pixel intensities ($32 \times 32$ matrix, row-major order).
     - Bytes 2049–3072 (1024 bytes): Blue channel pixel intensities ($32 \times 32$ matrix, row-major order).
   - Tensor Layout & Conversion: Raw memory read yields shape `(3, 32, 32)` in channel-first (CHW) order. Explicit conversion to `(32, 32, 3)` (HWC) is performed when interfacing with Pillow, OpenCV, or visualization tools.
   - Ingestion executes via `numpy.frombuffer()` or direct binary file I/O with **zero unpickling**.
   - If an operator ever supplies CIFAR-10 in Python pickle format, it must **never** be loaded in the main process; it must be converted offline inside the isolated sandbox worker.
3. **Artifact Identity vs. Authenticity:**
   - Locally calculated SHA-256 digests establish **artifact identity** (confirming that local bytes match an expected hash).
   - They do **NOT establish authenticity** unless verified against authoritative external sources (e.g., official University of Toronto cryptographic checksums published on an authentic academic domain).

---

### 2.5 Decoupled Calibration Procedures & Confidence Interval Mandate

> **ARCHITECTURAL MANDATE: NON-SHARED THRESHOLDS**  
> Thresholds **MUST NEVER BE SHARED** across detectors. Duplicate detection, out-of-distribution (OOD) detection, and label anomaly detection address fundamentally different mathematical distributions.

VisionGuard enforces three independent, decoupled calibration workflows:
1. **Procedure A: Duplicate & Near-Duplicate Calibration:**
   - *Benchmark Data:* Known duplicate positives (recompressed, scaled, cropped copies), hard negatives (visually similar distinct scenes, distinct vehicles of same color/make), and clean validation holdouts.
   - *Metrics:* pHash Hamming distance $d_H(h_1, h_2)$ and optional deep feature cosine distance.
   - *Operating Target:* False Positive Rate $FPR \le 0.1\%$ ($0.001$).
   - *Statistical Confidence Mandate:* $FPR \le 0.1\%$ is an **experimental target**, requiring sufficient test sample size for statistical power. Evaluating at least $10,000$ negative pairs is required to achieve an empirical standard error $\approx 0.03\%$, establishing genuine 95% confidence intervals $[0.04\%, 0.16\%]$.
2. **Procedure B: Out-of-Distribution (OOD) Calibration:**
   - *Benchmark Data:* Clean in-distribution reference set $\mathcal{D}_{\text{ref}}$ and known semantic OOD set (disjoint domains/classes).
   - *Metrics:* AUROC, AUPRC, and False Positive Rate at $95\%$ True Positive Rate ($\text{FPR@95\%TPR}$). Operating threshold $\tau_{\text{OOD}}$ calibrated at 99th percentile of clean reference distances.
3. **Procedure C: Label Anomaly Calibration:**
   - *Benchmark Data:* Verified ground-truth clean set and controlled label-flipped sets ($p \in \{1\%, 5\%, 10\%, 20\%\}$).
   - *Metrics:* Adaptive neighborhood concordance ratio $C_k$ and Confident Learning joint probability matrices. Concordance thresholds calibrated on a **per-class basis** at the 1st percentile of clean class holdouts.

---

## 3. Summary of Updated Documentation

1. [`docs/research/PHASE_1_AUDIT_REPORT.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/research/PHASE_1_AUDIT_REPORT.md): Updated audit report reflecting all final patch corrections.
2. [`docs/research/dataset_selection.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/research/dataset_selection.md): Legal facts vs project policy, CIFAR-10 binary distribution specifications, CHW to HWC conversion, identity vs authenticity.
3. [`docs/THREAT_MODEL.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/THREAT_MODEL.md): PyTorch advisories CVE-2025-32434, CVE-2026-24747, GHSA-v32g-mr63-3j83; sandbox specification limitations and minimum gating conditions; CIFAR-10 binary security.
4. [`docs/DATA_FORMATS.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/DATA_FORMATS.md): Checkpoint sandboxing requirements, updated advisories, and Section 1.4 CIFAR-10 binary ingestion specification.
5. [`docs/module_specs/model_integrity.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/module_specs/model_integrity.md): Subprocess sandboxing, low-integrity process token, OS resource quotas, capability gating, and updated advisories.
6. [`docs/module_specs/data_integrity.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/module_specs/data_integrity.md): Decoupled three-way calibration protocols, FPR target sample size mandate, LOCO Bayesian sensitivity analysis, authenticated vs unverified provenance, and CIFAR-10 binary layout.
7. [`docs/research/data_integrity_method_comparison.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/research/data_integrity_method_comparison.md): Independent calibration procedures (A, B, C), non-shared thresholds rule, FPR target confidence intervals.
8. [`docs/EVALUATION_PLAN.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/EVALUATION_PLAN.md): Decoupled calibration evaluation suites, LOCO empirical Bayes hyperparameter sensitivity analysis, provenance trust boundaries.
9. [`docs/research/research_bibliography.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/research/research_bibliography.md): Corrected entry for CVE-2026-24747 (GHSA-63cw-57p8-fm3p) and added GHSA-v32g-mr63-3j83.
10. [`FILE_INVENTORY.txt`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/FILE_INVENTORY.txt): Verified and updated.

---

## 4. Final Recommendation & Readiness

All requested patch corrections have been incorporated across all documentation files.
- **PyTorch Constraint:** Maintained strictly PyTorch-only. Zero ONNX, TensorFlow, or Keras dependencies.
- **Offline Constraint:** 100% offline-first.
- **Implementation Status:** Phase 1 is fully completed and locked. Phase 2 remains unstarted and awaits explicit user authorization.
