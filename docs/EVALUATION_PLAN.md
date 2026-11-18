# VisionGuard Evaluation Plan & Experimental Benchmark Protocol

**Document Version:** 1.1.0 (Audited)  
**Status:** Canonical Experimental Protocol Specification  
**Date:** November 2026  
**Scope:** Rigorous evaluation protocols, attack injection procedures, metric definitions, multi-contributor experimental design, and formal hardware benchmarking framework.

---

## 1. Scientific Principles & Experimental Rigor

VisionGuard’s evaluation framework adheres to strict scientific standards:
1. **Known Ground Truth:** All anomaly and attack evaluations utilize controlled, programmatic injection into clean baselines with verifiable per-sample ground truth (`is_anomaly: bool`, `attack_type: str`, `contaminant_id: str`).
2. **Strict Split Discipline:** Development, calibration, and test splits are strictly separated. Thresholds are calibrated on an independent validation set, never tuned on the final test benchmark.
3. **No Fabricated Benchmarks:** Baseline measurements reflect actual execution runs on the local hardware profile. Prior to execution, all figures are marked as `UNVERIFIED ESTIMATE`.
4. **Honest Reporting of Failure Modes:** High false-positive rates, ambiguous edge cases, and algorithm blind spots are reported prominently alongside detection successes.

---

## 2. Formal Hardware Benchmarking Protocol (To Be Executed in Phase 2)

To replace all unverified estimates with empirically reproducible data, VisionGuard defines this standardized benchmarking protocol:

### 2.1 Benchmark Environment Specification
- **Host CPU:** x86_64 multi-core processor (AVX2 supported).
- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM), Driver 592.82, CUDA 12.1/13.1.
- **Host System RAM:** 16 GB DDR4/DDR5.
- **Software Stack:** Python 3.11.15, PyTorch 2.15.0+cpu (with optional cu121 build), TorchVision 0.29.2, OS: Windows 11 64-bit.
- **Test Datasets:**
  - Low-resolution classification: CIFAR-10 ($32 \times 32$, RGB, JPEG/PNG).
  - High-resolution object detection: COCO-Val 2017 ($640 \times 480$, RGB, JPEG).
- **Dataset Scales Evaluated:** Standardized subsets of $N = 1,000$ and $N = 5,000$ images.
- **Batch Sizes Evaluated:** $B \in \{1, 16, 32, 64\}$.

### 2.2 Measurement Methodology & Instrumentation
1. **Warm-Up Runs:** 5 consecutive unmeasured forward passes to initialize CUDA context, allocate tensor memory pools, and load model weights into CPU cache.
2. **Repeated Measurement Iterations:** 20 consecutive measured passes over identical data subsets.
3. **Latency Instrumentation:**
   - Measured using `time.perf_counter_ns()`.
   - Metrics computed: Arithmetic Mean, Median ($p50$), 95th percentile ($p95$), and 99th percentile ($p99$).
4. **VRAM Instrumentation:**
   - Evaluated via PyTorch CUDA memory APIs:
     ```python
     torch.cuda.reset_peak_memory_stats()
     # ... execute batch ...
     peak_vram_bytes = torch.cuda.max_memory_allocated()
     ```
5. **System RAM Instrumentation:**
   - Sampled via `psutil.Process().memory_info().rss` using an asynchronous polling daemon at 10 ms sampling intervals to record peak resident set size.
6. **Reporting Output:** Results committed into version-controlled JSON manifests in `reports/benchmarks/benchmark_run_<timestamp>.json`. Zero fabricated values permitted.

---

## 3. Evaluation Metrics: Mathematical Formulations & Suitability

### 3.1 Sample-Level Detection Metrics

| Metric | Mathematical Formula | Optimal When... | Limitations / Caveats |
|---|---|---|---|
| **Precision** | $\frac{TP}{TP + FP}$ | Minimizing false alarms on clean operational data is paramount. | Can appear deceptively high if detector is extremely conservative. |
| **Recall (TPR)** | $\frac{TP}{TP + FN}$ | Safety-critical scenarios where missing a poisoned sample causes deployment failure. | High recall can be achieved trivially by flagging all samples. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Balancing false positives and false negatives on balanced evaluation sets. | Sensitive to class prevalence in the test set. |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | Measuring the fraction of benign clean samples incorrectly flagged. | Does not indicate precision when negatives vastly outnumber positives. |
| **Area Under ROC (AUROC)** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Evaluating threshold-independent ranking across continuous anomaly scores. | **Misleading under extreme class imbalance** (e.g., $1\%$ poison rate). |
| **Area Under PR (AUPRC)** | $\int_0^1 P(R) \, dR$ | **Mandatory metric for rare-event anomaly detection** ($p \le 5\%$). | Baseline is equal to the contamination rate (prevalence), not 0.5. |

---

## 4. Contributor-Level Risk Attribution Experimental Design & Sensitivity Analysis

### 4.1 Controlled Simulation Scenarios
To rigorously evaluate the **Leave-One-Contributor-Out (LOCO) Empirical Bayes** risk model, VisionGuard defines a controlled multi-contributor experiment suite using CIFAR-10 / Pascal VOC:

1. **Equal Contributor Volumes:** 10 contributors submitting identical volumes ($N_i = 200$ each). Contributor #1 has 20% injected label flips; all others clean.
2. **Highly Imbalanced Volumes:** 
   - 2 High-Volume Contributors: $N = 2,500$ samples each (one clean, one with 5% poisons).
   - 10 Low-Volume Contributors: $N = 20$ samples each (nine clean, one with 25% poisons = 5 poisons).
3. **Small Sample Sizes ($N < 5$):** 5 contributors submitting between 1 and 4 samples. Verifies that shrinkage prevents single blurry images from triggering catastrophic high-risk classifications.
4. **Missing Provenance:** 30% of samples stripped of `provenance.json` metadata to test graceful degradation to sample-only reporting.
5. **Spoofed Metadata / Sybil Attack:** Adversary distributes 50 poisoned samples across 50 distinct fake contributor IDs ($N_i = 1$ each) to test detection limits under sybil attacks.
6. **Small Contributor Populations ($N_{\text{contributors}} \le 3$):** Benchmarking shrinkage stability when peer contributor counts are minimal, triggering hybrid blending with global prior $\mu_0 = 0.05$.

### 4.2 Hyperparameter Sensitivity Analysis
Experiments measure attribution stability across prior strength weights:
- **Baseline:** Prior weight $M = 20$, reference baseline prior $\mu_0 = 0.05$.
- **Sensitivity Sweeps:** Evaluate risk rankings under $M = 5$ (low shrinkage / high defect responsiveness) and $M = 50$ (high shrinkage / heavy conservatism).
- **Metric Stability:** Verify that contributor ranking order is robust across $M \in [5, 50]$ for high-volume contributors ($N_i \ge 100$).

### 4.3 Provenance Trust Boundaries & Suspicion Rules
- **Unverified Identity vs. Authenticated Provenance:**
  - Evaluators must distinguish between arbitrary text strings in `provenance.json` (`"contributor_id"`) and cryptographically signed manifests (Ed25519 digital signatures).
  - Unauthenticated metadata provides attribution hints, but cannot guarantee non-repudiation or prevent identity spoofing.
- **Metadata Clustering as Weak Suspicion:**
  - Metadata correlations (shared IP subnets, upload time clustering, device headers) are evaluated as **weak suspicion signals**.
  - Evaluators must verify that the platform never treats metadata clustering as definitive proof of common ownership, sybils, or malicious intent.

### 4.4 Mathematical Attribution Metrics
- **Batch-Level AUPRC:** Ability of smoothed contributor risk $R_{\text{smoothed}}(i)$ to distinguish compromised batches from clean batches across decision thresholds.
- **False Attribution Rate (FAR):** Fraction of clean, benign contributors whose smoothed risk score exceeds the quarantine threshold.
  > **AUDIT MANDATE:** The $< 5\%$ False Attribution Rate is classified strictly as an **EVALUATION TARGET**, NEVER a guaranteed mathematical property.

---

## 5. Data Integrity Experimental Suite (Controlled Injection Protocols)

> **MANDATORY CALIBRATION DECOUPLING:**  
> In accordance with VisionGuard's calibration standards, evaluation protocols for Duplicate Detection (EXP-DATA-01), Label Anomaly Detection (EXP-DATA-02), and OOD / Backdoor Detection (EXP-DATA-03/04) operate under **completely independent calibration workflows**. Thresholds are never shared between experiments.

### EXP-DATA-01: Exact & Near-Duplicate Detection (Split Leakage)
- **Target Datasets:** CIFAR-10 (Classification) and COCO-Val 2017 (Object Detection).
- **Clean Baseline:** Clean training and validation partitions with verified zero overlap.
- **Attack / Contamination Procedure:**
  - *Tier A (Exact Duplicates):* Inject 200 exact validation images into training partition without modification.
  - *Tier B (Perceptual Duplicates):* Inject 200 validation images modified with: (1) JPEG quality reduction ($Q=80$), (2) Bilinear scaling ($32 \times 32 \to 64 \times 64 \to 32 \times 32$), (3) $2\%$ border crop.
- **Contamination Rates:** $0.5\%, 1.0\%, 2.0\%, 5.0\%$ of split volume.
- **Detection Mechanism:** SHA-256 Digest comparison and Perceptual Hashing (pHash with calibrated threshold $\tau_H \le 6$).
- **Evaluation Metrics & Statistical Confidence:** Exact match precision ($100\%$), Near-duplicate recall, False Positive Rate (Target $FPR \le 0.1\%$). Evaluated over $\ge 10,000$ negative pairs to establish empirical 95% confidence intervals ($[0.04\%, 0.16\%]$ at $FPR=0.001$).

---

### EXP-DATA-02: Systematic Label-Flipping & Inversion
- **Target Datasets:** CIFAR-10 (10 classes) and Pascal VOC (20 classes).
- **Clean Baseline:** Standard verified ground-truth labels.
- **Attack Procedure:**
  - *Targeted Class Inversion:* Flip label from class $c_i$ to visually adjacent class $c_j$ (e.g., "automobile" $\to$ "truck", "cat" $\to$ "dog").
  - *Random Noise:* Uniform random label assignment across all $K$ classes.
- **Contamination Rates:** $p \in \{1\%, 2\%, 5\%, 10\%, 20\%\}$.
- **Detection Mechanism:**
  - Tier 2: Frozen ResNet-18 k-NN neighborhood label concordance ($C_k(i)$ with adaptive $k$).
  - Tier 3: Confident Learning joint distribution estimation $\hat{Q}_{\tilde{y}, y^*}$ (when prediction probabilities supplied).
- **Evaluation Metrics:** Precision, Recall, F1, AUPRC, Label Noise Error Discovery Rate.

---

### EXP-DATA-03: Dirty-Label Backdoor Poisoning (BadNets)
- **Target Datasets:** CIFAR-10 and GTSRB (Traffic Signs).
- **Clean Baseline:** Unmodified training set.
- **Attack Procedure:**
  - Inject a $3 \times 3$ white checkerboard trigger at the bottom-right corner of images from source class $c_s$.
  - Alter the ground-truth label to target class $c^*$ ("Speed Limit 80" or "Airplane").
- **Contamination Rates:** $p \in \{0.5\%, 1.0\%, 2.5\%, 5.0\%, 10.0\%\}$ of target class volume.
- **Detection Mechanism:**
  - Tier 3: Spectral Signatures (top singular vector projection $\tau_i$ of target class feature covariance).
  - Tier 3: Activation Clustering (PCA + $k$-means on penultimate layer activations).
- **Evaluation Metrics:** Poison Detection Recall at $1\%$ and $5\%$ FPR, Attack Success Rate (ASR) before and after filtering, Silhouette score separation.

---

### EXP-DATA-04: Clean-Label Feature Collision Poisoning
- **Target Datasets:** CIFAR-10 and GTSRB.
- **Clean Baseline:** Unmodified training set.
- **Attack Procedure:**
  - Select clean target test sample $x_{\text{test}}$ from class A.
  - Optimize bounded perturbation $\|\delta\|_\infty \le 16/255$ on base samples from class B to minimize feature distance $\|f(x_{\text{base}} + \delta) - f(x_{\text{test}})\|_2$ while keeping label as B.
- **Contamination Rates:** $p \in \{1\%, 2\%, 5\%\}$.
- **Detection Mechanism:**
  - Tier 2: Deep feature distance to class B centroid and k-NN local density in feature space.
  - Tier 1: High-frequency residual energy via Laplacian variance filtering.
- **Evaluation Metrics:** Feature outlier AUROC, AUPRC, Poison sample ranking percentile.

---

### EXP-DATA-05: Out-of-Distribution (OOD) Semantic Contamination
- **Target Datasets:** In-Distribution = CIFAR-10; Out-of-Distribution = SVHN (Street View House Numbers) and Tiny-ImageNet.
- **Clean Baseline:** In-distribution CIFAR-10 training set.
- **Attack / Contamination Procedure:** Inject semantically alien images into the CIFAR-10 training set without altering directory class designations.
- **Contamination Rates:** $p \in \{1\%, 2\%, 5\%, 10\%\}$.
- **Detection Mechanism:**
  - Tier 2: Penultimate layer k-NN Euclidean distance ($D_k(x)$) against an approved clean reference set using a frozen ResNet-18.
- **Evaluation Metrics:** OOD Detection AUROC, AUPRC, FPR at $95\%$ TPR, Score separation gap ($\mu_{\text{OOD}} - \mu_{\text{ID}}$).

---

### EXP-DATA-06: Object Detection Bounding-Box Geometric Corruption
- **Target Datasets:** Pascal VOC 2007 (5,011 images) and COCO-Val 2017 (5,000 images).
- **Clean Baseline:** Validated annotations JSON and text label files.
- **Attack Procedure:**
  - Inject 100 malformed bounding boxes: (1) Absolute coordinate overflow in COCO ($x + w > W$), (2) Normalized coordinate overflow in YOLO ($x_c + w/2 > 1.0$), (3) Negative dimensions ($w < 0$), (4) Zero-area boxes, (5) Conflicting overlapping boxes ($\text{IoU} > 0.90$ with contradictory classes).
- **Detection Mechanism:** Tier 1 Structural Geometry Validator.
- **Evaluation Metrics:** Detection Precision ($100\%$), Detection Recall ($100\%$), False Positive Rate ($0\%$).

---

## 6. Multi-Contamination Benchmark Matrix

| Experiment ID | Evaluation Benchmark | Evaluated Module | Contamination Levels Tested | Primary Metric | Baseline Comparison |
|---|---|---|---|---|---|
| **EXP-DATA-01** | Duplicate / Split Leakage | Data Integrity | $0.5\%, 1\%, 2\%, 5\%$ | AUPRC / Recall | Exact SHA-256 vs pHash |
| **EXP-DATA-02** | Label Noise Discovery | Data Integrity | $1\%, 2\%, 5\%, 10\%, 20\%$ | Precision / AUPRC | Random vs k-NN Concordance vs Confident Learning |
| **EXP-DATA-03** | BadNets Poisoning | Data Integrity | $0.5\%, 1\%, 2.5\%, 5\%, 10\%$ | Recall @ 5% FPR | Naive Clustering vs Spectral Signatures |
| **EXP-DATA-04** | Clean-Label Poisoning | Data Integrity | $1\%, 2\%, 5\%$ | AUROC / AUPRC | Random Selection vs Feature k-NN Outlier |
| **EXP-DATA-05** | Semantic OOD Injection | Data Integrity | $1\%, 2\%, 5\%, 10\%$ | AUROC / FPR@95%TPR | Raw Pixels vs Frozen ResNet-18 k-NN |
| **EXP-DATA-06** | Bounding Box Errors | Data Integrity | Fixed 100 corruptions | Precision & Recall | Schema Parser vs Heuristic Checks |
| **EXP-DATA-07** | Contributor Attribution | Data Integrity | Mixed volumes ($N \in [5, 2500]$) | Batch AUPRC / FAR | Naive Ratio vs LOCO Empirical Bayes |
| **EXP-MOD-01** | Model Substitution | Model Integrity | N/A (Discrete replacement) | Correctness ($100\%$) | File Name check vs SHA-256 Digest |
| **EXP-MOD-02** | Weight Modification | Model Integrity | 1 to 100 tensors altered | Fingerprint Sensitivity | SHA-256 vs Behavioral Probe Logits |
| **EXP-INF-01** | Record Tampering | Inference Integrity | 100 random row mutations | Tamper Recall ($100\%$) | Unsigned Log vs Ed25519 Canonical Signature |
| **EXP-INF-02** | Chain Deletion / Reorder | Inference Integrity | 50 row deletions / swaps | Chain Break Recall | Plain Table vs SHA-256 Hash Chaining |
| **EXP-INF-03** | Receipt Replay | Inference Integrity | 100 replayed tokens | Replay Rejection ($100\%$) | Static Timestamp vs Monotonic Sequence Enforcer |
| **EXP-SHIFT-01** | Lighting / Sensor Drift | Distribution Shift | Continuous drift over windows | Detection Delay / False Alarm | Raw Pixel KS vs Pretrained ResNet MMD |
