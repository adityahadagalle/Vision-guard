# Module Specification: Data Integrity Engine

**Document Version:** 1.2.0 (Audited Patch)  
**Status:** Detailed Technical Specification  
**Date:** November 2026  
**Scope:** Complete algorithmic, structural, and architectural specification for VisionGuard's Data Integrity Engine, including LOCO Bayesian edge cases and three-way calibration protocols.

---

## 1. Goal & Mission

The Data Integrity Engine assesses computer vision datasets for corrupted files, structural defects, near-duplicate contamination, split leakage, label noise, semantic out-of-distribution (OOD) samples, and adversarial poisoning triggers. It links sample-level anomaly findings to provenance metadata (contributor, source sensor, batch) using Leave-One-Contributor-Out (LOCO) empirical Bayes risk smoothing with comprehensive edge-case handling, reporting all findings with capability-aware status and verifiable mathematical evidence.

---

## 2. Ingestion Contracts & Prerequisite Tiers

The module ingests computer vision datasets under an explicit four-tier prerequisite model:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Model-Free (Zero-Knowledge)                                    │
│   Inputs: Raw image directory or archive, annotations (COCO/YOLO/VOC)   │
│   Available: File checks, magic bytes, exact SHA-256, pHash, box geom  │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Representation-Dependent (Generic Pretrained Backbone)         │
│   Inputs: Tier 1 + Local frozen PyTorch backbone (ResNet-18 weights)   │
│   Available: Feature k-NN OOD distance, neighbor label concordance     │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Model-Dependent (Task-Trained Network / Predictions)           │
│   Inputs: Tier 2 + Checkpoint trained on data OR out-of-fold probs     │
│   Available: Spectral Signatures, Activation Clustering, Confident Lrn │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Metadata-Conditional (Provenance Manifest)                     │
│   Inputs: Provenance JSON manifest (`contributor_id`, `batch_id`)      │
│   Available: LOCO empirical Bayes contributor risk scoring & ranking   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Format-Specific Structural & Coordinate Validation (Tier 1)

Format-specific coordinate validation routines must **never** be mixed.

### 3.1 COCO 2017 Format (Absolute Pixel Bounding Boxes)
- Coordinate representation: `[x_min, y_min, width, height]` in **absolute floating-point pixels**.
- Origin: $(x_{\min}, y_{\min})$ is the top-left corner in pixel space.
- Mathematical Validation Rules:
  $$x_{\min} \ge 0.0, \quad y_{\min} \ge 0.0$$
  $$w > 0.0, \quad h > 0.0$$
  $$x_{\min} + w \le W_{\text{image}}, \quad y_{\min} + h \le H_{\text{image}}$$
- Validation Actions:
  - If $x_{\min} + w > W_{\text{image}}$ or $y_{\min} + h > H_{\text{image}}$, flag as `COCO_BOUNDING_BOX_OUT_OF_BOUNDS`.
  - If $w \le 0$ or $h \le 0$, flag as `COCO_DEGENERATE_BOX_DIMENSIONS`.
  - Validate that every `annotation.image_id` matches an existing record in `images[].id`.

### 3.2 YOLO Format (Normalized Center Bounding Boxes)
- Coordinate representation: line-delimited records `[class_id, x_center, y_center, width, height]` in **normalized coordinates** ($[0.0, 1.0]$).
- Origin: $(x_c, y_c)$ is the geometric center of the bounding box.
- Mathematical Validation Rules:
  $$0.0 \le x_c \le 1.0, \quad 0.0 \le y_c \le 1.0$$
  $$0.0 < w \le 1.0, \quad 0.0 < h \le 1.0$$
  $$x_c - \frac{w}{2} \ge 0.0, \quad x_c + \frac{w}{2} \le 1.0$$
  $$y_c - \frac{h}{2} \ge 0.0, \quad y_c + \frac{h}{2} \le 1.0$$
  $$\text{class\_id} \in \{0, 1, \dots, K-1\} \quad (\text{integer})$$
- Validation Actions:
  - If any normalized coordinate is $< 0$ or $> 1$, flag as `YOLO_COORDINATE_OUT_OF_RANGE`.
  - If $x_c - w/2 < 0$ or $x_c + w/2 > 1$, flag as `YOLO_BOX_EXCEEDS_IMAGE_BOUNDS`.
  - If `class_id` is outside the declared classes in `dataset.yaml`, flag as `YOLO_UNDEFINED_CLASS_ID`.
  - **Parsing Empty vs. Missing Labels:**
    - Empty `.txt` file $\implies$ Valid negative image (zero objects).
    - Missing `.txt` file for an existing image $\implies$ Flag `ORPHANED_IMAGE_ERROR`.
    - `.txt` file with no corresponding image $\implies$ Flag `ORPHANED_LABEL_ERROR`.

### 3.3 CIFAR-10 Official Binary Ingestion Format (`cifar-10-binary.tar.gz`)
To guarantee safe, non-executable ingestion without invoking Python `pickle`, VisionGuard ingests CIFAR-10 via its official binary distribution:
- **Binary Record Layout:** Contiguous 3,073-byte fixed records:
  - Byte 0 (1 byte): Class label ($0 \dots 9$).
  - Bytes 1–1024 (1024 bytes): Red channel ($32 \times 32$ pixels, row-major order).
  - Bytes 1025–2048 (1024 bytes): Green channel ($32 \times 32$ pixels, row-major order).
  - Bytes 2049–3072 (1024 bytes): Blue channel ($32 \times 32$ pixels, row-major order).
- **Tensor Layout & Conversion:** Ingested into memory as shape `(3, 32, 32)` in channel-first (CHW) layout. Explicit transpose to `(32, 32, 3)` (HWC) is performed whenever visual libraries (Pillow, OpenCV) or HWC transformations are invoked.
- **Identity vs. Authenticity Disclaimer:** Locally calculated SHA-256 digests confirm **artifact identity** (local byte consistency), NOT authenticity. Authenticity requires verification against official upstream hashes published on an authentic academic domain or signed release receipts.

---

## 4. Algorithmic Mechanisms & Three-Way Independent Calibration Protocols

> **CRITICAL ARCHITECTURAL DIRECTIVE: DECOUPLED CALIBRATION WORKFLOWS**  
> Detection thresholds **MUST NEVER BE SHARED** across detectors. Duplicate detection, out-of-distribution detection, and label anomaly detection each solve distinct mathematical problems under distinct data distributions. A threshold selected for perceptual hashing has zero applicability to OOD density scoring or label concordance.

VisionGuard defines three independent, decoupled calibration procedures:

### 4.1 Procedure A: Duplicate & Near-Duplicate Calibration (Tiers 1 & 2)
1. **Calibration Objective:** Maximize true duplicate recall while strictly bounding false alarm pairs on visually similar distinct images.
2. **Benchmark Data Partitions:**
   - *Known Duplicate Positives:* Synthetic pairs with controlled alterations: JPEG recompression ($Q \in \{60, 75, 90\}$), bilinear downscaling and upscaling, border crops ($1\% \dots 5\%$), and minor gamma shifts ($\gamma \in \{0.9, 1.1\}$).
   - *Hard Negative Pairs:* Visually similar but semantically distinct natural pairs (e.g., distinct sedans of the same color, consecutive frames of different objects, same-species distinct animals).
   - *Independent Clean Validation Split:* Unmodified validation holdout set to measure empirical background collision rates.
3. **Calibrated Metrics & Metrics Target:**
   - Evaluates pHash Hamming distance $d_H(h_1, h_2)$ and optional deep feature cosine similarity.
   - Operating threshold target: $FPR \le 0.1\%$ ($0.001$).
   - **Sample Size & Statistical Power Mandate:** The $FPR \le 0.1\%$ target is an **experimental target**, requiring sufficient validation pairs for statistical confidence. At $FPR = 0.001$, evaluating $\ge 10,000$ negative pairs yields an empirical standard error $\approx \sqrt{0.001 \times 0.999 / 10000} \approx 0.03\%$, establishing genuine 95% confidence intervals $[0.04\%, 0.16\%]$. Thresholds must not be claimed calibrated without documenting sample counts and confidence intervals.
4. **Split Leakage Evaluation:** Applies calibrated $\tau_H$ across train-val split pairs $(x_{\text{train}}, x_{\text{val}})$.

---

### 4.2 Procedure B: Out-of-Distribution (OOD) Detection Calibration (Tier 2)
1. **Calibration Objective:** Detect samples that fall outside the semantic support of the training data distribution while preserving clean in-distribution samples.
2. **Benchmark Data Partitions:**
   - *In-Distribution Reference Set ($\mathcal{D}_{\text{ref}}$):* Clean, verified in-distribution validation holdouts with no contamination.
   - *Known Semantic OOD Set:* Natural image samples from disjoint classes or disjoint domain datasets (e.g., SVHN vs. CIFAR-10, texture anomalies vs. traffic signs).
3. **Calibrated Metrics & Scoring:**
   - Measures normalized embedding distance to the $k$-th nearest neighbor in $\mathcal{D}_{\text{ref}}$:
     $$d_{\text{OOD}}(x) = \|\mathbf{z} - \mathbf{z}_{(k),\text{ref}}\|_2$$
   - Calibration curves: Plot AUROC, AUPRC, and measure False Positive Rate at $95\%$ True Positive Rate ($\text{FPR@95\%TPR}$).
   - Operating threshold $\tau_{\text{OOD}}$ is calibrated at the 99th percentile of clean in-distribution reference holdouts.
   - **Guardrail:** Output reports: *"Sample is statistically distant from reference distribution; this indicates an OOD sample, NOT confirmed adversarial poisoning."*

---

### 4.3 Procedure C: Label Anomaly & Noise Calibration (Tiers 2 & 3)
1. **Calibration Objective:** Identify mislabeled, inverted, or corrupted labels without falsely flagging difficult or ambiguous in-class samples.
2. **Benchmark Data Partitions:**
   - *Clean Ground-Truth Set:* Verified correctly labeled validation data.
   - *Controlled Label-Flipped Sets:* Synthetically corrupted datasets with controlled label flips (symmetric uniform noise and asymmetric visually adjacent class swaps at $p \in \{1\%, 5\%, 10\%, 20\%\}$).
3. **Calibrated Metrics & Scoring:**
   - *Neighborhood Concordance ($C_k$):* Evaluates fraction of $k$ nearest neighbors in embedding space sharing the sample's declared label using adaptive neighborhood $k = \max(3, \min(15, \lfloor 0.1 \cdot N_c \rfloor))$.
     $$C_k(i) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \mathbb{I}(y_j = y_i)$$
   - *Class-Specific Thresholds:* Because classes naturally vary in intra-class variance and feature density, concordance thresholds $\tau_{\text{concordance}, c}$ are calibrated on a **per-class basis** at the 1st percentile of clean class holdouts.
   - *Confident Learning (Tier 3):* When out-of-fold predicted probabilities $\hat{\mathbf{P}}$ are available, estimate the joint distribution $\hat{Q}_{\tilde{y}, y^*}$ and filter using confident threshold matrix $\mathbf{t}_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{P}(y^* = j \mid x)$.

---

### 4.4 Data Poisoning & Backdoor Trigger Detection (Tier 3)
1. **Spectral Signatures (Tran et al., NeurIPS 2018):**
   - Centers representations for candidate class $c$: $\mathbf{f}_i - \mathbf{\mu}_c$.
   - Computes top right singular vector $\mathbf{v}_1$ of centered representation matrix via SVD.
   - Computes outlier projection score:
     $$\tau_i = \left( (\mathbf{f}_i - \mathbf{\mu}_c) \cdot \mathbf{v}_1 \right)^2$$
   - Decision threshold calibrated on clean holdout representations using Chebyshev bounds or empirical 99th percentiles.
2. **Activation Clustering (Chen et al., 2018):**
   - Applies PCA dimensionality reduction to penultimate layer activations within each class, followed by 2-means clustering and silhouette score analysis. Flag classes with silhouette separation $> 0.35$ as suspected backdoors.

---

## 5. Contributor-Level Risk Analysis & Comprehensive Edge-Case Specifications (Tier 4)

### 5.1 Leave-One-Contributor-Out (LOCO) Prior Formulation & Sensitivity Analysis
To eliminate the self-influence of large contributors on the estimated global prior, VisionGuard implements **LOCO Empirical Bayes Beta-Binomial Smoothing**:

For contributor $i$ with $k_i$ flagged samples out of $N_i$ total submitted samples:
1. **Leave-One-Out Prior Anomaly Rate:**
   $$\mu_{\text{prior},-i} = \frac{\sum_{j \ne i} k_j}{\sum_{j \ne i} N_j}$$
2. **Prior Pseudo-Counts ($M = 20$ calibrated confidence weight):**
   $$\alpha_{-i} = M \cdot \mu_{\text{prior},-i}, \quad \beta_{-i} = M \cdot (1 - \mu_{\text{prior},-i})$$
3. **Smoothed Contributor Anomaly Rate:**
   $$R_{\text{smoothed}}(i) = \frac{k_i + \alpha_{-i}}{N_i + \alpha_{-i} + \beta_{-i}}$$
4. **Statistical Uncertainty (Posterior Variance):**
   $$\text{Var}(p_i \mid k_i) = \frac{(k_i + \alpha_{-i})(N_i - k_i + \beta_{-i})}{(N_i + \alpha_{-i} + \beta_{-i})^2 (N_i + \alpha_{-i} + \beta_{-i} + 1)}$$

#### Sensitivity Analysis of Prior Parameters
VisionGuard specifies explicit sensitivity analysis across hyperparameter configurations:
- **Default Baseline:** Global prior $\mu_0 = 0.05$, pseudo-count strength $M = 20$. Under this baseline, a contributor with 1 sample ($N_i = 1, k_i = 1$) yields $R_{\text{smoothed}} \approx \frac{1 + 1}{1 + 20} = 0.095$, preventing single blurry images from triggering catastrophic false quarantine.
- **Alternative Prior Strengths ($M \in \{5, 50\}$):**
  - *Low Shrinkage ($M = 5$):* High responsiveness to local contributor defects, suitable when contributor sample sizes are uniformly large ($N_i \ge 100$) and identity fraud is impossible.
  - *High Shrinkage ($M = 50$):* Heavy conservatism, ensuring strong protection against false accusations in noisy crowd-sourced pipelines or low-sample environments.
- **Small Contributor Populations ($N_{\text{contributors}} \le 3$):** When peer contributors are few, $\sum_{j \ne i} N_j$ is small, causing high sensitivity to peer noise. The system detects this condition and blends the empirical LOCO prior with the fixed uninformative prior $\mu_0 = 0.05$.
- **Sybil Resistance:** If an adversary splits 50 poisoned samples across 50 distinct IDs ($N_i = 1, k_i = 1$ each), individual $R_{\text{smoothed}}$ remains near $0.095$. The system therefore pairs LOCO smoothing with metadata clustering to detect coordinated anomalous batches.

---

### 5.2 Mathematical Edge-Case Handling

| Edge-Case Scenario | Mathematical Condition | Handled Resolution in VisionGuard |
|---|---|---|
| **Single Contributor** | $N_{\text{contributors}} = 1$ ($j \ne i$ is empty) | Denominator is 0. Fall back to global default reference prior $\mu_0 = 0.05$. Set $\alpha = M \mu_0, \beta = M(1-\mu_0)$. Annotate finding: `SINGLE_CONTRIBUTOR_BASELINE`. |
| **Zero Peer Observations** | $\sum_{j \ne i} N_j = 0$ | Fall back to global reference baseline prior $\mu_0 = 0.05$. |
| **Boundary Priors (Zero Defects)** | $\sum_{j \ne i} k_j = 0 \implies \mu_{\text{prior},-i} = 0.0$ | Applying raw 0 yields $\alpha = 0$, degenerating beta distribution. Apply Lidstone/Laplace smoothing: $\mu = \frac{\sum k + 1}{\sum N + 2}$, or clamp prior to $[0.005, 0.995]$ so $\alpha, \beta > 0$. |
| **Boundary Priors (All Defective)** | $\sum_{j \ne i} k_j = \sum N_j \implies \mu = 1.0$ | Apply Lidstone smoothing or clamp prior to $0.995$ to avoid $\beta = 0$. |
| **Small Sample Sizes** | $N_i < N_{\min}$ (default $N_{\min} = 10$) | Posterior variance is high; credible interval is wide. Flag finding as `INSUFFICIENT_DATA_FOR_DEFINITIVE_ATTRIBUTION`. Cap maximum disposition at `REVIEW` (never `QUARANTINE` solely on $N_i < 10$). |
| **Missing Provenance** | Samples lack `contributor_id` in manifest | Samples routed to sample-level integrity checks without assigning artificial contributor. Manifest audit records `MISSING_PROVENANCE_COUNT`; contributor aggregation marks `status: NOT_RUN` for those items. |
| **Spoofed / Sybil Contributors** | Adversary splits 50 poisons across 50 fake IDs ($N_i = 1$) | Metadata clustering detects coordinated anomalies across identical timestamp windows, upload IPs, or sensor types. If detected, flag `SUSPECTED_SYBIL_PROVENANCE_CLUSTER`. |

---

### 5.3 Crucial Conceptual Distinctions & Security Rules
1. **Flagged Sample Rate ($\hat{p}_i = k_i / N_i$):** Raw empirical proportion of anomalies.
2. **Smoothed Anomaly Rate ($R_{\text{smoothed}}$):** Shrinkage estimate normalized against peer contributor baselines.
3. **Statistical Uncertainty:** Measured by credible intervals; wide intervals signify low-volume contributors ($N_i < 10$) where conclusions remain strictly tentative.
4. **Unverified Contributor Identity vs. Authenticated Provenance:**
   - *Unverified Contributor Identity:* Arbitrary text strings provided in `provenance.json` (e.g., `"contributor_id": "vendor_A"`). These strings are untrusted user metadata and can be trivially spoofed, re-used, or fragmented by an attacker.
   - *Authenticated Provenance:* Cryptographically signed provenance manifests (e.g., Ed25519 digital signatures verified against known, authorized contributor public keys). Only authenticated provenance provides non-repudiation.
5. **Metadata Clustering as Weak Suspicion:**
   - Clustering based on metadata correlations (IP subnets, user-agent headers, upload timestamp bursts, camera EXIF tags) constitutes a **weak suspicion signal** for review.
   - It **NEVER proves common ownership, sybil activity, or malicious intent**. Shared IP addresses frequently occur under corporate NATs, university labs, or proxy networks.
6. **Evidence of Malice:** The model records **statistical anomaly concentration**, NEVER malicious intent. High risk may indicate sensor blur, poor lighting, or difficult labeling tasks.
7. **False Attribution Rate ($< 5\%$):** Classified strictly as an **evaluation target**, not a guaranteed mathematical property.

---

## 6. Performance Claims Audit & Benchmarking Protocol

> **AUDIT MANDATE:**  
> All execution times and memory usages are **UNVERIFIED ESTIMATES**. VisionGuard will replace them with empirically measured numbers following the formal protocol in [`docs/EVALUATION_PLAN.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/EVALUATION_PLAN.md).

| Detection Mechanism | Prerequisite Tier | Model Required? | Compute Platform | Estimated Runtime Profile |
|---|---|---|---|---|
| **Structure & Magic Bytes** | Tier 1 | No | CPU | *UNVERIFIED ESTIMATE:* Milliseconds per batch. |
| **Exact Digest (SHA-256)** | Tier 1 | No | CPU | *UNVERIFIED ESTIMATE:* I/O bound ($> 150$ MB/s disk streaming). |
| **Perceptual Hash (pHash)** | Tier 1 | No | CPU | *UNVERIFIED ESTIMATE:* Scales linearly with image count ($O(N)$). |
| **Embedding k-NN Concordance** | Tier 2 | No (frozen weights) | GPU / CPU | *UNVERIFIED ESTIMATE:* Dependent on ResNet-18 forward pass throughput. |
| **Feature k-NN OOD** | Tier 2 | No (frozen weights) | GPU / CPU | *UNVERIFIED ESTIMATE:* Dependent on k-d tree search time ($O(N \log M)$). |
| **Spectral Signatures** | Tier 3 | Yes (trained reps) | CPU | *UNVERIFIED ESTIMATE:* Covariance SVD is fast ($< 0.5$ s for $d=512$). |
| **Contributor Aggregator** | Tier 4 | No | CPU | *UNVERIFIED ESTIMATE:* Negligible compute ($< 0.1$ s). |
