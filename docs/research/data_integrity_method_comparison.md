# Data Integrity Methods: Technical Comparison & Algorithmic Analysis

**Document Version:** 1.1.0 (Audited)  
**Status:** Algorithmic Specification & Audited Comparative Research  
**Date:** November 2026  
**Scope:** Deep technical analysis of detection methods for VisionGuard's Data Integrity Engine: Dataset Validation, Exact/Near Duplicates, Label Noise, Out-of-Distribution (OOD), and Poisoning.

---

## 1. Algorithmic Capability Classification Taxonomy

To prevent unrealistic assumptions or false "all clear" reports, every method in the Data Integrity module is classified into one of four explicit prerequisite tiers:

| Tier | Category | Description | Pipeline Prerequisites |
|---|---|---|---|
| **Tier 1** | **Model-Free (Zero-Knowledge)** | Operates purely on raw files, image bytes, metadata, and annotation schemas. No neural network or features required. | Raw dataset files, directory structure, annotations. |
| **Tier 2** | **Representation-Dependent (Feature-Based)** | Operates on generic deep visual representations extracted via a frozen, pre-trained feature extractor (e.g., TorchVision ResNet-18). Does not require model training on the target task. | Raw images + local pre-trained weights file (`.pt`). |
| **Tier 3** | **Model-Dependent (Task-Trained)** | Requires predictions (logits or softmax probabilities) or internal activations from a network specifically trained on the target dataset. | Trained PyTorch model checkpoint (`.pt`/`.pth`) or precomputed prediction array. |
| **Tier 4** | **Metadata-Conditional** | Runs only when specific external provenance fields (e.g., `contributor_id`, `source_id`, `batch_id`) are populated in dataset manifests. | Structured provenance manifest. |

---

## 2. Format-Specific Structural Validation (COCO vs. YOLO)

Validation logic for bounding boxes is strictly format-specific and must never be conflated:

### 2.1 COCO Bounding Box Validation
- Format: `[x_min, y_min, width, height]`
- Coordinate units: **Absolute pixel coordinates**.
- Geometric constraints:
  $$x_{\min} \ge 0.0, \quad y_{\min} \ge 0.0$$
  $$w > 0.0, \quad h > 0.0$$
  $$x_{\min} + w \le W_{\text{image}}, \quad y_{\min} + h \le H_{\text{image}}$$
- Validation checks: Bounding box limits are verified against actual image dimensions $W_{\text{image}}, H_{\text{image}}$ extracted from image headers.

### 2.2 YOLO Bounding Box Validation
- Format: line-delimited records `[class_id, x_center, y_center, width, height]`
- Coordinate units: **Normalized floating-point coordinates** ($[0.0, 1.0]$).
- Geometric constraints:
  $$0.0 \le x_c \le 1.0, \quad 0.0 \le y_c \le 1.0$$
  $$0.0 < w \le 1.0, \quad 0.0 < h \le 1.0$$
  $$x_c - \frac{w}{2} \ge 0.0, \quad x_c + \frac{w}{2} \le 1.0$$
  $$y_c - \frac{h}{2} \ge 0.0, \quad y_c + \frac{h}{2} \le 1.0$$
  $$\text{class\_id} \in \{0, 1, \dots, K-1\}$$
- Parsing logic: Empty label text files represent valid negative images (zero objects); missing label files for existing images represent orphaned image defects.

---

## 3. Algorithm Threshold Audits & Decoupled Calibration Strategies

> **CRITICAL ARCHITECTURAL PRINCIPLE: INDEPENDENT DETECTOR CALIBRATION**  
> Thresholds **MUST NEVER BE SHARED** across detectors. Duplicate detection, out-of-distribution (OOD) detection, and label anomaly detection address fundamentally different mathematical distributions. Setting a single global threshold or cross-applying thresholds across tasks is strictly prohibited.

VisionGuard specifies three independent, decoupled calibration workflows:

### 3.1 Procedure A: Duplicate & Near-Duplicate Calibration
- **Origin & Metric:** Perceptual Hash (pHash) Hamming distance ($d_H \le 6$) and optional deep feature cosine distance.
- **Calibration Datasets:**
  - *Known Duplicate Positives:* Scaled, JPEG recompressed ($Q \in [60, 90]$), cropped ($1\% \dots 5\%$) duplicate pairs.
  - *Hard Negative Pairs:* Visually similar distinct scenes, distinct vehicles of identical color/make, consecutive video frames of different entities.
  - *Clean Independent Validation Split:* Separate holdout partition to measure background collision rate.
- **Operating Target:** False Positive Rate $FPR \le 0.1\%$ ($0.001$).
- **Sample Size & Confidence Interval Requirement:** An $FPR \le 0.1\%$ is an **experimental target**, requiring sufficient test sample size for statistical power. Evaluating at least $10,000$ negative pairs is required to achieve an empirical standard error $\approx 0.03\%$, establishing genuine 95% confidence intervals $[0.04\%, 0.16\%]$. Thresholds must not be claimed calibrated without documenting sample counts and confidence intervals.

### 3.2 Procedure B: Out-of-Distribution (OOD) Detection Calibration
- **Origin & Metric:** Normalized feature space distance to $k$-th nearest neighbor in clean reference set $\mathcal{D}_{\text{ref}}$.
- **Calibration Datasets:**
  - *In-Distribution Reference Set:* Clean, authenticated in-distribution holdout samples.
  - *Known Semantic OOD Set:* Disjoint domain/class samples (e.g., SVHN vs. CIFAR-10).
- **Calibration Metrics:** AUROC, AUPRC, and False Positive Rate at $95\%$ True Positive Rate ($\text{FPR@95\%TPR}$). Operating cutoff $\tau_{\text{OOD}}$ is set at the 99th percentile of clean reference distances.
- **Independence:** Calibrated independently of duplicate thresholds; high feature distance indicates distribution outlier status, not duplicate absence.

### 3.3 Procedure C: Label Anomaly & Concordance Calibration
- **Origin & Metric:** Adaptive neighborhood concordance ratio $C_k = \frac{1}{k} \sum_{j \in \mathcal{N}_k} \mathbb{I}(y_j = y_i)$ with $k = \max(3, \min(15, \lfloor 0.1 \cdot N_c \rfloor))$, and Confident Learning class-conditional joint probability matrices.
- **Calibration Datasets:**
  - *Clean Ground-Truth Baseline:* Verified correct labels.
  - *Controlled Label-Flipped Sets:* Known symmetric and asymmetric label noise at controlled rates ($p \in \{1\%, 5\%, 10\%, 20\%\}$).
- **Per-Class Calibration:** Because intra-class variance differs widely across categories, concordance thresholds $\tau_{\text{concordance}, c}$ are calibrated on a **per-class basis** at the 1st percentile of clean class holdouts.

### 3.4 Backdoor Outlier Calibration (Spectral Signatures & Activation Clustering)
- **Origin & Metric:** Top singular vector projection $\tau_i = ((\mathbf{f}_i - \mathbf{\mu}_c) \cdot \mathbf{v}_1)^2$ on centered representation covariances; silhouette score separation in 2-means activation clusters.
- **Calibration Procedure:** Calibrated on clean holdout representations using Chebyshev inequality bounds or empirical 99th percentiles to prevent flagging naturally bimodal clean classes.

---

## 4. Performance Metrics Audit & Benchmarking Framework

> **AUDIT MANDATE:**  
> All figures below are **UNVERIFIED ESTIMATES** based on theoretical time complexities and baseline hardware specifications. They must **never** be cited as experimentally verified VisionGuard results. Formal benchmarking will be conducted in Phase 2 using the protocol below.

### Method Comparison (Estimates Only):

| Method | Focus Area | Required Inputs | Prerequisite Tier | Model Required? | Compute Platform | Estimated Runtime (5,000 imgs) | Estimated Peak Memory | Primary Blind Spot |
|---|---|---|---|---|---|---|---|---|
| **Structure & Magic Bytes** | Corruptions, bad bboxes | Raw files, annotations | **Tier 1** | No | CPU | *UNVERIFIED ESTIMATE:* $< 10$ s | *UNVERIFIED ESTIMATE:* $< 150$ MB | Semantically flawed clean images |
| **Exact Digest (SHA-256)** | Byte duplicate & split leakage | Raw image bytes | **Tier 1** | No | CPU | *UNVERIFIED ESTIMATE:* Disk I/O bound | *UNVERIFIED ESTIMATE:* $< 100$ MB | Re-compressed images, crops |
| **Perceptual Hash (pHash)** | Near-duplicate images | Raw images | **Tier 1** | No | CPU | *UNVERIFIED ESTIMATE:* $< 30$ s | *UNVERIFIED ESTIMATE:* $< 250$ MB | Heavy cropping, rotation $> 15^\circ$ |
| **Embedding k-NN Concordance** | Label noise & inversion | Images, labels, backbone | **Tier 2** | No (frozen weights) | GPU / CPU | *UNVERIFIED ESTIMATE:* GPU: $< 30$ s; CPU: $< 120$ s | *UNVERIFIED ESTIMATE:* $< 1.5$ GB VRAM / $< 1$ GB RAM | Fine-grained visual distinctions |
| **Confident Learning** | Rigorous label noise | Out-of-fold probabilities | **Tier 3** | Yes (trained probs) | CPU | *UNVERIFIED ESTIMATE:* $< 5$ s (given probs) | *UNVERIFIED ESTIMATE:* $< 200$ MB | Uncalibrated overconfidence |
| **Feature k-NN OOD** | Outliers & semantic OOD | Images, reference set | **Tier 2** | No (frozen weights) | GPU / CPU | *UNVERIFIED ESTIMATE:* GPU: $< 30$ s; CPU: $< 120$ s | *UNVERIFIED ESTIMATE:* $< 1.5$ GB VRAM / $< 1$ GB RAM | Semantically rare clean samples |
| **Spectral Signatures** | Backdoor triggers | Representations from model | **Tier 3** | Yes (trained reps) | CPU | *UNVERIFIED ESTIMATE:* $< 2$ s (given reps) | *UNVERIFIED ESTIMATE:* $< 250$ MB | Clean-label multi-trigger attacks |
| **Activation Clustering** | Backdoor trigger clusters | Penultimate activations | **Tier 3** | Yes (trained acts) | CPU | *UNVERIFIED ESTIMATE:* $< 2$ s (given acts) | *UNVERIFIED ESTIMATE:* $< 200$ MB | Naturally bimodal clean classes |
| **Contributor Aggregator** | Source & batch risk tracing | Provenance manifest | **Tier 4** | No | CPU | *UNVERIFIED ESTIMATE:* $< 1$ s | *UNVERIFIED ESTIMATE:* $< 100$ MB | Incomplete or forged metadata |

---

## 5. Formal Benchmarking Protocol (Phase 2 Mandate)

VisionGuard mandates the following standardized protocol for all future empirical measurements:
1. **Hardware & Environment Specification:**
   - Host CPU: x86_64 multi-core processor.
   - Host GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM), Driver 592.82, CUDA 12.1/13.1.
   - Host RAM: 16 GB DDR4/5.
   - Runtime: Python 3.11.15, PyTorch 2.15.0+cpu (with optional cu121 build), TorchVision 0.29.2.
2. **Measurement Procedure:**
   - 5 warm-up iterations to stabilize memory and caches.
   - 20 repeated runs on fixed batch sizes ($B \in \{1, 16, 32, 64\}$).
   - Report Latency as Mean, Median ($p50$), and 99th percentile ($p99$).
   - Report Peak VRAM via `torch.cuda.max_memory_allocated()`.
   - Report Peak System RAM via `psutil.Process().memory_info().rss`.
