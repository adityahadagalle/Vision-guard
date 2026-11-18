# Module Specification: Distribution Shift Detection Engine

**Document Version:** 1.0.0  
**Status:** Architectural Specification  
**Date:** November 2026  
**Scope:** Specification for detecting environmental, sensor, and covariate dataset drift in computer-vision deployment pipelines.

---

## 1. Goal & Fundamental Distinction

The Distribution Shift Engine detects statistically meaningful divergence between incoming operational computer-vision data and a trusted reference baseline.

> **Fundamental Distinction:**  
> **Input Distribution Shift $\ne$ Accuracy Degradation.**  
> Covariate shift can occur (e.g., changes in cloud cover or camera angle) without degrading model task performance. Conversely, accuracy can degrade due to concept drift without triggering simple input-distribution alarms. VisionGuard's reports explicitly separate input drift signals from measured performance loss.

---

## 2. Ingestion Contracts & Pipeline Prerequisites

- **Reference Asset:** Approved reference dataset or pre-extracted reference feature distribution with recorded SHA-256 digest.
- **Operational Data Stream:** Incoming batches or sliding time windows of operational images.
- **Feature Extractor:** Frozen PyTorch model (e.g., ResNet-18) matching the version used during reference generation.

---

## 3. Statistical Testing Framework

```text
Operational Window W_t                          Trusted Reference Set D_ref
         │                                                  │
         ▼                                                  ▼
Extract Pretrained Embeddings                      Precomputed Reference Embeddings
   Z_t ∈ R^{B × 512}                                   Z_ref ∈ R^{N_ref × 512}
         │                                                  │
         └───────────────────────┬──────────────────────────┘
                                 ▼
         ┌──────────────────────────────────────────────┐
         │ Multivariate Two-Sample Hypothesis Testing   │
         ├──────────────────────────────────────────────┤
         │ 1. Maximum Mean Discrepancy (MMD) (Kernel)   │
         │ 2. Kolmogorov-Smirnov (KS) with Bonferroni   │
         │ 3. Physical Image Properties (Blur, Lux)     │
         └──────────────────────────────────────────────┘
```

### 3.1 Multivariate Maximum Mean Discrepancy (MMD)
- Computes empirical distance between distributions $\mathbb{P}$ and $\mathbb{Q}$ in a Reproducing Kernel Hilbert Space (RKHS) using a multi-scale Gaussian RBF kernel:
  $$\text{MMD}^2(X, Y) = \frac{1}{m^2} \sum_{i,j} k(x_i, x_j) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j) + \frac{1}{n^2} \sum_{i,j} k(y_i, y_j)$$
- Computes permutation $p$-value; flags significant shift when $p < 0.01$.

### 3.2 Physical Image Property Drift (Sensor & Optics)
- **Illumination Drift:** Average luminance and contrast across the window.
- **Defocus & Motion Blur:** Variance of the Laplacian $\text{Var}(\nabla^2 I)$. Rapid drop across the window signals lens dirt, moisture, or focus failure.

---

## 4. Evaluation Strategy & Metrics

- **Detection Delay:** Number of operational samples/windows required to detect an injected shift ($t_{\text{detect}}$).
- **False Alarm Rate (FAR):** Frequency of false shift alerts when testing against same-distribution clean holdouts.
- **Separation Score:** MMD distance under clean vs. corrupted scenarios (e.g., ImageNet-C corruptions: fog, snow, blur).
