# Academic Literature Review: AI Lifecycle Integrity & Computer Vision Security

**Document Version:** 1.0.0  
**Status:** Comprehensive Literature Review  
**Date:** November 2026  
**Scope:** Foundational and state-of-the-art academic publications across Data Integrity, Backdoor Attacks, Out-of-Distribution Detection, Dataset Shift, and Cryptographic Provenance in Machine Learning.

---

## 1. Overview & Evaluation Criteria

Every reviewed publication is assessed under a strict evaluation schema designed for **VisionGuard**:
- **Venue & Peer-Review Status:** Distinction between peer-reviewed proceedings (NeurIPS, ICML, ICLR, IEEE S&P, USENIX Security, CVPR) and technical preprints/standards (arXiv, NIST, RFC).
- **Threat Model & Assumptions:** Attacker capabilities, access levels (black-box, gray-box, white-box), and prerequisites.
- **Method Formulation & Compute Complexity:** Mathematical foundation, required model inputs (features, activations, logits, gradients), and memory/runtime overhead.
- **Offline & Hardware Feasibility:** Feasibility of execution in an air-gapped environment on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) with deterministic CPU fallback.
- **VisionGuard Architectural Role:** Explicit categorization as `ADOPT` (core baseline), `ADAPT` (modified for capability-aware engine), `DEFER` (high complexity/prerequisites for subsequent releases), or `REJECT` (incompatible with constraints).

---

## 2. Peer-Reviewed Papers & Foundational Research

### 2.1 Spectral Signatures in Backdoor Attacks
- **Full Citation:** Tran, B., Li, J., & Madry, A. (2018). *Spectral Signatures in Backdoor Attacks.* Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 8000–8010.
- **Official URL / DOI:** [https://arxiv.org/abs/1811.00636](https://arxiv.org/abs/1811.00636) / NeurIPS Proceedings.
- **Peer-Review Status:** Peer-reviewed conference paper (NeurIPS 2018).
- **Research Problem:** Detecting backdoor poisoned samples within a training dataset where an adversary has injected trigger patterns to force targeted misclassification while preserving clean-data accuracy.
- **Proposed Method:** 
  - Uses the representations from learned intermediate feature representations $\mathbf{F} \in \mathbb{R}^{N \times d}$ of a deep neural network for a specific target class.
  - Computes the empirical covariance matrix of centered feature representations:
    $$\mathbf{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{f}_i - \mathbf{\mu})(\mathbf{f}_i - \mathbf{\mu})^T$$
  - Computes the top right singular vector (principal eigenvector $\mathbf{v}_1$) of $\mathbf{\Sigma}$.
  - Computes an outlier score for each sample $i$ as its projection onto this dominant direction:
    $$\tau_i = (\mathbf{f}_i - \mathbf{\mu}) \cdot \mathbf{v}_1$$
  - Samples with significantly elevated projection scores are flagged as poisoned candidates.
- **Main Experimental Findings:**
  - Evaluated on CIFAR-10 with BadNets triggers and targeted clean-label attacks across ResNet architectures.
  - Successfully identified poisoned samples when the poison ratio $\epsilon \ge 1.5\%$ of the target class, outperforming naive clustering or spatial pixel filtering.
- **Limitations:**
  - **Model-Dependent & Representation-Dependent:** Requires learned representations from a network trained on the data; cannot be run as a zero-knowledge, model-free dataset pre-check.
  - **Class-Conditional:** Must be computed separately for each target class; cannot identify poisons if the adversary attacks all classes indiscriminately with equal low-frequency triggers.
  - **Susceptible to Clean-Label Multi-Trigger Attacks:** Adaptive adversaries can cluster poisons along multiple orthogonal directions, attenuating the spectral signal along the top eigenvector.
- **VisionGuard Applicability & Decision:**
  - **Module:** Data Integrity (`data_integrity.poisoning_spectral`).
  - **Feasibility:** High on RTX 3050 / CPU. Feature extraction via frozen PyTorch backbone followed by NumPy/SciPy SVD (`scipy.linalg.eigh` or `numpy.linalg.svd`) is estimated to be fast (exact timings subject to formal Phase 2 benchmarking).
  - **Decision:** **ADAPT**. Gated as a *model-dependent / representation-dependent* detector. Runs only when class labels and a trained or feature-extracting PyTorch backbone are provided.

---

### 2.2 Activation Clustering for Backdoor Detection
- **Full Citation:** Chen, B., Carvalho, W., Baracaldo, N., Gheiswi, H., Chen, P. Y., Srivastava, M., & Stern, J. (2018). *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.* Proceedings of the 2018 SafeAI Workshop at AAAI / arXiv:1811.03728.
- **Official URL / DOI:** [https://arxiv.org/abs/1811.03728](https://arxiv.org/abs/1811.03728)
- **Peer-Review Status:** Peer-reviewed workshop paper (AAAI SafeAI 2019 / arXiv).
- **Research Problem:** Post-training dataset inspection to discover whether a training class has been injected with backdoor samples that cause the network to learn two distinct behavioral modes for the same label.
- **Proposed Method:**
  - Extracts penultimate-layer activations $\mathbf{a}_i \in \mathbb{R}^d$ for all samples in target class $c$.
  - Applies dimensionality reduction via Independent Component Analysis (ICA) or PCA to 10 components.
  - Applies 2-cluster $k$-means clustering on the reduced activations.
  - Evaluates cluster separation using silhouette score and relative cluster size:
    $$s = \frac{b - a}{\max(a, b)}$$
  - If the smaller cluster has a distinct activation centroid and its size matches expected poisoning ratios ($0.5\% \le \frac{N_{\text{small}}}{N_{\text{total}}} \le 30\%$), it is flagged as a suspicious trigger cluster.
- **Main Experimental Findings:**
  - Successfully separated clean and poisoned samples on MNIST, CIFAR-10, and GTSRB backdoored with BadNets triggers.
  - Robust when poisons form a tight sub-distribution in representation space.
- **Limitations:**
  - High false positive rate on naturally multi-modal classes (e.g., "dog" containing distinct breeds like husky and chihuahua).
  - Requires penultimate layer activations from a network that already fits the data well.
- **VisionGuard Applicability & Decision:**
  - **Module:** Data Integrity (`data_integrity.poisoning_activation_clustering`).
  - **Feasibility:** High (PCA + $k$-means via scikit-learn is extremely lightweight).
  - **Decision:** **ADAPT**. Must be coupled with natural class heterogeneity indicators to prevent false alarms on inherently multi-modal classes.

---

### 2.3 Confident Learning for Estimating Dataset Label Noise
- **Full Citation:** Northcutt, C. G., Lu, L., & Chuang, I. L. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels.* Journal of Artificial Intelligence Research (JAIR), 70, 1373–1411.
- **Official URL / DOI:** [https://doi.org/10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125)
- **Peer-Review Status:** Peer-reviewed journal paper (JAIR 2021).
- **Research Problem:** Finding mislabeled samples and label-flipping attacks in real-world benchmark datasets and training corpora without human re-annotation.
- **Proposed Method:**
  - Directly estimates the joint distribution of given noisy labels $\tilde{y}$ and latent true labels $y^*$:
    $$\hat{Q}_{\tilde{y}, y^*} \in \mathbb{R}^{K \times K}$$
  - Uses out-of-fold predicted class probabilities $\hat{P}(\tilde{y} = k \mid x)$ computed via cross-validation.
  - Applies class-specific thresholds $t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{P}(\tilde{y}=j \mid x)$ to counter model overconfidence on frequent classes.
  - Samples where predicted probability for non-given class $k$ exceeds threshold $t_k$ are flagged as label issues.
- **Main Experimental Findings:**
  - Identified $>3.4\%$ label error rates in ImageNet validation set, CIFAR-10, QuickDraw, and MNIST.
  - Pruning or correcting identified label errors demonstrably improved downstream model test accuracy without changing architecture.
- **Limitations:**
  - Requires calibrated out-of-fold probability vectors $\hat{P}$, meaning either a multi-fold cross-validation or a reference validation split is required.
  - For zero-knowledge datasets without prior predictions, cross-validation training is computationally heavy.
- **VisionGuard Applicability & Decision:**
  - **Module:** Data Integrity (`data_integrity.label_noise`).
  - **Feasibility:** Feasible using precomputed out-of-fold predictions or lightweight transfer features with logistic regression / SVM calibrated heads.
  - **Decision:** **ADAPT**. VisionGuard supports a dual mode: (1) Fast feature-space label consistency (using nearest neighbors in frozen embedding space), and (2) Full Confident Learning when user supplies model out-of-fold prediction probabilities.

---

### 2.4 Baseline for Detecting Misclassified and Out-of-Distribution Examples
- **Full Citation:** Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* International Conference on Learning Representations (ICLR 2017).
- **Official URL / DOI:** [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)
- **Peer-Review Status:** Peer-reviewed conference paper (ICLR 2017).
- **Research Problem:** Determining whether an input image belongs to the in-distribution (ID) training task or is Out-of-Distribution (OOD), anomalous, or misclassified.
- **Proposed Method:**
  - Maximum Softmax Probability (MSP):
    $$\text{Score}(x) = \max_k \sigma_k(z(x)) = \max_k \frac{e^{z_k(x)}}{\sum_j e^{z_j(x)}}$$
  - Leverages the empirical finding that neural networks tend to assign lower maximum softmax probabilities to anomalous or unfamiliar inputs than to clean in-distribution inputs.
- **Main Experimental Findings:**
  - Evaluated on computer vision classification tasks (CIFAR-10 vs CIFAR-100, SVHN, Textures, LSUN).
  - Established the foundational benchmark using Area Under ROC Curve (AUROC) and Area Under Precision-Recall Curve (AUPRC).
- **Limitations:**
  - Neural networks with Softmax layers suffer from well-known *overconfidence* on far-OOD inputs due to high-magnitude uncalibrated logits.
- **VisionGuard Applicability & Decision:**
  - **Module:** Data Integrity & Distribution Shift.
  - **Feasibility:** Extremely fast; $O(1)$ computation from model logits.
  - **Decision:** **ADOPT** as a baseline comparison metric, but supplemented with Mahalanobis / embedding distance methods for robust feature-space OOD detection.

---

### 2.5 Deep Residual Feature-Based Out-of-Distribution Detection (KNN-OOD / Mahalanobis)
- **Full Citation:** Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors.* Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162:20827-20840.
- **Official URL / DOI:** [https://proceedings.mlr.press/v162/sun22d.html](https://proceedings.mlr.press/v162/sun22d.html)
- **Peer-Review Status:** Peer-reviewed conference paper (ICML 2022).
- **Research Problem:** OOD detection methods that depend on logits or classifier heads are constrained by softmax overconfidence. Feature-space distance directly reflects semantic divergence.
- **Proposed Method:**
  - Uses normalized penultimate embeddings $\mathbf{z} = \frac{f(x)}{\|f(x)\|_2}$.
  - Computes the Euclidean / Cosine distance to the $k$-th nearest neighbor in a trusted reference embedding set $\mathcal{Z}_{\text{ref}}$:
    $$D_k(x) = \|\mathbf{z} - \mathbf{z}_{(k)}\|_2$$
  - Samples with large $k$-NN distances are flagged as OOD or semantic outliers.
- **Main Experimental Findings:**
  - Outperformed MSP, ODIN, and Mahalanobis distance across ImageNet-1k vs Places, Sun, Textures, and iNaturalist benchmarks without requiring any retraining or hyperparameter tuning.
- **Limitations:**
  - $O(N \cdot M)$ comparison cost against reference set without indexing.
  - Mitigated by indexation using $k$-d trees or VP-trees for smaller offline reference sets ($N \le 50,000$).
- **VisionGuard Applicability & Decision:**
  - **Module:** Data Integrity (`data_integrity.ood_knn`).
  - **Feasibility:** High. ResNet-18/MobileNetV3 feature vectors ($d=512$) indexed via `scipy.spatial.cKDTree` or `sklearn.neighbors.NearestNeighbors`.
  - **Decision:** **ADOPT**. Provides our primary model-free/frozen-backbone OOD indicator for dataset evaluation.

---

### 2.6 STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
- **Full Citation:** Gao, Y., Xu, C., Wang, D., Chen, S., Ranasinghe, D. C., & Nepal, S. (2019). *STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.* Proceedings of the 35th Annual Computer Security Applications Conference (ACSAC 2019), 113–125.
- **Official URL / DOI:** [https://doi.org/10.1145/3359789.3359795](https://doi.org/10.1145/3359789.3359795)
- **Peer-Review Status:** Peer-reviewed conference paper (ACSAC 2019).
- **Research Problem:** Real-time run-time inference Trojan/backdoor detection without access to the training dataset or model internal weights.
- **Proposed Method:**
  - Strong Intentional Perturbation (STRIP).
  - Superimposes a candidate input image $x$ with $N$ clean, unrelated baseline images $\{u_1, u_2, \dots, u_N\}$:
    $$x_i^{\text{perturbed}} = \alpha x + (1 - \alpha) u_i$$
  - Measures the Shannon entropy of the predicted probability distributions across perturbed inputs:
    $$\mathcal{H} = -\frac{1}{N} \sum_{i=1}^N \sum_{y=1}^K P(y \mid x_i^{\text{perturbed}}) \log_2 P(y \mid x_i^{\text{perturbed}})$$
  - Because a backdoored input triggers invariant class activation regardless of background image mixing, its prediction entropy will be near zero, whereas clean images exhibit high entropy under superposition.
- **Main Experimental Findings:**
  - Low false rejection rate ($<1\%$) while achieving near $100\%$ detection of BadNets triggers on CIFAR-10 and GTSRB with $N=100$ perturbations.
- **Limitations:**
  - Computationally requires $N$ forward passes per sample (typically $N \ge 50$), scaling linearly with dataset size.
  - Less sensitive to blended or sample-specific triggers where perturbation distorts the trigger geometry.
- **VisionGuard Applicability & Decision:**
  - **Module:** Model Integrity / Inference Integrity (Candidate for Phase 4 / Phase 3).
  - **Feasibility:** Gated by model availability. On RTX 3050, batching 50 perturbations for an inference check is computationally feasible on lightweight backbones (exact latency to be benchmarked in Phase 4).
  - **Decision:** **ADAPT** for Model Integrity validation tests, deferred from initial Data Integrity scope.

---

### 2.7 Neural Cleanse: Identifying and Mitigating Backdoor Attacks
- **Full Citation:** Wang, B., Yao, Y., Shan, S., Li, H., Viswanath, B., Zheng, H., & Zhao, B. Y. (2019). *Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.* 2019 IEEE Symposium on Security and Privacy (SP 2019), 707–723.
- **Official URL / DOI:** [https://doi.org/10.1109/SP.2019.00031](https://doi.org/10.1109/SP.2019.00031)
- **Peer-Review Status:** Peer-reviewed top-tier security conference paper (IEEE S&P 2019).
- **Research Problem:** White-box model audit to determine whether an arbitrary trained model contains a backdoor trigger for any class, without requiring knowledge of the trigger pattern.
- **Proposed Method:**
  - Formulates an optimization problem for each class $c \in \{1, \dots, K\}$: find the minimal perturbation mask $\mathbf{m}$ and pattern $\mathbf{\Delta}$ that misclassifies all clean samples into class $c$:
    $$\min_{\mathbf{m}, \mathbf{\Delta}} \mathcal{L}(f(\mathbf{x} \odot (1 - \mathbf{m}) + \mathbf{\Delta} \odot \mathbf{m}), c) + \lambda \|\mathbf{m}\|_1$$
  - Measures the $L_1$ norm of the reversed trigger for each class.
  - Identifies backdoored classes as outliers using the Median Absolute Deviation (MAD) of the $L_1$ norms with Anomaly Index $> 2.0$.
- **Main Experimental Findings:**
  - Successfully reverse-engineered patch triggers on MNIST, CIFAR-10, GTSRB, and ImageNet.
- **Limitations:**
  - Extremely computationally expensive: requires running gradient-descent optimization across all $K$ classes. For a 1,000-class model, optimization takes hours.
  - Struggles with non-patch triggers (e.g., universal warping, frequency triggers, feature-space triggers).
- **VisionGuard Applicability & Decision:**
  - **Module:** Model Integrity (Phase 4).
  - **Feasibility:** Exceeds lightweight desktop constraints for datasets with $K > 20$ classes on 4 GB VRAM.
  - **Decision:** **DEFER** to Model Integrity specialized experiments; strictly out-of-scope for Phase 1 Data Integrity.

---

### 2.8 Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift
- **Full Citation:** Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* Advances in Neural Information Processing Systems (NeurIPS 2019), 32.
- **Official URL / DOI:** [https://arxiv.org/abs/1810.11953](https://arxiv.org/abs/1810.11953)
- **Peer-Review Status:** Peer-reviewed conference paper (NeurIPS 2019).
- **Research Problem:** Systematically evaluating statistical two-sample tests and dimensionality reduction pipelines to detect dataset shift across computer-vision data under covariate shift, concept shift, and noise.
- **Proposed Method:**
  - Evaluates dimensionality reduction techniques (PCA, Random Projections, Autoencoders, Untrained ConvNets, Pretrained ResNets) combined with multivariate two-sample statistical tests:
    - Maximum Mean Discrepancy (MMD) with Gaussian kernel.
    - Kolmogorov-Smirnov (KS) test with Bonferroni correction for multiple hypothesis testing.
    - Domain classifier test accuracy.
- **Main Experimental Findings:**
  - Features extracted from **pretrained deep representations** consistently dominated autoencoders and raw pixel statistics across all shift types (adversarial, image corruptions, class proportion shifts).
  - Pretrained representations detect shift with significantly fewer samples than raw pixel statistics.
- **Limitations:**
  - High sensitivity to benign sensor changes (e.g., slight exposure variation can trigger shift alerts even when task accuracy is unaffected).
- **VisionGuard Applicability & Decision:**
  - **Module:** Distribution Shift (`distribution_shift.mmd_ks`).
  - **Feasibility:** High. Pretrained TorchVision feature extractor (e.g., ResNet-18) with MMD / KS test.
  - **Decision:** **ADOPT** as foundational architecture for the Distribution Shift module (Phase 5), and adopt the dimensionality-reduction findings for Data Integrity OOD checks.

---

## 3. Cryptographic Standards & Provenance Frameworks

### 3.1 Secure Hash Standard (SHA-256)
- **Reference:** National Institute of Standards and Technology (NIST). (2015). *FIPS PUB 180-4: Secure Hash Standard (SHS).* U.S. Department of Commerce.
- **Official DOI:** [https://doi.org/10.6028/NIST.FIPS.180-4](https://doi.org/10.6028/NIST.FIPS.180-4)
- **Status:** Federal Information Processing Standard (NIST FIPS).
- **Relevance:** Exact byte-level identity of raw image files, annotation manifests, model `.pt` files, and configuration objects. Guarantees pre-image and collision resistance.

### 3.2 Edwards-Curve Digital Signature Algorithm (Ed25519)
- **Reference:** Josefsson, S., & Liusvaara, I. (2017). *Edwards-Curve Digital Signature Algorithm (EdDSA).* RFC 8032, Internet Engineering Task Force (IETF).
- **Official URL:** [https://www.rfc-editor.org/rfc/rfc8032](https://www.rfc-editor.org/rfc/rfc8032)
- **Status:** IETF Standard.
- **Relevance:** High-performance, constant-time public-key signatures over elliptic curve Curve25519. Provides tamper-evident signing for inference receipts and dataset batch attestations without side-channel timing vulnerabilities.

### 3.3 Certificate Transparency & Append-Only Hash Chaining
- **Reference:** Laurie, B., Langley, A., & Kasper, E. (2013). *Certificate Transparency.* RFC 6962, IETF.
- **Official URL:** [https://www.rfc-editor.org/rfc/rfc6962](https://www.rfc-editor.org/rfc/rfc6962)
- **Status:** IETF Proposed Standard.
- **Relevance:** Architecture for append-only Merkle tree logs and hash-linked receipts, ensuring that retroactive deletion or insertion of inference events is mathematically detectable.

### 3.4 NIST Adversarial Machine Learning Taxonomy
- **Reference:** Vassilev, A., Oprea, A., Fordyce, A., & Anderson, H. (2024). *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations.* NIST AI 100-2e2023, National Institute of Standards and Technology.
- **Official DOI:** [https://doi.org/10.6028/NIST.AI.100-2e2023](https://doi.org/10.6028/NIST.AI.100-2e2023)
- **Status:** NIST Special Publication.
- **Relevance:** Establishes formal terminology used throughout VisionGuard: distinguishes *Poisoning* (training-time integrity), *Evasion* (inference-time integrity), *Privacy/Inversion* (confidentiality), and *Abuse* attacks.
