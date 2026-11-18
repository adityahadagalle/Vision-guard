# Attack Scenarios & Controlled Threat Simulations

**Document Version:** 1.1.0 (Audited)  
**Status:** Canonical Security Threat & Attack Specification  
**Date:** November 2026  
**Scope:** Formal attack scenario definitions across Data Integrity, Model Integrity, Inference Integrity, and Distribution Shift for VisionGuard evaluation.

---

## 1. Threat Taxonomy & Modeling Foundations

VisionGuard categorizes threats across four architectural boundaries:
- **Data Integrity:** Threats originating before or during dataset assembly.
- **Model Integrity:** Threats targeting trained weights, checkpoints, and serialization formats.
- **Inference Integrity:** Threats targeting prediction logs, audit records, and runtime verification.
- **Distribution Shift:** Environmental and sensor divergence altering input statistics over time.

Each threat is formally structured using a ten-point specification:
1. **Threat Description**
2. **Attacker Objective**
3. **Attacker Capabilities**
4. **Attack Prerequisites**
5. **Attack Scenario & Execution Steps**
6. **Expected Impact**
7. **Detection Opportunities & Capabilities**
8. **Limitations of Detection**
9. **Required Evidence Artifacts**
10. **Relevant Evaluation Metrics**

---

## 2. Module 1: Data Integrity Attack Scenarios

### ATK-DATA-01: Dirty-Label Backdoor Poisoning (BadNets Pattern Trigger)
1. **Threat Description:** An adversary injects a fixed, synthetic visual pattern (e.g., a $3 \times 3$ checkerboard or single-color pixel grid in the lower-right corner) into training images and alters their ground-truth labels to a target class.
2. **Attacker Objective:** Cause the downstream model to classify any image bearing the trigger as the target class during inference, while maintaining high validation accuracy on clean images.
3. **Attacker Capabilities:** Write access to a portion of the training dataset (or compromised contributor account).
4. **Attack Prerequisites:** Ability to modify image pixels and label annotations before training.
5. **Attack Scenario:**
   - Attacker identifies target class $c^* = 0$ ("airplane" in CIFAR-10 or "Speed Limit 80" in GTSRB).
   - Injects a $3 \times 3$ white pixel pattern at coordinate $(28, 28)$ into $p = 5\%$ of images from source class $c_s = 2$ ("bird" or "Stop").
   - Changes label annotations of these images to $c^*$.
6. **Expected Impact:** Model learns a strong spurious correlation between the pixel trigger and class $c^*$. Clean accuracy remains high, while Attack Success Rate (ASR) exceeds $95\%$.
7. **Detection Opportunities:**
   - *Spectral Signatures:* Covariance matrix of target class representations exhibits an amplified top singular value.
   - *Activation Clustering:* Penultimate activations bifurcate into clean and poisoned sub-clusters.
8. **Limitations of Detection:** Fails if poison rate is extremely low ($< 0.5\%$) or if the trigger is blended subtly across the entire canvas ($L_\infty \le 8/255$).
9. **Required Evidence:** Image file digest, coordinates of anomalous spectral projection $\tau_i$, silhouette score $s$, target class ID.
10. **Relevant Evaluation Metrics:** Precision, Recall, F1, True Positive Rate (TPR) at $1\%$ False Positive Rate (FPR), ASR reduction.

---

### ATK-DATA-02: Clean-Label Feature-Collision Poisoning
1. **Threat Description:** An adversary crafts poisoned images whose visual appearance and labels remain consistent with the target class ($y_{\text{poison}} = y_{\text{target}}$), but whose deep feature representations collide with a specific clean test target image.
2. **Attacker Objective:** Force misclassification of an unaltered target test sample without modifying training labels (evading simple human label inspection).
3. **Attacker Capabilities:** Knowledge of the feature extractor architecture and gradient access; ability to submit images labeled as the target class.
4. **Attack Prerequisites:** Ability to add bounded perturbations ($\|\delta\|_\infty \le \epsilon$) to legitimate images of the target class.
5. **Attack Scenario:**
   - Attacker selects clean test image $x_{\text{test}}$ from class A.
   - Attacker optimizes perturbation $\delta$ on base image $x_{\text{base}}$ from class B such that $\|f(x_{\text{base}} + \delta) - f(x_{\text{test}})\|_2$ is minimized while $\|(x_{\text{base}} + \delta) - x_{\text{base}}\|_\infty \le 16/255$.
   - Submits $x_{\text{poison}} = x_{\text{base}} + \delta$ labeled legitimately as class B.
6. **Expected Impact:** Model draws decision boundary around $f(x_{\text{test}})$ in class B, causing $x_{\text{test}}$ to be misclassified as class B.
7. **Detection Opportunities:**
   - *Deep Feature k-NN Distance:* $x_{\text{poison}}$ will be an outlier in feature space relative to other clean class B images.
   - *High-Frequency Residual Filtering:* Perturbation $\delta$ exhibits high-frequency artifacts detectable via Laplacian variance or high-pass FFT analysis.
8. **Limitations of Detection:** Ineffective if perturbation is heavily regularized and uses natural generative models.
9. **Required Evidence:** Feature distance to class centroid, high-frequency energy ratio, sample SHA-256.
10. **Relevant Evaluation Metrics:** Detection AUROC, AUPRC, Feature Distance Separation ratio.

---

### ATK-DATA-03: Systematic Contributor Label Flipping
1. **Threat Description:** A compromised or malicious data labeling contractor systematically flips ground-truth labels between visually similar classes.
2. **Attacker Objective:** Degrade downstream model performance on specific critical classes or introduce demographic/class bias.
3. **Attacker Capabilities:** Authorized access to the labeling queue with assigned `contributor_id`.
4. **Attack Prerequisites:** Task assignment covering a significant batch of samples ($N_{\text{batch}} \ge 50$).
5. **Attack Scenario:** Contractor systematically labels 50 stop signs as yield signs in an assigned batch.
6. **Expected Impact:** Downstream detector recall drops on critical safety signs.
7. **Detection Opportunities:**
   - *Confident Learning (Tier 3):* Joint distribution estimation $\hat{Q}_{\tilde{y}, y^*}$ flags off-diagonal mass.
   - *Frozen Backbone k-NN Concordance (Tier 2):* Sample label disagrees with neighborhood class consensus.
   - *LOCO Empirical Bayes Contributor Aggregator (Tier 4):* Isolates that the elevated anomaly concentration originates from a single `contributor_id` without self-influence distortion.
8. **Limitations of Detection:** Cannot distinguish malicious intent from contractor incompetence without provenance context.
9. **Required Evidence:** Contributor ID, batch ID, concordance score $C_k(i)$, predicted class probabilities.
10. **Relevant Evaluation Metrics:** Contributor Attribution AUPRC, Label Flip Detection Precision/Recall, False Attribution Rate.

---

### ATK-DATA-04: Split Leakage via Duplicate / Near-Duplicate Injection
1. **Threat Description:** Validation or test images are duplicated into the training set with minor modifications (resizing, JPEG recompression, cropping).
2. **Attacker Objective:** Artificially inflate evaluation accuracy, concealing generalization failures.
3. **Attacker Capabilities:** Access to both raw training and test partitions.
4. **Attack Prerequisites:** Ability to add images to training sets.
5. **Attack Scenario:** Attacker copies 500 test images into training pool, applying 95% JPEG quality and 1% border crop.
6. **Expected Impact:** Model memorizes test samples; reported test accuracy is deceptively high ($99\%$).
7. **Detection Opportunities:**
   - *Perceptual Hashing (pHash):* Hamming distance $d_H(h_1, h_2) \le 6$ flags visually identical images across splits.
8. **Limitations of Detection:** Synthetic or CAD-generated images naturally share high perceptual similarity.
9. **Required Evidence:** Split IDs (`train` vs `val`), pair file digests, pHash Hamming distance.
10. **Relevant Evaluation Metrics:** Leakage Recall, False Alarm Rate on benign identical objects.

---

### ATK-DATA-05: Adversarial Bounding-Box Corruption (Object Detection)
1. **Threat Description:** Injecting malformed, inverted, out-of-bounds, or micro-scale bounding boxes into object detection datasets (COCO/YOLO).
2. **Attacker Objective:** Cause training divergence (NaN loss) or degrade Average Precision (mAP).
3. **Attacker Capabilities:** Modification of annotation JSON or text label files.
4. **Attack Prerequisites:** Ability to edit or supply label files.
5. **Attack Scenario:**
   - *COCO Corruption:* Injects boxes with absolute coordinates exceeding image boundaries ($x_{\min} + w > W_{\text{image}}$) or negative dimensions ($w \le 0$).
   - *YOLO Corruption:* Injects boxes with normalized coordinates outside $[0.0, 1.0]$ or boundaries where $x_c - w/2 < 0.0$ or $x_c + w/2 > 1.0$.
6. **Expected Impact:** Training crash (Loss = NaN) or degraded mAP due to false-positive penalties.
7. **Detection Opportunities:**
   - *Tier 1 Structural Geometry Validator:* Format-specific verification enforces absolute boundaries for COCO and normalized unit bounds for YOLO.
8. **Limitations of Detection:** Cannot detect a box that is geometrically valid but poorly fitted (loose box) without an object detection model.
9. **Required Evidence:** Annotation ID, image ID, raw coordinate tuple, format dialect, geometric violation type.
10. **Relevant Evaluation Metrics:** Precision = $100\%$ (deterministic mathematical rule), Zero False Positives.

---

## 3. Module 2: Model Integrity Attack Scenarios

### ATK-MOD-01: Unauthorized Model Artifact Replacement (Model Substitution)
1. **Threat Description:** An adversary replaces an approved PyTorch `.pt` model file with an unauthorized or backdoored model.
2. **Attacker Objective:** Deploy an unapproved model into production.
3. **Attacker Capabilities:** Local filesystem write access.
4. **Attack Prerequisites:** Unprotected model deployment directory.
5. **Attack Scenario:** Attacker replaces `model_final.pt` with a surrogate model.
6. **Expected Impact:** Production inference executes unauthorized weights.
7. **Detection Opportunities:**
   - *SHA-256 Digest Verification:* Immediate cryptographic mismatch against approved model manifest.
8. **Limitations of Detection:** If the attacker has access to rewrite approved manifest digests, external key custody is required.
9. **Required Evidence:** Candidate SHA-256 digest, approved SHA-256 digest, registration timestamp.
10. **Relevant Evaluation Metrics:** Deterministic match/mismatch correctness ($100\%$).

---

### ATK-MOD-02: Targeted Weight Tensor Modification (Bit-Flip / Trojan Insertion)
1. **Threat Description:** An adversary alters a subset of weight tensors within an approved PyTorch checkpoint (e.g., editing final classification head weights).
2. **Attacker Objective:** Stealthily alter behavior on specific inputs without retraining.
3. **Attacker Capabilities:** Byte-level modification access to `.pt` files.
4. **Attack Prerequisites:** Knowledge of PyTorch state-dictionary structure.
5. **Attack Scenario:** Attacker modifies 12 float values in the linear head of a ResNet-18 model.
6. **Expected Impact:** Targeted misclassification of chosen classes.
7. **Detection Opportunities:**
   - *SHA-256 Checkpoint Digest:* Byte-level mismatch.
   - *Behavioral Fingerprint Suite:* Running a fixed suite of 100 test inputs reveals logit divergence ($\Delta_{\max} > 10^{-4}$).
8. **Limitations of Detection:** Benign re-saving across different PyTorch versions alters byte hashes even when weights are semantically identical (fingerprint clarifies semantic equivalence).
9. **Required Evidence:** State-dict key differences, $L_2$ norm weight difference, fingerprint logit divergence.
10. **Relevant Evaluation Metrics:** Fingerprint sensitivity, False Alarm Rate under benign device float32 variations.

---

### ATK-MOD-03: Malicious Checkpoint Deserialization (Pickle RCE & DoS Exploit)
1. **Threat Description:** An adversary supplies a crafted PyTorch checkpoint containing malicious Python bytecode (`__reduce__` exploit) or astronomical tensor dimension headers.
2. **Attacker Objective:** Execute arbitrary operating system commands, steal private keys, or crash the host daemon via memory exhaustion (Denial of Service).
3. **Attacker Capabilities:** Ability to upload or submit a candidate model file.
4. **Attack Prerequisites:** System uses naive `torch.load()` without sandboxing or memory quotas.
5. **Attack Scenario:**
   - *RCE Payload:* Attacker embeds an OS subprocess payload (`os.system("nc ...")`) inside the pickle stream.
   - *DoS Payload:* Attacker constructs a valid tensor header with shape $[10^6, 10^6, 10^6]$, causing immediate allocation of hundreds of gigabytes.
6. **Expected Impact:** Total host compromise, shell execution, or host freeze/crash.
7. **Detection & Defense Opportunities:**
   - *Strict Prohibition:* `torch.load(..., weights_only=False)` is prohibited.
   - *Subprocess Sandboxing:* Loading runs inside an isolated ephemeral worker subprocess.
   - *OS Resource Limits:* Memory quota capped at 2 GB max (Windows Job Objects / Linux cgroups), with a 30-second hard execution timeout.
   - *Post-Load Verification:* Traverses keys, verifies strictly `torch.Tensor` instances, rejects excessive tensor shapes.
8. **Limitations of Detection:** Legacy checkpoints requiring custom class definitions cannot load under `weights_only=True`.
9. **Required Evidence:** Subprocess exit code, memory usage logs, extracted tensor metadata.
10. **Relevant Evaluation Metrics:** Exploit prevention rate ($100\%$), DoS containment rate ($100\%$).

---

## 4. Module 3: Inference Integrity Attack Scenarios

### ATK-INF-01: Audit Trail Record Modification
1. **Threat Description:** An operator or attacker modifies logged prediction outputs or scores in the database after an incident.
2. **Attacker Objective:** Conceal a safety-critical model misprediction.
3. **Attacker Capabilities:** Direct read/write SQL access to SQLite database.
4. **Attack Prerequisites:** Access to the database file `visionguard.db`.
5. **Attack Scenario:** Attacker alters `predicted_class` from "Pedestrian" to "Background" in record #4021.
6. **Expected Impact:** Corrupted audit trail; accountability destroyed.
7. **Detection Opportunities:**
   - *Ed25519 Signature Verification:* Re-computing canonical serialization (RFC 8785) and verifying against the stored signature fails immediately.
8. **Limitations of Detection:** Detects tampering, but cannot restore original values without external backups.
9. **Required Evidence:** Record ID, canonical byte payload, public key, signature validation error.
10. **Relevant Evaluation Metrics:** Tamper Detection Rate ($100\%$).

---

### ATK-INF-02: Record Deletion & Sequence Reordering
1. **Threat Description:** An attacker deletes an incriminating inference record or swaps the chronological execution order of two inference events.
2. **Attacker Objective:** Hide an event or create a false sequence of events.
3. **Attacker Capabilities:** Direct database modification.
4. **Attack Prerequisites:** Access to database rows.
5. **Attack Scenario:** Attacker deletes record #500 and updates record #501's sequence ID to #500.
6. **Expected Impact:** Gaps in audit trail.
7. **Detection Opportunities:**
   - *Hash-Chained Previous Hash Link:* Record #501 stores `previous_hash = SHA256(Record #500)`. When record #500 is deleted, the hash chain breaks at record #501.
8. **Limitations of Detection:** If the attacker has access to all chain records and no independent checkpoint has been exported, they can rewrite the entire chain from the deletion point onward (mitigated by periodic checkpoint export).
9. **Required Evidence:** Broken link sequence index, expected previous hash vs actual computed previous hash.
10. **Relevant Evaluation Metrics:** Chain Integrity Verification Precision ($100\%$).

---

### ATK-INF-03: Inference Event Replay Attack
1. **Threat Description:** An attacker captures a valid, signed inference receipt from an earlier event and re-submits it as proof of a new event.
2. **Attacker Objective:** Spoof an inference result without executing the model.
3. **Attacker Capabilities:** Network interception or read access to historical receipts.
4. **Attack Prerequisites:** Absence of monotonic sequence validation or nonce tracking.
5. **Attack Scenario:** Attacker submits receipt from 10:00 AM at 11:30 AM to validate a new inspection batch.
6. **Expected Impact:** Unauthorized action approved based on stale evidence.
7. **Detection Opportunities:**
   - *Monotonic Sequence Enforcer:* System verifies that `sequence_id = last_sequence + 1`.
   - *Nonce / Event ID Deduplication:* Re-used event UUIDs or nonces are rejected as replayed.
8. **Limitations of Detection:** In distributed inference nodes without centralized clock or consensus, sequence ordering requires timestamp windows.
9. **Required Evidence:** Duplicate event ID, timestamp discrepancy, out-of-order sequence index.
10. **Relevant Evaluation Metrics:** Replay Rejection Rate ($100\%$).

---

## 5. Module 4: Distribution Shift Scenarios (Environmental & Sensor)

### ATK-SHIFT-01: Severe Lighting & Illumination Drift
1. **Threat Description:** Ambient illumination changes drastically (daylight to dusk/fog) compared to daytime training data.
2. **Attacker Objective:** Natural environmental shift causing unflagged model degradation.
3. **Capabilities / Source:** Physical environment.
4. **Prerequisites:** Time-of-day progression or weather change.
5. **Scenario:** Camera stream shifts from bright daylight to dusk.
6. **Expected Impact:** Contrast loss, sensor grain noise; model accuracy drops.
7. **Detection Opportunities:**
   - *Multivariate MMD on Pretrained ResNet Features:* Detects distributional divergence between reference batch and incoming window ($p < 0.01$).
8. **Limitations of Detection:** Shift alert fires, but cannot determine whether accuracy is degraded without ground truth labels.
9. **Required Evidence:** Window size, reference dataset digest, MMD statistic, $p$-value.
10. **Relevant Evaluation Metrics:** Detection Latency (number of windows to flag), False Alarm Rate on clean reference holdouts.

---

### ATK-SHIFT-02: Sensor Degradation (Defocus Blur & Lens Dirt)
1. **Threat Description:** Camera lens acquires physical dirt, smudges, condensation, or loses focus over months of deployment.
2. **Attacker Objective:** Natural hardware degradation.
3. **Capabilities / Source:** Physical hardware wear.
4. **Prerequisites:** Unmaintained field installation.
5. **Scenario:** Progressive Gaussian blur ($\sigma$ increases from 0 to 4.0).
6. **Expected Impact:** Loss of high-frequency edges; small object detection misses.
7. **Detection Opportunities:**
   - *Laplacian Variance:* Variance of the Laplacian operator $\text{Var}(\nabla^2 I)$ drops below calibrated sharpness threshold $\tau_{\text{blur}}$.
8. **Limitations of Detection:** Naturally smooth scenes (e.g., sky) have low Laplacian variance without being blurry.
9. **Required Evidence:** Average Laplacian variance across window, high-frequency energy ratio.
10. **Relevant Evaluation Metrics:** Blur Detection AUROC, Correlation between sharpness metric and degradation severity.

---

## 6. Summary of Experimental Threat Simulation Suite

| Attack ID | Module | Attack Name | Controlled Attack Procedure | Target Dataset | Ground Truth | Expected Detection Tool |
|---|---|---|---|---|---|---|
| **ATK-DATA-01** | Data | BadNets Trigger Poisoning | Inject $3\times3$ grid, flip label | CIFAR-10 / GTSRB | Binary flag per image | Spectral Signatures / Activation Clustering |
| **ATK-DATA-02** | Data | Clean-Label Feature Collision | Perturb target image, keep label | CIFAR-10 / GTSRB | Binary flag per image | Feature k-NN Outlier Distance |
| **ATK-DATA-03** | Data | Systematic Label Flipping | Invert labels for specific contributor | CIFAR-10 / VOC | Flipped label ground truth | LOCO Contributor Risk Aggregator + k-NN Concordance |
| **ATK-DATA-04** | Data | Train/Val Split Leakage | Duplicate validation images into train | CIFAR-10 / COCO | Known duplicate image IDs | SHA-256 + Perceptual Hash (pHash) |
| **ATK-DATA-05** | Data | Malformed Bounding Boxes | Absolute overflow (COCO), normalized overflow (YOLO) | Pascal VOC / COCO | Synthesized error types | Format-Specific Geometry Validator |
| **ATK-MOD-01** | Model | Model Substitution | Swap `.pt` file with surrogate | PyTorch ResNet-18 | Known reference digest | SHA-256 Digest Verification |
| **ATK-MOD-02** | Model | Weight Tensor Modification | Alter linear layer weights | PyTorch ResNet-18 | Ground truth weight diff | Behavioral Fingerprint Suite |
| **ATK-MOD-03** | Model | Pickle RCE & DoS Exploit | Embed malicious bytecode or giant tensor header | PyTorch `.pt` file | Malicious bytecode flag | Subprocess Sandboxing & OS Resource Limits |
| **ATK-INF-01** | Inference | Record Field Modification | Alter database record payload | SQLite audit log | Known modified row ID | Ed25519 Signature Verification (RFC 8785) |
| **ATK-INF-02** | Inference | Record Deletion / Reorder | Delete record, swap rows | SQLite audit log | Target deletion index | SHA-256 Hash Chaining / Checkpoints |
| **ATK-INF-03** | Inference | Event Replay Attack | Resubmit valid signed receipt | REST API stream | Replayed event ID | Monotonic Sequence / Nonce Enforcer |
| **ATK-SHIFT-01** | Shift | Illumination Shift | Adjust brightness/contrast | GTSRB / CIFAR-10 | Shift window index | MMD / Kolmogorov-Smirnov Test |
| **ATK-SHIFT-02** | Shift | Defocus Blur Shift | Apply Gaussian blur kernels | GTSRB / COCO | Blur level $\sigma$ | Laplacian Variance & Power Spectrum |
