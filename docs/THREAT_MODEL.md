# VisionGuard Unified Threat Model

**Document Version:** 1.2.0 (Audited Patch)  
**Status:** Unified Threat Model Baseline  
**Date:** November 2026  
**Scope:** Comprehensive threat modeling covering Data Integrity, Model Integrity, Inference Integrity, Distribution Shift, PyTorch Advisories (CVE-2025-32434, CVE-2026-24747), and Multi-Layer Process Isolation.

---

## 1. Security Objectives & Core Principles

VisionGuard is an offline-first AI lifecycle integrity and security assurance platform for computer vision systems. The system's primary security objectives are:
1. **Preserve Integrity & Provenance:** Ensure that datasets, model artifacts, inference records, and shift references are cryptographically verified and bound to traceable origins.
2. **Surface Evidence-Backed Anomaly Findings:** Detect data corruption, poisoning triggers, model tampering, and inference manipulation using statistically grounded signals.
3. **Prevent Unsafe Processing of Untrusted Inputs:** Defend the host platform from malicious archives, path traversal exploits, decompression bombs, dataset unpickling, and checkpoint-based code execution.
4. **Enforce Capability-Aware Reporting:** Make detector prerequisites, execution status (`available`, `unavailable`, `failed`, `not_run`), and limitations explicit to prevent false "all clear" assumptions.
5. **Human-in-the-Loop Governance:** Restrict automated actions to review and quarantine suggestions (`accept`, `review`, `quarantine`); avoid autonomous data destruction.

---

## 2. In-Depth PyTorch Checkpoint Ingestion Threat Model

PyTorch checkpoint loading is one of the highest-risk attack surfaces in modern machine learning systems.

```text
Untrusted Model File (.pt / .pth)
               │
               ▼
   [Layer 1: Static Pre-Check & Version Verification]
   ├── Max archive size check (< 2 GB)
   ├── Magic number signature inspection
   ├── PyTorch version-specific vulnerability check (CVE-2025-32434, CVE-2026-24747, GHSA-v32g-mr63-3j83)
   └── Strictly PROHIBIT torch.load(..., weights_only=False)
               │
               ▼
   [Layer 2: Sandboxed Process Isolation (OS-Enforced)]
   ├── Host Isolation Check: If adequate OS isolation absent ──▶ Status: UNAVAILABLE
   ├── Windows Sandbox: Low-integrity token / AppContainer + Job Object + Network disabled
   ├── Linux Sandbox: cgroups v2 + unshare (-m, -n, -p) + seccomp-bpf + tmpfs scratch
   ├── OS-level memory quota (2 GB hard cap) & execution timeout (30 seconds)
   └── torch.load(path, map_location="cpu", weights_only=True)
               │
               ▼
   [Layer 3: Post-Load State-Dict Verification]
   ├── Verify object is strictly dict[str, torch.Tensor]
   ├── Validate tensor dimensions (reject astronomical shapes: DoS defense)
   ├── Enforce expected float32 / float16 dtypes
   └── Verify state-dict keys against approved architecture contract
               │
               ▼
   Candidate Model Safe for Memory Loading / Behavioral Inspection
```

### 2.1 Explicit Prohibition of `weights_only=False`
> **MANDATORY SECURITY DIRECTIVE:**  
> `torch.load(..., weights_only=False)` is **STRICTLY PROHIBITED** on all untrusted model files. Standard PyTorch loading invokes Python's `pickle` engine, allowing arbitrary command execution via `__reduce__` exploit payloads.

### 2.2 Critical Limitations of `weights_only=True` & Relevant PyTorch Advisories
- `weights_only=True` (introduced in PyTorch 1.13 and made default in 2.6) restricts unpickling to a whitelist of primitive types and tensors.
- **`weights_only=True` IS NOT A COMPLETE SECURITY SANDBOX:**
  - Upstream vulnerabilities in PyTorch unpickling and C++ archive parsing have demonstrated that whitelist bypasses and memory corruptions occur.
  - **CVE-2025-32434 (PyTorch Deserialization Security Advisory):**
    - *Affected Versions:* PyTorch $< 2.5.1 / 2.6.0$.
    - *Vulnerability:* Improper restriction in `torch.load` with `weights_only=True` where crafted nested object graphs bypassed type whitelisting, enabling arbitrary object instantiation.
    - *Official Reference:* PyTorch Security Advisory / GHSA-2025-32434.
    - *Mitigation:* Ensure PyTorch runtime $\ge 2.6.0$ or patched 2.5.1+ baseline; enforce OS-level sandboxing.
  - **CVE-2026-24747 (PyTorch C++ Checkpoint Parser Integer Overflow / Type Confusion):**
    - *Affected Versions:* PyTorch $\le 2.9.1$.
    - *Patched Versions:* PyTorch $\ge 2.10.0$.
    - *Impact:* `weights_only` unpickler memory corruption and potential arbitrary code execution via integer overflow during tensor buffer size calculation.
    - *Official Reference:* PyTorch Security Advisory [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
    - *Mitigation:* Enforce runtime $\ge 2.10.0$, validate archive byte limits prior to parsing, and isolate deserialization in an unprivileged, sandboxed subprocess.
  - **GHSA-v32g-mr63-3j83 (PyTorch Arbitrary File Read Advisory):**
    - *Advisory ID:* GHSA-v32g-mr63-3j83.
    - *Published Date:* November 10, 2026.
    - *Reported Affected Versions:* $\ge 2.14.0$.
    - *Reported Patched Version:* $2.15.0$.
    - *Impact:* Potential arbitrary file read during `torch.load(weights_only=True)`.
    - *CVE Status:* No CVE assigned in current advisory.
    - *Advisory Nature & Tracking:* Explicitly noted as a newly published AI-assisted advisory with no package listed; VisionGuard tracks its status and will verify applicability with future official upstream updates.
    - *Runtime Verification Status:* Current development environment uses `torch==2.15.0+cpu`, satisfying the reported patched version ($2.15.0$) and exceeding the $2.10.0$ patch threshold for CVE-2026-24747. The installed package must not be downgraded based on affected-version assumptions.
  - *(Correction Note: CVE-2024-27318 concerns an ONNX directory traversal vulnerability in `onnx.load_model` and is unrelated to PyTorch; it was removed from PyTorch threat modeling).*

### 2.3 Denial of Service (DoS) & Memory Exhaustion Threats
- An adversary can construct a syntactically valid tensor checkpoint with astronomical dimension headers (e.g., shape $[10^6, 10^6, 10^6]$).
- Attempting to allocate this tensor causes immediate host memory exhaustion (OOM), crashing the VisionGuard daemon.
- Nested collection bombs can cause infinite CPU loops during dictionary traversal.
- Static Python AST inspection or naive pickle opcode filtering is insufficient: pickle bytecode is a Turing-complete stack machine, and obfuscated opcode sequences cannot be statically proven benign.

### 2.4 Mandatory Defense-in-Depth Sandbox Specifications & Limitations
> **IMPORTANT SECURITY STATUS:**  
> The OS isolation mechanisms specified below are **architectural specifications and security targets**, NOT currently implemented security guarantees. Until a fully validated isolation runtime is verified on the target deployment machine, untrusted PyTorch checkpoints cannot be presumed safe.

VisionGuard defines the following **minimum conditions for permitting untrusted checkpoint deserialization**:
1. **Strict Process Isolation:** Non-elevated child process running under an unprivileged, low-integrity security token (Windows `Low Integrity Level` or `AppContainer`; Linux unshared user namespace `unshare -U`), completely isolated from the supervisor daemon.
2. **Complete Network Isolation:** Network sockets disabled (loopback disabled or network namespace `unshare -n` with zero routes).
3. **Restricted Filesystem Access:** Read-only handle to the candidate checkpoint file; write access restricted strictly to an isolated, ephemeral scratch directory (`tmpfs`) that is purged immediately upon process exit. No access permitted to host configuration, credentials, source code, or application state.
4. **Strict Resource Quotas:** Hard memory limit (e.g., 2 GB max) and CPU cycle cap enforced via Windows Job Objects or Linux cgroups v2 (`memory.max`), backed by a 30-second hard execution timeout with immediate `SIGKILL` termination.

**Capability Gating on Sandbox Absence:**  
If any of these minimum isolation conditions (network isolation, restricted filesystem access, or process isolation) cannot be proven on the host platform, VisionGuard **strictly refuses to deserialize untrusted checkpoints** and reports:
- `status: UNAVAILABLE`
- `reason: "Host platform lacks proven process sandboxing, network severance, or filesystem isolation controls required for untrusted checkpoint deserialization"`

VisionGuard supports and recommends `safetensors` (pure JSON header + flat raw byte buffers, zero executable code, zero unpickling) as a preferred model storage format while preserving PyTorch compatibility via `safetensors.torch.load_file()`.

---

## 3. Dataset Ingestion Threat Model & CIFAR-10 Security

1. **Pickle-Based Dataset Ingestion Vulnerability:**
   - Standard Python distributions of certain benchmarks (notably `cifar-10-python.tar.gz`) store images in pickled dictionary batches.
   - Unpickling dataset files inside the application runtime creates identical RCE risks as unpickling untrusted model checkpoints.
2. **Mandatory Binary Ingestion Policy:**
   - VisionGuard **strictly mandates the official binary distribution** of CIFAR-10 (`cifar-10-binary.tar.gz`).
   - **Binary Record Layout Specification:**
     - Exactly 3,073 bytes per image record:
       - **Byte 0 (1 byte):** Unsigned 8-bit integer representing class label ($0 \dots 9$).
       - **Bytes 1–1024 (1024 bytes):** Red channel ($32 \times 32$ pixels, row-major order).
       - **Bytes 1025–2048 (1024 bytes):** Green channel ($32 \times 32$ pixels, row-major order).
       - **Bytes 2049–3072 (1024 bytes):** Blue channel ($32 \times 32$ pixels, row-major order).
     - **Tensor Layout & Reshaping:** Raw byte ingestion produces shape `(3, 32, 32)` in channel-first (CHW) order. An explicit conversion / transpose to `(32, 32, 3)` (HWC) is executed when interfacing with standard image processing libraries (Pillow, OpenCV) or visualization tools.
   - Ingestion executes via `numpy.frombuffer()` or direct binary file reads with **zero unpickling**.
   - If an untrusted dataset in Python pickle format is provided by an operator, it must **never** be loaded in the main process; it must be rejected or converted offline inside a hardened, isolated sandbox worker.
3. **Identity vs. Authenticity of Checksums:**
   - A locally calculated SHA-256 digest of a dataset archive or image file establishes **artifact identity** (confirming exact byte matches across local operations or pipeline stages), **NOT authenticity**.
   - Authenticity is only established when the computed digest is verified against a **trusted, independent external source** (e.g., authoritative upstream checksums published on an authentic academic domain or an Ed25519-signed release manifest from a verified authority).

---

## 4. Assets to Protect & Trust Boundaries

| Asset Category | Specific Asset Item | Sensitivity / Integrity Requirement |
|---|---|---|
| **Data Assets** | Raw images (`.jpg`, `.png`), binary dataset records, annotations (COCO JSON, YOLO txt), split partitions, provenance manifests. | High integrity. Tampering leads to poisoned or degraded downstream models. |
| **Model Assets** | PyTorch model checkpoints (`.pt`, `.pth`), state dictionaries, architecture configs, approved SHA-256 digests. | Critical integrity. Replacement or weight alteration alters production behavior. |
| **Audit Assets** | Inference inputs, outputs, Ed25519 signed receipts, SHA-256 hash chains, checkpoint states. | Critical integrity and non-repudiation. Tampering invalidates safety/regulatory compliance. |
| **Cryptographic Assets** | Ed25519 private signing keys, public keys, key custody metadata. | Critical confidentiality and integrity. Key compromise allows forged inference receipts. |
| **Reference Assets** | Baseline reference datasets, extracted statistical feature distributions for drift detection. | High integrity. Contaminated reference data suppresses distribution shift alerts. |
| **System State Assets** | SQLite database (`visionguard.db`), assessment configs, detector versions, generated reports. | High integrity. Tampering hides findings and misleads operators. |

---

## 5. Threat Scenarios Across All Four Modules

### 5.1 Module 1: Data Integrity Threats

| Threat ID | Threat Name | Attacker Capability & Scenario | Impact | Primary Detection Opportunity |
|---|---|---|---|---|
| **T-DATA-01** | **Dirty-Label Backdoor Poisoning** | Attacker injects visual trigger patterns (e.g., BadNets) and changes labels to target class. | Model misclassifies trigger images; safety failure. | Spectral Signatures (covariance SVD) & Activation Clustering (penultimate PCA + k-means). |
| **T-DATA-02** | **Clean-Label Poisoning** | Attacker perturbs images within target class to collide with test sample feature representations. | Misclassification of targeted clean test sample; evades manual visual audit. | Feature-space k-NN distance and high-frequency residual filtering. |
| **T-DATA-03** | **Systematic Label Flipping** | Contributor systematically inverts labels between critical classes. | High confusion on specific classes; demographic/safety bias. | Confident Learning joint distribution ($\hat{Q}_{\tilde{y}, y^*}$) & neighbor concordance ($C_k$). |
| **T-DATA-04** | **Split Leakage via Duplicates** | Attacker copies validation/test images into training set with minor crops/recompression. | Artificially inflated validation scores; real-world generalization failure. | Exact SHA-256 byte hashing & Perceptual Hashing (pHash) across split sets. |
| **T-DATA-05** | **Malformed Bounding Boxes** | Attacker injects negative dimensions, out-of-bounds coordinates, or conflicting overlapping boxes. | Training divergence (NaN loss) or degraded object detection mAP. | Tier 1 Structural Geometry Validator (COCO absolute pixel bounds, YOLO normalized unit bounds). |
| **T-DATA-06** | **Out-of-Distribution Contamination** | Attacker or scraper injects semantically unrelated images into specialized dataset. | Unpredictable model behavior on anomalous inputs. | Penultimate layer k-NN Euclidean distance against approved reference set. |
| **T-DATA-07** | **Contributor Risk Masking & Sybils** | Attacker submits malicious samples across multiple fake contributor identities. | Evades single-contributor volume flags. | LOCO empirical Bayesian risk smoothing, posterior uncertainty intervals, and metadata cluster tracking. |

### 5.2 Module 2: Model Integrity Threats

| Threat ID | Threat Name | Attacker Capability & Scenario | Impact | Primary Detection Opportunity |
|---|---|---|---|---|
| **T-MOD-01** | **Model Substitution** | Attacker swaps production `.pt` artifact with unauthorized surrogate. | Execution of unapproved model. | SHA-256 digest comparison against approved model manifest. |
| **T-MOD-02** | **Weight Tensor Modification** | Attacker alters specific weight tensors in an approved model checkpoint. | Silent behavioral shift on specific classes. | Behavioral Fingerprint Suite (deterministic logit divergence on fixed probe inputs). |
| **T-MOD-03** | **Pickle RCE & DoS Exploit** | Attacker embeds arbitrary Python bytecode or oversized tensor headers in `.pt` file. | Arbitrary command execution or host memory exhaustion (DoS). | Sandboxed worker subprocess, OS memory caps (2GB), strict timeout (30s), post-load tensor validation. |
| **T-MOD-04** | **Trained Backdoor Insertion** | Attacker trains model with hidden backdoor trigger before exporting checkpoint. | Invariant trigger activation during inference. | Gated backdoor verification (STRIP entropy tests / Neural Cleanse). |

### 5.3 Module 3: Inference Integrity Threats

| Threat ID | Threat Name | Attacker Capability & Scenario | Impact | Primary Detection Opportunity |
|---|---|---|---|---|
| **T-INF-01** | **Record Modification** | Attacker edits logged prediction outputs or scores in the database. | False audit history; incident cover-up. | Ed25519 digital signature verification over canonical serialized receipt (RFC 8785). |
| **T-INF-02** | **Record Deletion & Insertion** | Attacker removes an incriminating record or inserts an artificial retroactive record. | Corrupted chronological sequence. | Append-only SHA-256 hash chaining (previous hash linking) verified against checkpoints. |
| **T-INF-03** | **Event Replay Attack** | Attacker resubmits a valid signed receipt from an earlier event to bypass inspection. | Stale result accepted as proof of current safety. | Monotonic sequence validation, unique event UUID deduplication, nonce policy. |
| **T-INF-04** | **Signing-Key Compromise** | Attacker steals local Ed25519 private key from disk. | Unrestricted receipt forgery. | Strict OS file ACLs (`chmod 600`), key rotation procedures, offline key custody documentation. |

### 5.4 Module 4: Distribution Shift Threats

| Threat ID | Threat Name | Attacker Capability & Scenario | Impact | Primary Detection Opportunity |
|---|---|---|---|---|
| **T-SHIFT-01** | **Illumination / Contrast Drift** | Natural ambient lighting changes drastically (daylight to dusk/fog). | Model accuracy drops due to covariate shift. | Multivariate Maximum Mean Discrepancy (MMD) on deep embeddings + luminance statistics. |
| **T-SHIFT-02** | **Sensor Blur & Lens Dirt** | Field camera lens defocuses or collects physical grime over time. | High-frequency edge loss; small object detection misses. | Laplacian variance sharpness metric $\text{Var}(\nabla^2 I)$ and frequency power spectrum slope. |
| **T-SHIFT-03** | **Class Proportion Shift** | Target environment exhibits severe change in real-world prior class frequencies. | Uncalibrated posterior probabilities; elevated false alarm rates. | Kolmogorov-Smirnov (KS) tests on predicted probability distributions across window. |

---

## 6. What VisionGuard Does NOT Prove (Non-Goals & Honest Limitations)

1. **An Anomaly Score Is Not Proof of an Attack:** An anomaly flag is a measurement of data divergence or physical defect; human review is mandatory.
2. **A Hash Mismatch Proves Byte Difference, Not Malicious Intent:** Re-exporting a PyTorch model in a newer PyTorch version alters byte hashes due to ZIP container metadata.
3. **A Valid Signature Does Not Prove Prediction Accuracy:** An Ed25519 signature confirms non-repudiation and receipt integrity; it does not guarantee neural network correctness.
4. **An Unanchored Hash Chain Can Be Completely Rewritten:** If an attacker has full database access and no external, trusted checkpoint was exported, they can re-compute the chain. Periodic external checkpoint export is an operational requirement.
5. **Input Distribution Shift Does Not Automatically Prove Accuracy Degradation:** Covariate shift can occur while accuracy remains stable.
6. **An Unavailable Method Is Not a Passing Clean Result:** Missing prerequisites or inadequate host sandboxes trigger `status: UNAVAILABLE` with explicit reasons; they never produce clean bills of health.
7. **Contributor Attribution Does Not Prove Contributor Malice:** High contributor risk indicates statistical defect concentration, not malicious intent.
