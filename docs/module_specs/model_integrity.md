# Module Specification: Model Integrity Engine

**Document Version:** 1.2.0 (Audited Patch)  
**Status:** Architectural Specification  
**Date:** November 2026  
**Scope:** Specification for PyTorch model artifact verification, sandboxed checkpoint loading, state-dictionary validation, behavioral fingerprinting, and gated backdoor analysis.

---

## 1. Goal & Mission

The Model Integrity Engine assesses whether a candidate PyTorch model artifact (`.pt` / `.pth`) matches an approved reference artifact, verifies structural state-dictionary integrity within a hardened OS sandbox, detects behavioral shifts across deterministic test probes, and runs gated backdoor checks when prerequisites are met.

---

## 2. Ingestion Contracts & Prerequisite Tiers

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Artifact & Checkpoint Verification (Static Analysis)           │
│   Inputs: Candidate `.pt`/`.pth` file, approved SHA-256 digest manifest│
│   Available: File hash matching, state-dict keys/shapes inspection,     │
│              sandboxed loading with OS resource quotas & isolation     │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Behavioral Fingerprint Suite (Black-Box Execution)             │
│   Inputs: Tier 1 + Executable PyTorch model, deterministic probe inputs│
│   Available: Logit divergence, top-1 agreement, numerical tolerances   │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Gated Backdoor Analysis (White/Black-Box Verification)         │
│   Inputs: Tier 2 + Clean reference inputs + Task classes               │
│   Available: STRIP entropy perturbation (black-box), Neural Cleanse    │
│              trigger optimization (white-box, small label spaces)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sandboxed Model Ingestion & Defense-in-Depth

### 3.1 Strict Prohibition of Naive Unpickling
> **MANDATORY SECURITY DIRECTIVE:**  
> Calling `torch.load(..., weights_only=False)` on untrusted files is **STRICTLY PROHIBITED**. Standard PyTorch loading invokes Python's `pickle` engine, allowing arbitrary command execution via `__reduce__` exploit payloads.

### 3.2 PyTorch Security Advisories & Version Enforcement
Security checks verify that the installed PyTorch runtime is patched against known deserialization vulnerabilities:
- **CVE-2025-32434:** PyTorch deserialization bypass in `torch.load` with `weights_only=True` allowing arbitrary object instantiation in PyTorch $< 2.5.1/2.6.0$. Runtime must be $\ge 2.6.0$ or patched baseline.
- **CVE-2026-24747 (GHSA-63cw-57p8-fm3p):** PyTorch C++ unpickler integer overflow during tensor buffer calculation in PyTorch $\le 2.9.1$. Patched in $\ge 2.10.0$. Causes `weights_only` unpickler memory corruption and potential arbitrary code execution.
- **GHSA-v32g-mr63-3j83:** PyTorch advisory published November 10, 2026. Reported affected versions $\ge 2.14.0$, reported patched in $2.15.0$. Identifies potential arbitrary file read during `torch.load(weights_only=True)`. No CVE currently assigned; noted as a newly published AI-assisted advisory with no package listed, tracked for upstream confirmation. Current project environment uses `torch==2.15.0+cpu` (meeting the 2.15.0 patch baseline and exceeding 2.10.0), which must not be downgraded.

### 3.3 Defense-in-Depth Checkpoint Sandbox Architecture & Capability Gating
> **MANDATORY SECURITY DIRECTIVE & CAPABILITY GATING:**  
> The OS isolation mechanisms specified below represent **enforced capability gates**.
> If any minimum isolation condition (network isolation, restricted filesystem access, or kernel resource quotas) cannot be proven on the host platform, VisionGuard **strictly refuses to deserialize untrusted checkpoints** and reports:
> - `status: UNAVAILABLE`
> - `reason: "Host platform lacks proven process sandboxing, network severance, or filesystem isolation controls required for untrusted checkpoint deserialization"`

Recognizing that `weights_only=True` is not a complete security sandbox and that static AST/opcode parsing can be bypassed by Turing-complete pickle bytecodes, VisionGuard enforces the following **fail-closed capability gating rules**:
1. **Isolation Assessment:** Prior to executing inspection, [`HostIsolationAssessor`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/inspection/worker.py) verifies the host OS security controls:
   - **Windows:** Evaluates AppContainer / Low Integrity tokens, network severance, and Job Object quotas. Standard desktop/server Windows processes report `UNHARDENED_HOST_SUBPROCESS` and fail closed with `CapabilityStatus.UNAVAILABLE`.
   - **Linux:** Evaluates unshared network namespaces (no default routing), container mounts, and cgroups v2 memory limits (`/sys/fs/cgroup/memory.max`). Unhardened Linux hosts report `UNHARDENED_HOST_SUBPROCESS` and fail closed with `CapabilityStatus.UNAVAILABLE`.
   - **Hardened Linux Containers:** Report `HARDENED_ISOLATION_VERIFIED` when all three OS controls are positively proven.
2. **Strict Deserialization Guard:** Under no circumstances is `torch.load` invoked on an untrusted checkpoint if isolation assessment is not `HARDENED_ISOLATION_VERIFIED`. No configuration bypass, environment variable override, or debug flag is permitted.
3. **Execution Environment:** When verified, the worker executes inside a disposable subprocess with sanitized minimal environment (`clean_env`), CPU-only device allocation (`map_location="cpu"`), and hard 30-second kill timeout.
4. **Post-Load State-Dict Verification & Canonical Fingerprinting:**
   - Verifies root structure is strictly a `dict`.
   - Validates that values are `torch.Tensor` or `nn.Parameter`.
   - Computes canonical, deterministic digests across all supported tensor types:
     - **Dense:** `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`, and zero-sized tensors.
     - **Quantized:** `qint8`, `quint8`, `qint32` (hashes quantization scheme, scale, zero_point, and integer representation).
     - **Sparse:** `sparse_coo` (coalesced, hashes indices and values).
     - **Complex:** `complex64`, `complex128`.
   - Evaluates `NaN` and `Inf` counts across all parameters.
   - Evaluates `state_dict_fingerprint` against approved reference digests when configured (fails with `CRITICAL` finding on mismatch; reports `SKIPPED_NO_REFERENCE` when omitted without claiming verified authenticity).

### 3.4 Safer Tensor-Only Storage (`safetensors`)
VisionGuard will evaluate supporting `safetensors` (pure JSON header + flat binary buffers, zero unpickling, zero executable bytecode) as an alternative native model storage format using `safetensors.torch.load_file()`.

---

## 4. Behavioral Fingerprinting & Backdoor Indicator Analysis (Tier 2 & 3)

### 4.1 Architecture Registry & Safe Model Instantiation
To prevent arbitrary code execution during behavioral inference, model weights are loaded strictly with `weights_only=True` and instantiated into predefined, statically verified architectures from [`TrustedArchitectureRegistry`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/architecture/registry.py):
- `simple_cnn`: Native 4-layer CNN for fast edge evaluation ($3 \times 32 \times 32$).
- `resnet18`: TorchVision ResNet-18 residual classifier ($3 \times 224 \times 224$).
- `mobilenet_v2`: TorchVision MobileNetV2 inverted residual edge model ($3 \times 224 \times 224$).

VisionGuard **never executes arbitrary uploaded Python classes or module scripts**.

### 4.2 Isolated Subprocess Worker & Fail-Closed Capability Gating
All behavioral probing and inference forward passes execute within a dedicated disposable child subprocess ([`probe_cli.py`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/behavioral/probe_cli.py)) managed by [`SubprocessBehavioralWorker`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/behavioral/worker.py):
1. **Granular OS Isolation Assessment:** Prior to spawning the worker, [`HostIsolationAssessor`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/inspection/worker.py) positively probes enforceable host controls:
   - **Enforceable Controls Checked:** Container boundary markers (`/.dockerenv`), cgroups v2 memory limits (`memory.max`), cgroups v2 PID limits (`pids.max` for fork-bomb prevention), severed routing tables (`/proc/net/route` empty or loopback-only), `NoNewPrivileges` enforcement (`/proc/self/status`), unprivileged non-root user execution (`uid != 0`), and read-only root mounts (`/proc/mounts`).
   - **Fail-Closed Guarantee:** On unhardened environments (standard Windows or unconfined Linux missing any required control), all 5 behavioral detectors immediately return `CapabilityStatus.UNAVAILABLE` with a structured list of missing controls.
   - **Epistemic Honesty Regarding Sandboxing:** A basic subprocess with stripped environment (`clean_env`) does not constitute kernel namespace or filesystem isolation. True containment requires external hardened container deployment. Without positive verification of all OS controls, capability status is strictly `UNAVAILABLE`.
2. **PRNG Isolation:** Probe generation uses an explicit, isolated `torch.Generator().manual_seed(seed)` to guarantee zero mutation of global Python or PyTorch random state.
3. **Canonical Compound Cache Isolation:** In-memory assessment caching ([`common.py`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/backend/app/modules/model_integrity/behavioral/common.py)) utilizes a 64-character SHA-256 compound key bound to candidate artifact SHA-256, reference SHA-256, architecture ID, assessment ID, lease generation, probe parameters, and detection thresholds. Metadata is verified on retrieval, preventing stale report reuse across mutated configurations or retried job attempts.

### 4.3 Behavioral Detection Suite (5 Registered Detectors)

1. **Behavioral Fingerprint Detector (`model_integrity.behavioral_fingerprint`):**
   - Evaluates $N=30$ deterministic synthetic reference probes in $[0.0, 1.0]$.
   - Computes canonical SHA-256 fingerprint over sorted, canonicalized per-probe records:
     $$\text{record}_i = \{\text{index}, \text{class}, \text{confidence}, \text{logit\_digest}\}$$
   - When configured with `expected_behavioral_fingerprint`, verifies exact match. Emits `UNAUTHORIZED_MODEL_CHANGE` finding (Severity: `HIGH`, Disposition: `QUARANTINE`) on mismatch.

2. **Prediction Consistency Detector (`model_integrity.prediction_consistency`):**
   - Evaluates $R=3$ repeated inference evaluations under fixed `model.eval()` and `torch.no_grad()`.
   - Computes prediction class flips and maximum logit drift against numerical tolerance ($10^{-4}$).
   - Detects stochastic dropout layers left unpinned, stateful buffers, or non-deterministic kernels.
   - Emits `STRUCTURAL_DEFECT` finding (Severity: `HIGH`, Disposition: `REVIEW`) if flips $> 0$.

3. **Input Sensitivity Detector (`model_integrity.input_sensitivity`):**
   - Evaluates model stability under bounded uniform input perturbation in $[-\epsilon, \epsilon]$ ($\epsilon=0.02$).
   - Measures prediction flip rate, mean confidence shift, and logit amplification:
     $$\text{amplification} = \frac{\|\mathbf{z}_{\text{pert}} - \mathbf{z}_{\text{clean}}\|_2}{\|\mathbf{\delta}\|_2}$$
   - Emits `STRUCTURAL_DEFECT` finding (Severity: `MEDIUM`, Disposition: `REVIEW`) if flip rate $> 40\%$.

4. **Backdoor Indicator Detector (`model_integrity.backdoor_indicator`):**
   - Evaluates 4 candidate trigger patterns:
     - `corner_checkerboard`: High-contrast $4 \times 4$ checkerboard in bottom-right corner.
     - `corner_square`: Solid white $4 \times 4$ patch in top-left corner.
     - `center_cross`: Centered crosshair pattern.
     - `blended_watermark`: Low-magnitude ($\alpha=0.15$) full-frame static grid.
   - Evaluates target-class concentration ratio $C_{\text{trigger}}$, concentration delta $\Delta C$ over clean baseline, and average target confidence $\bar{p}$.
   - Flags `SUSPECTED_POISON` finding (Severity: `CRITICAL` if $C \ge 0.95$, `HIGH` if $C \ge 0.85$; Disposition: `QUARANTINE`) when:
     $$C_{\text{trigger}} \ge 0.85 \quad \land \quad \Delta C \ge 0.50 \quad \land \quad \bar{p} \ge 0.80$$

5. **Reference Comparison Detector (`model_integrity.reference_comparison`):**
   - Compares candidate model outputs against an authoritative trusted reference model.
   - **Self-Reference Protection:** Candidate checkpoints can never serve as their own reference; rejected via canonical path and content SHA-256 digest comparison.
   - **Authoritative Provenance Verification:** Reference checkpoints must possess an authoritative persisted `ModelRepository` record in `ACCEPTED` status. On-disk file SHA-256 is verified against the authoritative database digest before execution.
   - Measures top-1 prediction agreement rate and logit divergence:
     $$\text{agreement} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i^{(\text{cand})} = \hat{y}_i^{(\text{ref})}), \quad \Delta_{\max} = \max_{i, c} |z_{\text{cand}, i, c} - z_{\text{ref}, i, c}|$$
   - Reports `SKIPPED` if no reference model is configured.
   - Emits `UNAUTHORIZED_MODEL_CHANGE` finding (Severity: `HIGH` if agreement $< 90\%$, `MEDIUM` if $< 100\%$; Disposition: `QUARANTINE`) on behavioral deviation.

### 4.4 Crucial Epistemic Honesty Rules & Known Limitations
- **Candidate Triggers Are Non-Exhaustive:** Testing candidate triggers detects standard heuristic patch/blend patterns but **does NOT prove the model is free of backdoors**. Backdoors triggered by custom semantic features, imperceptible universal perturbations, or complex physical triggers cannot be ruled out by heuristic probing.
- **Epistemic Labeling:** The absence of detected backdoor indicators is recorded as "clean under candidate suite" with an explicit caveat: `"Absence of detected backdoor indicators does not prove absence of backdoors."` It is **never reported as a mathematical guarantee of backdoor absence**.
- **No False Attribution:** An anomalous sensitivity score or high concentration is flagged as `SUSPECTED_POISON`, not an absolute confirmed attack, unless explicit ground truth is available.
- **Trusted Reference Constraints:** Reference models cannot be supplied ad-hoc without formal ingestion and approval in `ModelRepository`. Unverified or caller-asserted reference files fail closed.

---

## 5. Evaluation Strategy & Metrics

- **Digest Verification:** Accuracy ($100\%$), False Reject Rate on identical copies ($0\%$).
- **Behavioral Fingerprint:** Deterministic canonical SHA-256 reproducibility ($100\%$), sensitivity to single-layer weight modification.
- **Consistency Verification:** Zero tolerance for stochastic output under evaluation mode.
- **Backdoor Detection:** Attack Success Rate (ASR) reduction, class concentration detection on heuristic trigger patterns, epistemic honesty audit.
- **Fail-Closed Boundary:** Positive verification of `CapabilityStatus.UNAVAILABLE` on unhardened platforms before deserialization or execution.
