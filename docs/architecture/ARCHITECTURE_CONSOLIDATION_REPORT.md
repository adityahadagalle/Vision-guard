# Architecture Consolidation & Design Lock Report
**Document Version:** 1.0.0  
**Phase:** 1A (Architecture Consolidation & Design Lock)  
**Date:** 2026-11-12  
**System:** VisionGuard AI Lifecycle Integrity & Security Platform  
**Author:** Senior Software Architect & AI Security Engineer  

---

## 1. Executive Summary

This report documents the comprehensive audit, consolidation, and formal locking of the system architecture for **VisionGuard**. Prior to this phase, architectural guidance, threat models, and module specifications were distributed across foundational documents, research summaries, and audit patch notes. In some instances, conflicting assumptions emerged regarding ingestion workflows, feature extraction responsibilities, checkpoint deserialization security guarantees, and vulnerability tracking.

Through Phase 1A, all existing architectural artifacts have been audited, conflicting decisions resolved, and one unified, authoritative technical specification locked across:
1. `docs/architecture/SYSTEM_ARCHITECTURE.md`
2. `docs/architecture/SECURITY_ARCHITECTURE.md`
3. `docs/architecture/DATA_AND_FINDINGS_MODEL.md`
4. `docs/architecture/IMPLEMENTATION_ROADMAP.md`
5. `docs/architecture/ARCHITECTURE_DECISIONS.md`

All architectural decisions strictly preserve the project's core invariants: **PyTorch-only deep learning stack**, **offline-first air-gapped autonomy**, **modular monolith with ports and adapters**, **isolated worker execution**, and **content-addressed immutable provenance**.

---

## 2. Existing Documents Inspected

An exhaustive review of all codebase and documentation assets was conducted:

| Document Path | Document Purpose | Role in Phase 1A Review |
|---|---|---|
| `AGENTS.md` | Core agent rules and invariants | Defined immutable engineering, security, and technology constraints |
| `PROJECT_CONTEXT.md` | Project identity, problem statement, goals | Established operational scope (SIH 2026, air-gapped vision systems) |
| `REQUIREMENTS.md` | Functional & non-functional requirements | Verified module requirements against architectural capabilities |
| `TECH_STACK.md` | Approved technology dependencies | Confirmed allowed dependencies; enforced prohibition of TF/ONNX/Celery/Redis |
| `docs/ARCHITECTURE.md` | Initial high-level architecture draft | Evaluated for architectural patterns; superseded by `docs/architecture/` |
| `docs/THREAT_MODEL.md` | Threat taxonomy and attacker profiles | Integrated threat boundaries into `SECURITY_ARCHITECTURE.md` |
| `docs/IMPLEMENTATION_ROADMAP.md` | Initial implementation schedule | Restructured into strict vertical slices in `docs/architecture/IMPLEMENTATION_ROADMAP.md` |
| `docs/module_specs/01_data_integrity.md` | Data Integrity technical spec | Reconciled detector interfaces, tier models, and quarantine workflows |
| `docs/module_specs/02_model_integrity.md` | Model Integrity technical spec | Reconciled checkpoint inspection, Trojan scanning, and sandbox rules |
| `docs/module_specs/03_inference_integrity.md` | Inference Integrity technical spec | Reconciled receipt signing, hash chaining, and edge runtime architecture |
| `docs/module_specs/04_distribution_shift.md` | Distribution Shift technical spec | Reconciled drift metrics, baseline snapshots, and feature provider integration |
| `docs/research/dataset_selection.md` | Dataset licensing and benchmark audit | Audited licensing restrictions (COCO, CIFAR-10, GTSRB, VOC) and record layouts |
| `docs/research/cve_analysis.md` | Deep learning CVE vulnerability research | Audited PyTorch unpickler CVEs and verified version boundaries |
| `docs/research/detector_survey.md` | Detection algorithms and baselines | Formatted detector inputs, calibration procedures, and FPR targets |
| `docs/PHASE_1_AUDIT_REPORT.md` | Audit findings from Phase 1 | Tracked resolution of audit items |
| `docs/PHASE_1_SECURITY_CORRECTIONS.md` | Specific security corrections applied | Verified advisory IDs (GHSA-63cw-57p8-fm3p, GHSA-v32g-mr63-3j83) |
| `FILE_INVENTORY.txt` | Repository file listing | Maintained file traceability and structure |
| `backend/app/main.py` & tests | Working Phase 0 backend | Verified operational boundaries, endpoints, and health checks |

---

## 3. Unavailable Source Documents

- **Claude's Independent Architecture Review**:  
  The user prompt noted the possible existence of an independent architecture review conducted by Claude. A thorough filesystem search (`dir /s`, git history search) confirmed that **no standalone file named Claude's Architecture Review exists on disk**.
  - **Resolution**: The audit reports (`PHASE_1_AUDIT_REPORT.md`) and security correction notes were used as the primary review corpus. Where independent critique was implied regarding sandbox limitations, receipt provability claims, and Bayesian priors, these were addressed directly and resolved conservatively in the consolidated specifications.

---

## 4. Conflicting Decisions Identified & Resolved

Ten concrete architectural conflicts were identified during the audit and resolved:

### Conflict 1: Ingestion Order & Format Detection vs. Quarantine Isolation
- **Prior Conflict**: Early module drafts implied that incoming datasets would undergo format detection (detecting COCO vs. YOLO vs. ImageNet) *before* or *during* extraction to determine unpacking logic.
- **Security Flaw**: Inspecting unvalidated archive headers or parsing inner paths prior to quarantine creates severe zip-bomb and directory-traversal vulnerabilities.
- **Resolution**: The ingestion order is strictly locked in [SYSTEM_ARCHITECTURE.md](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/SYSTEM_ARCHITECTURE.md):
  1. Raw upload directly to `data/quarantine/`.
  2. Isolated worker performs bounded defensive inspection (zip-slip checks, ratio checks, device name sanitization).
  3. Safe extraction into an immutable snapshot directory with canonical `manifest.json`.
  4. Format detection occurs **only after** safe extraction, operating strictly on safe, local files.

### Conflict 2: Feature Extraction Duplication vs. Shared Provider
- **Prior Conflict**: Data Integrity (duplicate detection, OOD), Model Integrity (Trojan activation clustering), and Distribution Shift (Wasserstein/MMD drift) each described independent feature extraction passes.
- **Performance Flaw**: Multiple detectors running deep backbones independently would cause $4\times\text{--}6\times$ redundant inference on multi-gigabyte datasets, exhausting CPU/GPU resources.
- **Resolution**: Resolved in [ADR-008](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/ARCHITECTURE_DECISIONS.md#adr-008-shared-feature-provider-with-content-addressed-embedding-cache). Feature extraction is decoupled into a `SharedFeatureProvider` with a persistent content-addressed embedding cache keyed by `SHA256(sample_sha256 + backbone_name + weights_sha256 + layer)`. Detectors declare feature requirements; the orchestrator computes embeddings once per snapshot.

### Conflict 3: Checkpoint Deserialization Security Guarantees
- **Prior Conflict**: Early drafts suggested that using `torch.load(..., weights_only=True)` completely eliminates arbitrary code execution risks, allowing in-process model loading.
- **Security Flaw**: `weights_only=True` restricts unpickler globals, but C++ memory corruption vulnerabilities (e.g., CVE-2026-24747) can bypass Python unpicklers, causing process crash or shellcode execution.
- **Resolution**: Resolved in [SECURITY_ARCHITECTURE.md](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/SECURITY_ARCHITECTURE.md) and [ADR-009](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/ARCHITECTURE_DECISIONS.md#adr-009-multi-layer-worker-isolation-with-capability-gating). Multi-layer defense is mandatory:
  1. Format validation (disallowing non-PyTorch binaries).
  2. Enforcement of `weights_only=True` on PyTorch $\ge 2.4.0$.
  3. Mandatory execution inside disposable child worker subprocesses contained by Windows Job Objects or Linux containers.
  4. Explicit capability gating: if required security isolation is unavailable, the operation is marked `UNAVAILABLE`.

### Conflict 4: Inaccurate CVE Attribution
- **Prior Conflict**: Early drafts associated `CVE-2024-27318` with PyTorch.
- **Fact**: `CVE-2024-27318` is an ONNX directory traversal vulnerability in `onnx.load_external_data_for_model`, wholly unrelated to PyTorch.
- **Resolution**: Completely purged the erroneous association from all documentation. Reinforced that ONNX is prohibited in the technology stack.

### Conflict 5: CVE-2026-24747 Patched Version Inaccuracy
- **Prior Conflict**: Historical audit notes mentioned PyTorch 2.6.2 as the patched version for `CVE-2026-24747`.
- **Fact**: Official GitHub Security Advisory [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) states affected versions are $\le 2.9.1$ and patched versions are $\ge 2.10.0$.
- **Resolution**: Corrected across all threat models and architecture documents. The installed virtual environment (`torch==2.15.0+cpu`) is verified to be well above the patched threshold.

### Conflict 6: Dataset Licensing Claims
- **Prior Conflict**: Previous drafts made sweeping claims that datasets like CIFAR-10, GTSRB, and VOC are "forbidden for research" or "strictly non-commercial".
- **Fact**: CIFAR-10 lacks an explicit open-source license but is widely used under fair-use research conventions; GTSRB has academic-use terms; Pascal VOC has visual rights held by Flickr owners with academic research permissions; COCO splits licensing between Flickr CC image licenses and CC-BY-4.0 annotations.
- **Resolution**: Reconciled legal facts from conservative platform distribution policy:
  - VisionGuard establishes a strict platform distribution rule: **No raw dataset archives are redistributed or bundled with VisionGuard binaries**.
  - Users provide their own local dataset paths.
  - Licensing metadata is recorded per-dataset with explicit separation between image rights and annotation rights.

### Conflict 7: Inference Receipt Provability Limitations
- **Prior Conflict**: Early marketing-style descriptions implied that signed inference receipts "prove the inference was authentic and untampered."
- **Security Flaw**: A digital signature only proves that the private key holder signed the canonical payload. It cannot prove that the camera sensor was not spoofed, that memory was not modified before signing, or that the physical environment was unaltered, unless hardware-rooted attestation (TPM/Secure Enclave) is present.
- **Resolution**: Locked the honest provability boundary in [SECURITY_ARCHITECTURE.md](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/SECURITY_ARCHITECTURE.md#7-inference-provenance-and-receipt-architecture):
  - Receipts guarantee: Ingestion integrity, non-repudiation by key holder, hash-chain continuity, and detection of payload modification in transit/storage.
  - Receipts DO NOT guarantee: Ground truth of raw camera captures, immunity to malware in the same memory address space prior to signing, or uncompromised local system clocks.

### Conflict 8: Capability Reporting Model (Binary vs. 8-State)
- **Prior Conflict**: Basic test plans used binary Pass/Fail statuses. If a detector lacked labels or baselines, it was either marked failed or silently skipped.
- **Integrity Flaw**: Treating unperformed security checks as "Passed" generates false confidence; treating structural incompatibilities as "Failed" creates alert fatigue.
- **Resolution**: Locked the **8-State Capability Reporting Model** in [DATA_AND_FINDINGS_MODEL.md](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/DATA_AND_FINDINGS_MODEL.md#6-assessment-state-machine--job-lifecycle): `COMPLETED`, `NOT_APPLICABLE`, `UNAVAILABLE`, `SKIPPED`, `FAILED`, `TIMED_OUT`, `INCONCLUSIVE`, `CONFLICTED`. Furthermore, contradictory detector findings are surfaced visibly without artificial heuristic smoothing.

### Conflict 9: Detector Threshold Calibration Workflows
- **Prior Conflict**: Detector specifications suggested a global threshold for "anomalous samples".
- **Integrity Flaw**: Image duplicate thresholds, out-of-distribution thresholds, and label corruption thresholds operate on distinct distributions and metric scales. Sharing thresholds across detection modalities is mathematically invalid.
- **Resolution**: Decoupled calibration into three distinct procedures:
  - **Procedure A (Duplicate Detection)**: Cosine similarity threshold calibrated on clean reference pairs to enforce $\text{FPR} \le 0.1\%$.
  - **Procedure B (Out-of-Distribution)**: Energy / Mahalanobis score calibrated on validation splits with explicit rejection boundaries.
  - **Procedure C (Label Anomaly Detection)**: Cross-entropy margin calibrated per-class with confidence weighting.

### Conflict 10: Task Queue Architecture (Redis/Celery vs. SQLite Subprocess)
- **Prior Conflict**: Some system suggestions recommended Redis and Celery for background workers.
- **Operational Flaw**: Redis and Celery violate the offline, low-dependency, air-gapped architectural constraints of VisionGuard.
- **Resolution**: Locked [ADR-005](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/ARCHITECTURE_DECISIONS.md#adr-005-sqlite-backed-metadata-state-and-job-queue-with-subprocess-worker). All state, metadata, and task queue operations are handled by SQLite in WAL mode with worker subprocesses managed directly by the backend orchestrator.

---

## 5. Consolidated & Locked Architectural Decisions

The consolidated architecture is now locked across five authoritative documents:

1. **Modular Monolith + Ports & Adapters**: Fully specified in `SYSTEM_ARCHITECTURE.md`.
2. **Defensive Ingestion & Multi-Layer Worker Isolation**: Fully specified in `SECURITY_ARCHITECTURE.md`.
3. **Unified Findings & 8-State Lifecycle Model**: Fully specified in `DATA_AND_FINDINGS_MODEL.md`.
4. **Phase-by-Phase Roadmap (Phase 0 to Phase 9)**: Fully specified in `IMPLEMENTATION_ROADMAP.md`.
5. **Architectural Decision Records (ADR-001 to ADR-012)**: Fully specified in `ARCHITECTURE_DECISIONS.md`.

---

## 6. Decisions Still Unresolved & Requiring Future Work

While core architecture is locked, three specific technical items remain open for upstream resolution and long-term milestones:

1. **Advisory Tracking for GHSA-v32g-mr63-3j83**:  
   - Published Nov 10, 2026; reports potential arbitrary file read during `torch.load(weights_only=True)`.
   - Lacks an official CVE identifier and package specification in the current upstream advisory text.
   - *Action*: Track upstream PyTorch security bulletins; maintain current installed PyTorch (2.15.0+cpu) without downgrading; enforce worker sandbox isolation as primary defense.
2. **Hardware TPM Integration for Edge Timestamps**:  
   - Current inference receipt architecture relies on local system time and Ed25519 signatures.
   - Without a hardware Real-Time Clock (RTC) or TPM, local system clocks can be rewound or tampered with by a privileged host attacker.
   - *Action*: Deferred to Phase 8 / 9 enterprise hardware integration.
3. **Large-File Resumable Chunked Upload Protocol**:  
   - Current ingestion handles archive files through standard multi-part HTTP upload.
   - For 50GB+ vision datasets over constrained edge links, a resumable chunked protocol (e.g., TUS protocol compliant) will be evaluated in Phase 6.

---

## 7. Sign-off and Readiness for Phase 2

With Phase 1A complete:
- Architecture specifications are mathematically sound, defensively rigorous, and internally consistent.
- Boundaries, schemas, and interfaces are documented and locked.
- No code implementation for Phase 2 has commenced.
- The project is fully prepared to enter **Phase 2: Data Integrity Module — Vertical Slice 1 (Structural Ingestion & Integrity Engine)**.
