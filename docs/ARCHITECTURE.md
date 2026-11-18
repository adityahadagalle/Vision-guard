# VisionGuard System Architecture & Pipeline Specification

**Document Version:** 1.1.0 (Superseded & Consolidated by Phase 1A Suite)  
**Status:** Historical Architectural Draft — Authoritative Consolidated Specifications in `docs/architecture/`  
**Date:** November 2026  

> [!IMPORTANT]
> **Phase 1A Consolidated Architecture Suite:**
> This document served as the initial architecture draft. The authoritative, locked, and consolidated architectural specifications are now maintained in the [`docs/architecture/`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/) suite:
> 1. [System Architecture](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/SYSTEM_ARCHITECTURE.md) — Modular monolith, Ports & Adapters, 16-step pipeline, detector plugin registry.
> 2. [Security Architecture](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/SECURITY_ARCHITECTURE.md) — Threat boundaries, defensive ingestion, worker isolation, receipt provability.
> 3. [Data & Findings Model](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/DATA_AND_FINDINGS_MODEL.md) — Asset models, snapshot schemas, universal finding schema, 8-state capability model.
> 4. [Implementation Roadmap](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/IMPLEMENTATION_ROADMAP.md) — Phase 0 to Phase 9 vertical slices and acceptance gates.
> 5. [Architecture Decisions (ADRs)](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/ARCHITECTURE_DECISIONS.md) — Consolidated ADR-001 through ADR-012.
> 6. [Architecture Consolidation Report](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/ARCHITECTURE_CONSOLIDATION_REPORT.md) — Audit and resolution log.

---

## 1. Architectural Principles & High-Level Topology

VisionGuard is designed as an offline-first, high-assurance computer vision integrity platform. Its design adheres to seven non-negotiable principles:
1. **Offline Autonomy:** Operates without outbound network connections, cloud telemetry, or paid APIs.
2. **Strict PyTorch-Only Ecosystem:** PyTorch and TorchVision are the exclusive deep-learning engines.
3. **Decoupled Architecture:** Strict separation between Ingestion, Detection Logic, API Routes, Persistence, and Cryptography.
4. **Zero-Trust Input Policy:** All uploaded datasets, archives, model weights, and logs are untrusted.
5. **Traceable Evidence & Lineage:** Every finding links to an input asset digest, detector version, configuration digest, and exact mathematical signals.
6. **Capability-Gated Execution:** Explicit execution states (`available`, `unavailable`, `failed`, `not_run`) prevent unexecuted checks from producing deceptive clean passes.
7. **Human-in-the-Loop Governance:** Automated outputs are restricted to review and quarantine suggestions (`accept`, `review`, `quarantine`). Destructive actions require operator confirmation.

```text
               Operator / Security Auditor (Local Web UI)
                                 │
                   React + TypeScript Dashboard
                                 │ (HTTP / JSON)
                       FastAPI API Layer
                                 │
                 Application Orchestration Service
                                 │
   ┌───────────────────┬─────────┴─────────┬───────────────────┐
   ▼                   ▼                   ▼                   ▼
Data Integrity   Model Integrity   Inference Integrity   Distribution Shift
Engine           Engine            Engine                Engine
(9-Stage Flow)   (Sandboxed Probes)(Receipt Chaining)    (MMD / KS Tests)
   │                   │                   │                   │
   └───────────────────┼───────────────────┴───────────────────┘
                       ▼
          Unified Evidence & Finding Aggregator
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
SQLite Metadata Store         Local Artifact Store
(Assets, Runs, Findings,      (Raw files, embeddings,
 Receipts, Audit Events)       checkpoints, manifests)
       │                               │
       └───────────────┬───────────────┘
                       ▼
         Ed25519 Signing & SHA-256 Chaining
                       │
           Self-Contained Audit Manifest
```

---

## 2. The Nine-Stage Data Integrity Pipeline

```text
[Stage 1: Dataset Input]
         │
         ▼
[Stage 2: Format Detection]
         │
         ▼
[Stage 3: Dataset Validation & Sandbox Ingestion]
         │
         ▼
[Stage 4: Image & Format-Specific Annotation Analysis]
         │
         ▼
[Stage 5: Deep Feature Extraction (Frozen Backbone)]
         │
         ▼
[Stage 6: Multi-Tier Detection Engines]
   ├── Tier 1: Exact Digests & Perceptual Hashing (SHA-256, pHash)
   ├── Tier 1: Coordinate Geometry (COCO Absolute vs. YOLO Normalized)
   ├── Tier 2: Feature-Space k-NN Label Concordance & OOD Distance
   ├── Tier 3: Spectral Signatures & Activation Clustering (Conditional)
   └── Tier 4: Contributor Attribution & LOCO Empirical Bayes
         │
         ▼
[Stage 7: Risk Aggregation & Shrinkage Modeling]
         │
         ▼
[Stage 8: Evidence Generation & Reference Linking]
         │
         ▼
[Stage 9: Structured Finding Assembly & Report Output]
```

### Detailed Stage Specifications

#### Stage 1: Dataset Input
- **Responsibility:** Ingest raw archive file (`.zip`, `.tar.gz`) or local directory path from the user.
- **Expected Input:** Archive file stream or local filesystem directory path.
- **Expected Output:** Raw file handles, initial job identifier, assessment context.
- **Dependencies:** Standard OS filesystem, safe unarchiver.
- **Failure Conditions:** File not found, permission denied, archive corrupt, file size exceeds 250 MB upload limit.
- **Validation Requirements:** File existence check, path sanitization, permission validation.

#### Stage 2: Format Detection
- **Responsibility:** Automatically identify dataset layout dialect without user intervention.
- **Expected Input:** Extracted directory tree.
- **Expected Output:** Enum format flag: `COCO_DETECTION`, `YOLO_DETECTION`, `FOLDER_PER_CLASS`, or `CUSTOM_IMAGE_FOLDER`.
- **Dependencies:** Directory inspection utilities.
- **Failure Conditions:** Ambiguous layout (e.g., both YOLO `.txt` and COCO `.json` found), zero valid images found.
- **Validation Requirements:** Checks for presence of `annotations/*.json`, `dataset.yaml`, or class subfolders.

#### Stage 3: Dataset Validation & Sandbox Ingestion
- **Responsibility:** Defend against directory traversal, decompression bombs, corrupted headers, and path exploits.
- **Expected Input:** File tree from Stage 2.
- **Expected Output:** Validated local asset directory, computed SHA-256 byte digests for all files.
- **Dependencies:** `PIL.Image`, `hashlib.sha256`.
- **Failure Conditions:** Path traversal attempt (`../`), decompression bomb ($> 40$M pixels), corrupted magic bytes.
- **Validation Requirements:** Validates magic bytes (JPEG `0xFFD8FF`, PNG `0x89504E47`), verifies image dimensions.

#### Stage 4: Image & Format-Specific Annotation Analysis
- **Responsibility:** Perform model-free structural and geometric verification of annotations and class distributions.
- **Format-Specific Geometric Rules:**
  - **COCO:** Coordinates are **absolute pixel values** `[x_min, y_min, width, height]`. Verifies $w > 0, h > 0$, $x_{\min} + w \le W_{\text{image}}$, $y_{\min} + h \le H_{\text{image}}$.
  - **YOLO:** Coordinates are **normalized** `[class_id, x_center, y_center, width, height]`. Verifies $0 \le x_c, y_c, w, h \le 1.0$, $x_c \pm w/2 \in [0, 1]$, $y_c \pm h/2 \in [0, 1]$. Class ID mapped to `dataset.yaml`. Empty label file treated as valid negative; missing label file treated as orphaned defect.
- **Dependencies:** Standard JSON parser, NumPy.
- **Failure Conditions:** Coordinate overflow, inverted boxes, missing image references.

#### Stage 5: Deep Feature Extraction
- **Responsibility:** Extract standardized deep visual embeddings using a local, frozen PyTorch backbone.
- **Expected Input:** Cleaned, readable images from Stage 3.
- **Expected Output:** Normalized feature embedding matrix $\mathbf{Z} \in \mathbb{R}^{N \times 512}$.
- **Dependencies:** PyTorch, TorchVision (frozen ResNet-18 weights loaded locally from disk).
- **Failure Conditions:** Missing local weights file, CUDA OOM (handled by automatic CPU fallback).
- **Validation Requirements:** Input tensor dimensions $[B, 3, 224, 224]$, standard ImageNet normalization applied.

#### Stage 6: Multi-Tier Detection Engines
- **Responsibility:** Execute all eligible detectors across Tiers 1 through 4 based on available inputs.
- **Expected Input:** Raw bytes, image features $\mathbf{Z}$, labels $\mathbf{y}$, optional model checkpoints, optional provenance.
- **Expected Output:** Raw anomaly score vectors per detector ($\tau_{\text{spectral}}$, $d_{\text{pHash}}$, $C_k$, $d_{\text{kNN}}$).
- **Dependencies:** SciPy (`cKDTree`, `svd`), Scikit-learn (`PCA`, `KMeans`), NumPy.
- **Failure Conditions:** Mathematical non-convergence, missing target class representation.
- **Validation Requirements:** Explicit capability status recorded for every detector in the registry.

#### Stage 7: Risk Aggregation & Shrinkage Modeling
- **Responsibility:** Aggregate sample-level anomaly scores by `contributor_id`, `source_id`, or `batch_id` using **Leave-One-Contributor-Out (LOCO) Empirical Bayes** shrinkage to eliminate self-influence and prevent small-sample distortion.
- **Expected Input:** Sample-level anomaly flags, provenance manifest records.
- **Expected Output:** Smoothed contributor risk scores ($R_{\text{smoothed}}$), posterior credible intervals, priority rankings.
- **Dependencies:** NumPy, SciPy statistical distributions.
- **Failure Conditions:** Missing provenance (gracefully falls back to sample-only reporting without failing).

#### Stage 8: Evidence Generation & Reference Linking
- **Responsibility:** Bind mathematical signals, feature coordinates, and affected asset digests into traceable evidence objects.
- **Expected Input:** Detector score outputs, sample digests, anomaly coordinates.
- **Expected Output:** List of immutable `EvidenceReference` objects.
- **Dependencies:** SQLite database engine.

#### Stage 9: Structured Finding Assembly & Report Output
- **Responsibility:** Compile all evidence into standardized `Finding` entities with severity ratings, capability matrices, and suggested dispositions (`accept`, `review`, `quarantine`).
- **Expected Input:** Evidence references, detector run statuses, assessment metadata.
- **Expected Output:** Structured JSON report and persisted database entities.

---

## 3. Sandboxed Model Deserialization Boundary

To protect the host platform from arbitrary code execution and denial-of-service memory exhaustion:

```text
Untrusted Checkpoint (.pt / .pth)
              │
              ▼
   [Ephemeral Subprocess Sandbox]
   ├── OS-Level Memory Cap: 2 GB max (Job Objects on Windows / setrlimit on Linux)
   ├── OS-Level Execution Timeout: 30 seconds hard limit
   ├── Strict Command: torch.load(path, map_location="cpu", weights_only=True)
   └── Post-Load Validation: Traverses keys, verifies strictly torch.Tensor types,
                             rejects astronomical tensor dimensions (DoS defense)
```

---

## 4. Capability-Aware Execution State Machine

```text
                    [Assessment Initiated]
                              │
                              ▼
                  [Inspect Available Assets]
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
[Prerequisites Met]   [Prerequisites Missing]   [Execution Crash]
         │                    │                    │
         ▼                    ▼                    ▼
 status: COMPLETED    status: UNAVAILABLE    status: FAILED
                      (Records Reason)       (Records Exception)
```

### Execution Status Definitions
- `available`: The detector's prerequisites (e.g., labels, model weights, reference set) are present; ready for execution.
- `completed`: The detector ran to completion and generated signals.
- `unavailable`: The detector could not run because mandatory prerequisites were absent (e.g., Spectral Signatures requested on a raw dataset without a trained model). A clear `reason` string is mandatory.
- `failed`: The detector crashed or timed out during execution. Safe error details are preserved.
- `not_run`: The detector was omitted by user configuration.

---

## 5. Core Domain Entities (Database Schema Overview)

```text
┌──────────────────┐       1:N       ┌────────────────────────┐
│      Asset       │────────────────▶│   EvidenceReference    │
└──────────────────┘                 └────────────────────────┘
         │                                       │
         │ 1:N                                   │ N:1
         ▼                                       ▼
┌──────────────────┐       1:N       ┌────────────────────────┐
│    Assessment    │────────────────▶│        Finding         │
└──────────────────┘                 └────────────────────────┘
         │                                       ▲
         │ 1:N                                   │ N:1
         ▼                                       │
┌──────────────────┐                             │
│   DetectorRun    │─────────────────────────────┘
└──────────────────┘
```

1. **`Asset`:** Represents an ingested entity (Dataset, Model, Reference Set, Provenance Manifest). Fields: `asset_id`, `asset_type`, `sha256`, `storage_path`, `source_id`, `contributor_id`, `created_at`.
2. **`Assessment`:** Represents an evaluation session. Fields: `assessment_id`, `name`, `status`, `created_at`, `config_digest`.
3. **`DetectorRun`:** Represents an execution instance of a specific detector. Fields: `run_id`, `assessment_id`, `detector_name`, `status`, `reason`, `started_at`, `duration_ms`.
4. **`EvidenceReference`:** Granular mathematical proof. Fields: `evidence_id`, `asset_id`, `sample_path`, `signal_name`, `score`, `threshold`, `details_json`.
5. **`Finding`:** Operator-facing result. Fields: `finding_id`, `assessment_id`, `asset_id`, `category`, `severity` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `title`, `description`, `suggested_disposition` (`ACCEPT`, `REVIEW`, `QUARANTINE`).
6. **`InferenceReceipt`:** Verifiable prediction record. Fields: `sequence_id`, `event_id`, `timestamp`, `input_digest`, `model_digest`, `output_payload_json`, `previous_hash`, `signature_hex`, `public_key_id`.
