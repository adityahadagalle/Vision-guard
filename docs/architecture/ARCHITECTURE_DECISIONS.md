# Architecture Decision Records (ADRs)
**Document Version:** 1.0.0  
**Status:** Approved & Locked  
**Date:** 2026-11-12  
**System:** VisionGuard AI Lifecycle Integrity & Security Platform  

---

## Document Overview

This document records the foundational Architectural Decision Records (ADRs) governing VisionGuard. Every architectural choice has been evaluated against the core requirements of:
1. **Security & Integrity Assurance**: High-confidence detection, tamper resistance, and defense against adversarial attacks and malicious payloads.
2. **Offline-First Operability**: Zero mandatory internet connectivity, cloud service dependence, or runtime external API reliance.
3. **Reproducibility & Auditability**: Deterministic pipelines, cryptographic provenance, and complete audit trails.
4. **Engineering Pragmatism**: Modular monolith simplicity, low deployment overhead, and minimal operational complexity.

---

## Index of Architecture Decision Records

- [ADR-001: PyTorch and TorchVision as Exclusive Deep Learning Framework](#adr-001-pytorch-and-torchvision-as-exclusive-deep-learning-framework)
- [ADR-002: Offline-First Autonomy and Air-Gapped Deployment Model](#adr-002-offline-first-autonomy-and-air-gapped-deployment-model)
- [ADR-003: PyTorch-Native Checkpoint Support (.pt / .pth) with Safetensors Evaluation](#adr-003-pytorch-native-checkpoint-support-pt--pth-with-safetensors-evaluation)
- [ADR-004: Modular Monolith Architecture with Ports and Adapters](#adr-004-modular-monolith-architecture-with-ports-and-adapters)
- [ADR-005: SQLite-Backed Metadata, State, and Job Queue with Subprocess Worker](#adr-005-sqlite-backed-metadata-state-and-job-queue-with-subprocess-worker)
- [ADR-006: Content-Addressed Filesystem Storage and Immutable Asset Snapshots](#adr-006-content-addressed-filesystem-storage-and-immutable-asset-snapshots)
- [ADR-007: Plugin Detector Registry with Minimal Common Interface](#adr-007-plugin-detector-registry-with-minimal-common-interface)
- [ADR-008: Shared Feature Provider with Content-Addressed Embedding Cache](#adr-008-shared-feature-provider-with-content-addressed-embedding-cache)
- [ADR-009: Multi-Layer Worker Isolation with Capability Gating](#adr-009-multi-layer-worker-isolation-with-capability-gating)
- [ADR-010: RFC 8785 Canonical JSON Serialization and Ed25519 Cryptographic Provenance](#adr-010-rfc-8785-canonical-json-serialization-and-ed25519-cryptographic-provenance)
- [ADR-011: Standalone Lightweight Runtime Client (`visionguard-runtime`)](#adr-011-standalone-lightweight-runtime-client-visionguard-runtime)
- [ADR-012: Eight-State Capability Reporting and Visible Conflict Model](#adr-012-eight-state-capability-reporting-and-visible-conflict-model)

---

## ADR-001: PyTorch and TorchVision as Exclusive Deep Learning Framework

### Status
Approved & Locked

### Context
VisionGuard inspects, validates, and monitors computer-vision models, datasets, and inference pipelines. In enterprise, defense, and research environments, multiple deep learning frameworks exist (e.g., PyTorch, TensorFlow, Keras, ONNX). Supporting multiple frameworks from the outset introduces excessive complexity, increases the attack surface (e.g., parsing untrusted graph definitions in diverse C++ engines), and fragments maintenance and validation resources.

### Decision
PyTorch (`torch`) and TorchVision (`torchvision`) are locked as the **exclusive deep learning framework and computer-vision utility stack** for VisionGuard.
- All internal model wrappers, feature extractors, embedding backbones, and neural anomaly detectors must be implemented using PyTorch.
- CPU execution must be fully supported as a baseline, with optional CUDA/ROCm hardware acceleration if locally present.
- Support for TensorFlow, Keras, ONNX, and ONNX Runtime is strictly prohibited in initial releases.

### Alternatives Considered & Reasons for Rejection
1. **TensorFlow / Keras**: Rejected due to high runtime footprint, complex serialization formats (SavedModel / HDF5 / pb) with significant historical vulnerabilities, and lower dominance in recent AI security research benchmarks.
2. **ONNX / ONNX Runtime**: Rejected due to complex C++ parsing surfaces and known historical path traversal / parsing vulnerabilities (e.g., CVE-2024-27318 in onnx). While ONNX provides framework portability, running untrusted ONNX graphs introduces high security risks in air-gapped environments without hardened runtimes.
3. **Multi-framework Abstraction Layer**: Rejected due to premature abstraction costs, leaky abstractions, and ballooning dependency size.

### Known Limitations & Assumptions
- Models trained in non-PyTorch frameworks cannot be directly assessed without conversion or export to PyTorch-compatible formats (which VisionGuard does not perform automatically).
- Assumes the host environment can install PyTorch CPU wheels (~200MB) offline.

### Decisions Requiring Future Review
- Evaluation of ONNX support may only occur in a future major milestone (Phase 9+) under an isolated, sandboxed parser container if customer demand requires cross-framework validation.

---

## ADR-002: Offline-First Autonomy and Air-Gapped Deployment Model

### Status
Approved & Locked

### Context
VisionGuard targets security-conscious deployments including defense networks, critical infrastructure, air-gapped forensic labs, and regulated enterprise environments where egress internet connectivity is disabled by policy. Standard modern software architectures frequently rely on cloud APIs, hosted databases, remote SaaS services, and dynamic package/weight downloads at runtime.

### Decision
VisionGuard is designed and implemented as an **offline-first, fully autonomous platform**.
- The entire system must function indefinitely without outbound or inbound internet connectivity.
- Dynamic runtime downloading of pretrained model weights, dataset samples, license files, or documentation is strictly forbidden.
- All dependencies, static assets, model backbones, and baseline distributions must be bundled during build time or provisioned locally by administrators via explicit filesystem mount or offline archive transfer.
- Telemetry, cloud phone-homes, and remote license verification are prohibited.

### Alternatives Considered & Reasons for Rejection
1. **Cloud-Native / SaaS Architecture**: Rejected as fundamentally incompatible with air-gapped defense and industrial requirements.
2. **Hybrid / Cloud-Assisted Fallback**: Rejected because fallback paths introduce hidden dependencies, complex failure handling when offline, and potential data leakage risks.

### Known Limitations & Assumptions
- Pretrained feature backbones (e.g., ResNet-50 weights) must be pre-packaged into offline installation bundles.
- Vulnerability feeds (e.g., PyTorch CVE databases) cannot update automatically and must be updated via offline signature/patch bundles.

### Decisions Requiring Future Review
- Structure and format of offline patch bundles and cryptographic update verification in Phase 8.

---

## ADR-003: PyTorch-Native Checkpoint Support (.pt / .pth) with Safetensors Evaluation

### Status
Approved & Locked

### Context
Computer vision model artifacts are predominantly distributed as `.pt` or `.pth` files. In standard Python/PyTorch, checkpoint loading involves `pickle` deserialization (`torch.load`), which possesses inherent arbitrary code execution risks if an attacker crafts malicious opcodes within the pickled stream. Recently, Hugging Face `safetensors` has emerged as a safer serialization alternative, but raw PyTorch checkpoints remain the primary industry format.

### Decision
VisionGuard will provide **PyTorch-native checkpoint support (`.pt` / `.pth`)** as its primary target, governed by a rigorous defensive loading protocol:
1. All checkpoint files are treated as untrusted bytecode.
2. Direct, unconstrained `torch.load` is strictly prohibited.
3. The system enforces `torch.load(..., weights_only=True)` on PyTorch versions $\ge 2.4.0$ (installed baseline is PyTorch 2.15.0+cpu).
4. Deserialization must execute inside an isolated worker process sandbox.
5. In Phase 3, an architectural evaluation of `safetensors` will be conducted to establish whether dual-support or an automated conversion utility is warranted.

### Alternatives Considered & Reasons for Rejection
1. **Rejecting `.pt`/`.pth` in Favor of Safetensors Exclusively**: Rejected because real-world security evaluation mandates inspecting existing user `.pt` files. Refusing `.pt` would render VisionGuard useless for assessing the vast majority of deployed vision models.
2. **Unsandboxed `torch.load` with Bytecode Scanning**: Rejected because heuristic bytecode scanners (e.g., scanning for `posix.system` strings) can be bypassed by polymorphic or obfuscated pickle payloads.

### Known Limitations & Assumptions
- Memory corruption vulnerabilities in PyTorch C++ unpicklers (e.g., CVE-2026-24747 in $\le 2.9.1$) highlight that `weights_only=True` alone is not a complete sandbox. Hence, process isolation (Job Objects / Linux namespaces) is mandatory.
- Residual risk: Zero-day memory corruption in the PyTorch unpickler could compromise the isolated worker process; host isolation prevents privilege escalation to the orchestrator.

### Decisions Requiring Future Review
- Phase 3 milestone: Formally evaluate `safetensors` adoption and benchmark its throughput and safety guarantees against `weights_only=True`.

---

## ADR-004: Modular Monolith Architecture with Ports and Adapters

### Status
Approved & Locked

### Context
Software architectures often polarize between distributed microservices and tightly coupled monostructure systems. Microservices introduce network RPC overhead, complex deployment orchestration (Kubernetes, service meshes), network security perimeter challenges, and difficult local debugging. Conversely, an unstructured monolith leads to entangled dependencies, spaghetti code, and accidental tight coupling between domain modules.

### Decision
VisionGuard adopts a **Modular Monolith architecture with Ports and Adapters (Hexagonal Architecture)**.
- Single unified deployable codebase structured into distinct modules: `backend/app/core/`, `backend/app/shared/`, `backend/app/data_integrity/`, `backend/app/model_integrity/`, `backend/app/inference_integrity/`, and `backend/app/distribution_shift/`.
- **Ports & Adapters**: Domain logic within modules must not import database models, filesystem I/O, or web frameworks directly. Modules define abstract Ports (protocols), implemented by Adapters in the infrastructure layer.
- **Dependency Rule**: Dependencies point inward toward the domain. Domain modules communicate exclusively through Shared Kernel interfaces (`shared/domain/` and `shared/ports/`), never importing internal logic from sibling domain modules.
- Single process tree with isolated subprocess workers for execution tasks.

### Alternatives Considered & Reasons for Rejection
1. **Microservices Architecture (Docker-Compose / K8s multi-container)**: Rejected due to unnecessary operational complexity for air-gapped single-node or edge deployments, excessive resource overhead (multiple runtimes), and complex distributed transaction management.
2. **Layered Monolith (Classic 3-Tier)**: Rejected because layered monoliths frequently result in circular dependencies across domain boundaries and tight coupling to database schemas.

### Known Limitations & Assumptions
- Scales vertically on a single host. Horizontal scaling across a cluster is deferred to future enterprise architectures.

### Decisions Requiring Future Review
- None for the core platform lifecycle (Phases 1–7).

---

## ADR-005: SQLite-Backed Metadata, State, and Job Queue with Subprocess Worker

### Status
Approved & Locked

### Context
Asynchronous job processing, queue management, and relational metadata persistence are required for long-running dataset hashing and model inspections. Standard stacks utilize PostgreSQL, Redis, and Celery. In air-gapped, lightweight, or embedded installations, administering external database servers and message broker daemons creates significant deployment hurdles, failure modes, and security surface.

### Decision
VisionGuard locks **SQLite (in WAL mode)** for metadata persistence and state management, combined with a **database-backed job queue** and a dedicated **subprocess worker architecture**:
- `sqlite3` in WAL mode (`PRAGMA journal_mode=WAL;`, `PRAGMA busy_timeout=30000;`, `PRAGMA foreign_keys=ON;`) enables concurrent readers alongside serialized atomic writers without lock contention.
- **Atomic Job Claiming**: Uses single-statement `UPDATE jobs SET status='RUNNING', ... WHERE job_id = (SELECT job_id FROM jobs WHERE status='QUEUED' ORDER BY created_at ASC LIMIT 1) RETURNING job_id;` guaranteeing race-free serialization across multiple OS processes without non-functional `SELECT FOR UPDATE`.
- **Lease Fencing & Monotonic Generations**: Each job claim issues a cryptographic `lease_token` (UUID), sets `lease_expires_at`, and increments `lease_generation`. Job completion and failure require the matching active `lease_token` and unexpired lease timestamp, preventing stale or zombie workers from overwriting new worker results.
- **Startup Reconciliation**: Reconciles only genuinely expired stranded jobs (`lease_expires_at < now`). Unexpired leases from active concurrent workers are strictly preserved across server restarts.
- **Retry Safety & Idempotent Publication**: Ingestion verifies and reuses matching disk snapshots while purging corrupted staging directories. Assessment result publication atomically purges previous runs and publishes runs, findings, and evidence in a single transaction, eliminating primary key collision errors on job retries.
- Microservices, Redis, Celery, RabbitMQ, and external SQL daemons are explicitly rejected for the core distribution.

### Alternatives Considered & Reasons for Rejection
1. **PostgreSQL + Redis + Celery**: Rejected due to high operational complexity, multi-process daemon management, port exposure requirements, and heavy footprint on edge devices.
2. **Pure In-Memory Queue (Python `asyncio.Queue`)**: Rejected because jobs and state do not survive server restarts, preventing resumption of multi-gigabyte dataset assessments.
3. **`SELECT ... FOR UPDATE` with SQLite**: Rejected because SQLite does not implement row-level locks or `FOR UPDATE` semantics; atomic `UPDATE ... RETURNING` is utilized instead.

### Known Limitations & Assumptions
- Single-host write throughput limit of SQLite (thousands of writes/sec, far exceeding VisionGuard's assessment logging requirements).
- Concurrent database writers are coordinated through SQLite WAL mode, `busy_timeout=30000ms`, and immediate write serialization.

### Decisions Requiring Future Review
- Phase 7+: Benchmark queue throughput under continuous multi-worker stress; evaluate external Postgres adapter only if multi-node clustering is requested as an optional plugin.

---

## ADR-006: Content-Addressed Filesystem Storage and Immutable Asset Snapshots

### Status
Approved & Locked

### Context
Assessing datasets and models requires verifiable guarantees that the files inspected are identical to the files audited and reported. If files can be modified in-place during or after assessment, audit findings become invalid, susceptible to time-of-check to time-of-use (TOCTOU) tampering. Furthermore, duplicate copies of multi-gigabyte datasets waste disk space.

### Decision
VisionGuard adopts **Content-Addressed Filesystem Storage with Immutable Asset Snapshots**:
1. All ingested raw archives enter an isolated `data/quarantine/` staging directory.
2. Upon successful quarantine validation, files are unpacked and moved into a content-addressed store (`data/assets/{asset_id}/snapshots/{snapshot_id}/`) named by the SHA-256 digest of the entire manifest.
3. Once created, snapshot directories are marked read-only at the filesystem level as an operational protection against accidental modification. Note that read-only file permissions provide operational defense rather than adversarial kernel-level immutability; integrity is cryptographically enforced via authoritative manifest digests.
4. A canonical `manifest.json` maps relative file paths to their exact SHA-256 hashes, file sizes, and MIME types.
5. In-place modification of existing snapshots is strictly prohibited; updates create a new distinct snapshot ID.

### Alternatives Considered & Reasons for Rejection
1. **In-Place File Inspection**: Rejected because modifying datasets while running detectors creates race conditions and renders cryptographic audit trails invalid.
2. **Object Storage Engine (MinIO / S3 API)**: Rejected due to added daemon overhead, network port exposure, and unnecessary complexity over native POSIX/NTFS content-addressed directories.

### Known Limitations & Assumptions
- Requires sufficient local storage to maintain snapshots. Deduplication across snapshots relies on hardlinks or shared chunk stores (to be evaluated in storage optimization).
- Filesystem read-only permissions do not prevent a local process running under the same user account from calling chmod to overwrite files; cryptographic verification is required to detect adversarial tampering.

### Decisions Requiring Future Review
- Storage retention policies and automated pruning of old snapshots (Phase 6).

---

## ADR-007: Plugin Detector Registry with Minimal Common Interface

### Status
Approved & Locked

### Context
VisionGuard incorporates diverse integrity detectors across four major modules: image corruption, label noise, duplicate detection, backdoor weight inspection, distribution drift, etc. Hardcoding detector invocations inside monolithic business logic hampers testability, prevents pluggability, and makes adding new detection techniques fragile. Conversely, over-engineered plugin frameworks (e.g., loading arbitrary third-party Python modules dynamically) introduce security risks.

### Decision
VisionGuard locks a **Plugin Detector Registry with a Minimal Common Interface**:
- Every detector implements the `BaseDetector` abstract protocol:
  - `detector_id: str`
  - `version: str`
  - `requirements: DetectorRequirements`
  - `evaluate(context: AssessmentContext) -> DetectorResult`
- Explicit runtime registration: Detectors are explicitly registered at runtime in an in-memory catalog (`DetectorRegistry.register(...)`). Unsafe runtime dynamic module scanning (`pkg_resources`, untrusted `importlib`) is strictly prohibited.
- Detectors are stateless; all dependencies, configs, and inputs are passed via `AssessmentContext`.
- Detectors must never swallow exceptions; if a prerequisite is missing (e.g., no labels present for a label anomaly detector), the detector emits a capability record with status `NOT_APPLICABLE` or `UNAVAILABLE` rather than failing silently.

### Alternatives Considered & Reasons for Rejection
1. **Dynamic Python Module Discovery (`pkg_resources` / `importlib.import_module` on arbitrary dirs)**: Rejected as an untrusted code execution vulnerability in enterprise environments.
2. **Monolithic Procedural Pipeline**: Rejected because adding or disabling detectors requires modifying core orchestrator loops, violating the Open/Closed Principle.

### Known Limitations & Assumptions
- All built-in detectors must be compiled/packaged with the core application.

### Decisions Requiring Future Review
- Formal schema for signed third-party detector plugins (Phase 9+).

---

## ADR-008: Shared Feature Provider with Content-Addressed Embedding Cache

### Status
Approved & Locked

### Context
Multiple detectors across Data Integrity (duplicate detection, out-of-distribution detection) and Distribution Shift (Wasserstein drift, MMD, feature drift) require deep feature representations (embeddings) of image datasets. Computing embeddings using a deep neural network (e.g., ResNet-50) is the most computationally expensive operation in VisionGuard. If each detector independently extracts features, runtime balloons by $4\times\text{--}6\times$, causing severe CPU/GPU contention.

### Decision
VisionGuard implements a **Shared Feature Provider with an SQLite + Filesystem Content-Addressed Embedding Cache**:
1. Feature extraction is decoupled into a shared service (`SharedFeatureProvider`).
2. Embeddings are cached using a composite key: `cache_key = SHA256(sample_sha256 + backbone_name + weights_sha256 + layer_name)`.
3. Detectors declare their feature requirements in their metadata. The orchestrator batches feature extraction once per snapshot and passes feature tensors to all subscribed detectors.
4. If a detector requires raw pixel access (e.g., Tier-1 corruption checks), it bypasses the feature provider.

### Alternatives Considered & Reasons for Rejection
1. **Detector-Internal Feature Extraction**: Rejected due to massive redundant computation, repeated disk reads, and excessive power/time consumption.
2. **Mandatory Pre-Extraction for All Assets**: Rejected because lightweight assessments (e.g., manifest verification or structural checks) should not incur heavy neural network inference delays.

### Known Limitations & Assumptions
- Feature caching requires local disk space for embedding storage (typically float32 arrays compressed with NPZ or SQLite BLOBs).
- Backbones must produce deterministic outputs across runs given identical inputs.

### Decisions Requiring Future Review
- Eviction policies for the embedding cache when disk thresholds are exceeded (Phase 6).

---

## ADR-009: Multi-Layer Worker Isolation with Capability Gating

### Status
Approved & Locked

### Context
Ingesting untrusted datasets and inspecting untrusted model checkpoints introduces severe risks: zip bombs, decompression path traversals, PIL image parser vulnerabilities, and PyTorch deserialization exploits. Running untrusted parsing inside the main web server process could crash or compromise the entire platform.

### Decision
VisionGuard enforces **Multi-Layer Worker Isolation with Capability Gating**:
1. All parsing of untrusted archives, image decoding, and model checkpoint unpickling executes in dedicated, disposable worker subprocesses.
2. **Platform-Specific Containment**:
   - **Linux**: Target production worker isolation uses hardened Linux containers or unprivileged namespaces with network isolation (`CLONE_NEWNET`), read-only root filesystems, drop-all capabilities, and explicit seccomp filters.
   - **Windows & Host Environments**: Workers run in dedicated child processes providing process separation, execution timeouts, streaming uncompressed byte limits, path containment checks, and duplicate collision tracking. Note: This provides process separation and failure isolation, not an OS-level kernel sandbox. Windows Job Object kernel quotas and descendant-process kill containment represent a Phase 8 containerization roadmap rather than baseline OS sandboxing.
3. **Capability Gating**: VisionGuard explicitly recognizes that standard subprocess separation does not provide full kernel security isolation. When hardened container isolation is mandated by policy (`require_hardened_isolation=True`), the system raises `WorkerIsolationUnavailableError` and marks the assessment status as `UNAVAILABLE` rather than falsely claiming sandboxed execution.

### Alternatives Considered & Reasons for Rejection
1. **In-Process Parsing with Exception Handlers**: Rejected as completely unsafe against memory corruption, denial-of-service, or remote code execution exploits.
2. **Assuming Subprocess Separation Provides Root Isolation**: Rejected as technically inaccurate; standard child processes inherit host user privileges, access to readable files, and lack OS-level memory quotas unless wrapped in a container runtime.

### Known Limitations & Assumptions
- On developer Windows workstations without container virtualization, extraction and checkpoint loading execute with process separation, execution timeouts, and path validation. OS-enforced memory limits and descendant process trees require container virtualization (Phase 8).
- Model checkpoint evaluation is restricted to `weights_only=True` and static file inspection.

### Decisions Requiring Future Review
- Integration of microVMs (e.g., Firecracker / WSL2 container backend) for advanced Windows deployment isolation in Phase 8.

---

## ADR-010: Deterministic Key-Sorted JSON Serialization and Cryptographic Provenance

### Status
Approved & Locked

### Context
VisionGuard produces audit findings, dataset manifests, and inference receipts that must be cryptographically signed to guarantee authenticity and non-repudiation. In JSON, identical semantic objects can produce divergent byte strings due to whitespace, key ordering, number formatting, and Unicode escaping. Signing non-canonical JSON leads to verification failures across different platforms, languages, or deserializers.

### Decision
VisionGuard adopts **Deterministic Key-Sorted JSON Serialization** and **Ed25519 digital signatures** for cryptographic provenance:
1. Every signable entity (Inference Receipt, Assessment Report, Dataset Manifest) specifies a strict, schema-validated representation.
2. For internal manifest generation and snapshot sealing, the payload is serialized deterministically using lexicographically sorted keys, compact separators (`","`, `":"`), and UTF-8 encoding via `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
3. Note on RFC 8785: While modeled after JSON canonicalization principles, the baseline Python `json.dumps` implementation does not strictly enforce ECMAScript IEEE 754 float representation (e.g. `1.0` vs `1`) or UTF-16 code-unit key sorting for non-BMP Unicode characters. A standalone, verified RFC 8785 serializer will be evaluated for the independent runtime client in Phase 5 / Phase 8.
4. Signatures are generated and verified using Ed25519 (`cryptography.hazmat.primitives.asymmetric.ed25519`).
5. Verification algorithms must verify both the hash chain linkage and the Ed25519 cryptographic signature.

### Alternatives Considered & Reasons for Rejection
1. **Unsorted Ad-Hoc JSON Serialization**: Rejected because non-deterministic key ordering and formatting differences cause cryptographic signature verification to fail unpredictably.
2. **RSA-2048 / RSA-4096**: Rejected due to larger signature and key sizes, higher computational overhead on edge devices, and vulnerability to padding oracle attacks.

### Known Limitations & Assumptions
- In-process deterministic serialization relies on standard library `json.dumps(sort_keys=True)`; cross-language verification across diverse platforms must account for float serialization semantics until a dedicated RFC 8785 serializer is embedded.

### Decisions Requiring Future Review
- Full RFC 8785 ECMAScript float canonicalizer integration for cross-platform signing verification (Phase 5 / Phase 8).

---

## ADR-011: Standalone Lightweight Runtime Client (`visionguard-runtime`)

### Status
Approved & Locked

### Context
Inference Integrity requires capturing inference events, computing cryptographic input/output hashes, and signing inference receipts at the point of inference (edge camera, server inference container). The main VisionGuard platform includes heavy dependencies: FastAPI, PyTorch, TorchVision, SciPy, Pandas (~2GB disk footprint). Requiring edge devices running lightweight inference engines (e.g., PyTorch Mobile, TensorRT, or dedicated vision gateways) to install the full VisionGuard backend is completely infeasible.

### Decision
VisionGuard splits the inference receipt generation into a **standalone, zero-dependency lightweight client package: `visionguard-runtime`**:
1. `visionguard-runtime` depends only on Python standard libraries plus `cryptography` (for Ed25519 and SHA-256).
2. It contains zero machine learning frameworks, zero web servers, and zero heavy math libraries (<5MB total size).
3. It exposes a minimal, high-throughput API: `ReceiptEmitter.sign(input_bytes, output_prediction, metadata) -> InferenceReceipt`.
4. The generated canonical receipt is written to a local append-only log or forwarded to the VisionGuard backend asynchronously.
5. Verification, receipt chain validation, and anomaly detection are performed exclusively on the VisionGuard platform side.

### Alternatives Considered & Reasons for Rejection
1. **Unified Python Package**: Rejected because shipping PyTorch to low-power edge capture devices running embedded vision runtimes creates severe dependency and resource conflicts.
2. **Direct REST API Calls for Every Inference**: Rejected because network roundtrips introduce unacceptable latency (10–50ms) into high-fps video inference loops and fail when disconnected.

### Known Limitations & Assumptions
- The runtime client assumes the edge environment has access to a secure private key storage mechanism (e.g., environment variable, filesystem permissions, or local TPM).

### Decisions Requiring Future Review
- C/C++ or Rust bindings for `visionguard-runtime` for integration into non-Python embedded edge devices (Phase 9+).

---

## ADR-012: Eight-State Capability Reporting and Visible Conflict Model

### Status
Approved & Locked

### Context
Automated integrity tools frequently compress nuanced assessment results into binary "Pass/Fail" scores. In complex security assessments, an algorithm may not run because required metadata is absent (e.g., no clean reference distribution for drift), because the host environment lacks necessary isolation, or because two detectors produce mutually contradictory evidence (e.g., label noise vs. out-of-distribution sample). Silently treating an unrun check as "Passed" creates dangerous false assurances.

### Decision
VisionGuard implements an **Eight-State Capability Reporting Model** and a **Visible Conflict Findings Model**:
1. Every detector assessment must report its execution status using one of eight formal states:
   - `COMPLETED`: Detector ran to full completion with valid results.
   - `NOT_APPLICABLE`: Detector cannot evaluate the asset due to valid structural reasons (e.g., bounding-box validation on classification datasets).
   - `UNAVAILABLE`: Detector is disabled due to missing system capabilities or sandbox requirements.
   - `SKIPPED`: Detector skipped due to user configuration or dependency failure.
   - `FAILED`: Execution crashed or threw an unhandled exception.
   - `TIMED_OUT`: Execution exceeded its configured timeout threshold.
   - `INCONCLUSIVE`: Execution finished, but statistical confidence was insufficient to reach a conclusive finding.
   - `CONFLICTED`: Execution completed, but its finding directly contradicts another detector's finding on the same asset.
2. **Visible Conflict Model**: Conflicting findings are never averaged, hidden, or arbitrarily resolved by heuristics. The platform surfaces both findings explicitly in the report, flags the conflict status, and alerts the human auditor.

### Alternatives Considered & Reasons for Rejection
1. **Binary Pass / Fail Status**: Rejected because it conceals unperformed checks as clean, creating severe liability and false trust.
2. **Majority-Vote Heuristic Resolution**: Rejected because in adversarial scenarios, a sophisticated attack may trigger a single specialized detector while bypassing two weaker heuristic detectors; majority voting would dangerously silence the true positive.

### Known Limitations & Assumptions
- Requires human review and domain expertise to resolve flagged `CONFLICTED` or `INCONCLUSIVE` assessments.

### Decisions Requiring Future Review
- Refinement of formal automated conflict detection rules between specific pairs of detectors during Phase 4 & Phase 5.

---

## Summary Matrix of Architectural Decisions

| ADR ID | Decision Scope | Core Technology / Pattern | Key Constraint / Invariant |
|---|---|---|---|
| **ADR-001** | Deep Learning Framework | PyTorch & TorchVision | Exclusive framework; zero TF, Keras, ONNX |
| **ADR-002** | Deployment Autonomy | Offline-First Air-Gapped | Zero runtime internet connectivity or cloud APIs |
| **ADR-003** | Model Checkpoints | PyTorch-Native (`.pt`/`.pth`) | Strict `weights_only=True` inside isolated worker |
| **ADR-004** | Architecture Pattern | Modular Monolith (Ports & Adapters) | Modules communicate only via Shared Kernel ports |
| **ADR-005** | State & Job Processing | SQLite (WAL) + Subprocess Worker | Zero Redis, Celery, or external database daemons |
| **ADR-006** | File & Asset Storage | Content-Addressed Snapshots | Immutable read-only directories; canonical manifests |
| **ADR-007** | Detector Architecture | Plugin Detector Registry | Common `BaseDetector` protocol; zero runtime imports |
| **ADR-008** | Feature Computation | Shared Feature Provider | Unified embedding cache keyed by SHA-256 digests |
| **ADR-009** | Execution Security | Multi-Layer Worker Isolation | Container/Job Object isolation; capability gating |
| **ADR-010** | Cryptographic Provenance | RFC 8785 JCS + Ed25519 | Canonical serialization before hashing and signing |
| **ADR-011** | Inference Client | Standalone `visionguard-runtime` | Zero-ML lightweight package (<5MB) for edge devices |
| **ADR-012** | Assessment Reporting | 8-State Capability & Visible Conflict | Never hide unrun checks or silence conflicting detectors |
