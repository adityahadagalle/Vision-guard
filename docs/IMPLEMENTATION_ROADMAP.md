# VisionGuard Implementation Roadmap

**Document Version:** 1.0.0 (Superseded & Consolidated by Phase 1A Roadmap)  
**Status:** Historical Initial Roadmap — Authoritative Consolidated Roadmap in `docs/architecture/IMPLEMENTATION_ROADMAP.md`  

> [!IMPORTANT]
> **Phase 1A Consolidated Implementation Roadmap:**
> The authoritative, locked, and consolidated Phase-by-Phase Roadmap is maintained in [`docs/architecture/IMPLEMENTATION_ROADMAP.md`](file:///C:/Users/adity/Downloads/VisionGuard_Complete_Starter_Files/VisionGuard_Starter_Pack/docs/architecture/IMPLEMENTATION_ROADMAP.md).
> It defines the complete vertical-slice sequence from Phase 0 to Phase 9, with explicit acceptance gates, verification tests, and security validation criteria.

**Approach:** research first, then build one vertical slice at a time. This is a proposed sequence, not a calendar commitment.

## Phase 0 — Research and threat modelling

**Outputs**
- Literature review for each selected detector family.
- Existing-tool comparison and documented limitations.
- Threat model and attack scenario definitions.
- Final v1 scope and acceptance criteria.
- Reproducible experiment protocol.

**Exit criteria**
- Every v1 detector has a stated purpose, prerequisites, assumptions, baseline, and evaluation method.
- Unsupported/too-expensive methods are deferred rather than represented as implemented.

## Phase 1 — Repository foundation

**Work**
- Establish Python project, dependency policy, formatting/linting, pytest, type checking decision.
- Define domain models: Asset, Assessment, DetectorRun, Finding, EvidenceReference.
- Add SQLite persistence and migrations/schema versioning approach.
- Add structured logging and configuration management.
- Create FastAPI health/version endpoints.

**Exit criteria**
- Clean local install works.
- Basic API and DB tests pass.
- No detector code yet; repository conventions are stable.

## Phase 2 — Cryptographic provenance foundation

**Work**
- SHA-256 artifact digest service.
- Canonical serialization specification for inference receipts.
- Ed25519 key generation/loading/verification interface.
- Receipt schema and signing/verification.
- Hash-chain append and verify operations.
- Checkpoint/export mechanism and tamper tests.

**Exit criteria**
- Valid records verify.
- Mutated, removed, inserted, reordered, and replayed test records are detected according to documented policy.
- Key custody limitations are documented.

## Phase 3 — Data Integrity v1

**Work**
- Dataset ingestion/validation for selected layouts.
- Exact duplicate detection by content digest.
- Near-duplicate baseline using perceptual hashing, after validation on chosen data.
- Annotation/label consistency checks for selected format(s).
- Simple reference-based OOD/anomaly baseline.
- Contributor/source/batch aggregation where metadata exists.

**Exit criteria**
- Controlled clean/anomalous dataset tests exist.
- Metrics and false-positive behavior reported.
- Sample findings link back to evidence and provenance.

## Phase 4 — Model Integrity v1

**Work**
- PyTorch artifact registration and safe-loading policy.
- Model digest matching against an approved reference.
- Architecture/task metadata and input preprocessing contract.
- Behavioral fingerprint suite on controlled inputs.
- Research and implement one backdoor-check method only if prerequisites and compute cost are manageable.

**Exit criteria**
- Substitution/byte-change tests pass.
- Fingerprints are reproducible under documented conditions.
- Backdoor capability status is honest and prerequisite-aware.

## Phase 5 — Distribution Shift v1

**Work**
- Trusted reference registration and provenance.
- Select image representation/features and baseline statistics.
- Select statistical tests/distance measures after literature review.
- Batch/window comparison and threshold calibration.
- Delayed-label performance analysis if labels become available.

**Exit criteria**
- No-shift and controlled-shift evaluation exists.
- False alarms and detection delay are measured.
- Reports distinguish drift from accuracy degradation.

## Phase 6 — Unified risk/evidence reporting

**Work**
- Common finding schema and severity rationale.
- Evidence links, detector status, configs, and asset lineage.
- Source/contributor aggregation.
- Review/accept/quarantine workflow (human decision).
- Local report export.

**Exit criteria**
- Reports can be reproduced from stored assessment metadata.
- Every output has a traceable source and no fabricated coverage.

## Phase 7 — Frontend and integration

**Work**
- React + TypeScript workspace, asset ingestion, assessment creation.
- Findings and evidence inspection.
- Method coverage and failure states.
- Local report/export interactions.

**Exit criteria**
- End-to-end user flow works locally.
- API contract and UI behavior tests pass.

## Phase 8 — Offline packaging and hardening

**Work**
- Docker build and documented offline dependency/model provisioning.
- CPU-only smoke tests; optional GPU tests.
- Input validation, archive safety, checkpoint-loading hardening.
- Resource limits, backup/restore, and error recovery.

**Exit criteria**
- Demonstrate operation with network unavailable after provisioning.
- Document clean setup, upgrade, backup, and recovery.

## Phase 9 — Final experiments and portfolio/SIH presentation

**Outputs**
- Full experimental report with seeds and versions.
- Ablation/baseline comparison where meaningful.
- Threat-to-control traceability matrix.
- Architecture diagram and demo script.
- README setup instructions and short demo video.
- Explicit limitations and future research.

## Scope control

Do not start all four detector modules in parallel. Build shared ingestion, evidence, configuration, and provenance contracts first. Keep a feature out of v1 if its prerequisites, data, or evaluation cannot be made rigorous.
