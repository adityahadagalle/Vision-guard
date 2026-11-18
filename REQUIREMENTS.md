# VisionGuard Requirements

**Status:** Initial requirements baseline. Requirements are intentionally testable, but thresholds and some file-format details remain TBD after research.

## 1. Functional requirements

### Project and asset management

- **FR-001:** The system shall create a local project/workspace for an assessment.
- **FR-002:** The system shall register dataset, model, reference-set, configuration, and inference-log assets with unique internal IDs.
- **FR-003:** The system shall record asset provenance fields when supplied: source, contributor, batch, creation/import time, and version.
- **FR-004:** The system shall calculate SHA-256 digests for ingested artifacts and preserve the digest with the assessment metadata.

### Data integrity

- **FR-010:** The system shall accept supported image dataset layouts and validate their structure before analysis.
- **FR-011:** The system shall identify exact and near-duplicate images using configurable methods.
- **FR-012:** The system shall calculate label/annotation consistency signals for supported annotation formats.
- **FR-013:** The system shall calculate configurable outlier/OOD-related signals using an explicitly documented representation and reference strategy.
- **FR-014:** The system shall run selected poisoning indicators only when required labels/model features/prerequisites are available.
- **FR-015:** The system shall aggregate sample findings by contributor/source/batch when provenance metadata exists.
- **FR-016:** The system shall preserve per-sample evidence references and detector/config versions.

### Model integrity

- **FR-020:** The system shall register an approved PyTorch model artifact and its SHA-256 digest.
- **FR-021:** The system shall compare a candidate artifact digest with the approved digest and report match/mismatch.
- **FR-022:** The system shall run a configurable behavioral fingerprint suite when an executable model and valid test inputs are available.
- **FR-023:** The system shall run selected backdoor checks only when their prerequisites are met.
- **FR-024:** The system shall record whether checks were available, unavailable, failed, or not run, with a reason.
- **FR-025:** The system shall distinguish artifact difference, behavioral difference, anomaly indicators, and confirmed experimental attack labels.

### Inference integrity

- **FR-030:** The system shall create an inference receipt binding configured input, model, configuration, output, and event metadata.
- **FR-031:** The system shall serialize receipt data canonically before hashing/signing.
- **FR-032:** The system shall sign receipts with Ed25519 using locally managed keys.
- **FR-033:** The system shall append receipts to a hash-linked sequence.
- **FR-034:** The system shall verify signatures and chain continuity.
- **FR-035:** The system shall detect tampering, deletion, insertion, and reordering relative to a trusted chain/checkpoint.
- **FR-036:** The system shall implement a documented replay policy using event IDs, nonces, or monotonic sequence constraints.

### Distribution shift

- **FR-040:** The system shall register a trusted reference dataset or reference statistics with provenance and version information.
- **FR-041:** The system shall compare incoming data batches/windows with the selected reference using documented features and statistical tests.
- **FR-042:** The system shall report which feature(s), tests, windows, and thresholds contributed to a shift finding.
- **FR-043:** The system shall distinguish detected input shift from measured model performance degradation.
- **FR-044:** The system shall support delayed evaluation when labels become available, subject to the selected workflow.

### Findings, reports, and interface

- **FR-050:** Every finding shall identify the affected asset and the detector(s) that generated it.
- **FR-051:** Every finding shall include reason, evidence references, relevant scores/statistics, configuration/version, status, and severity rationale.
- **FR-052:** The system shall offer a suggested disposition such as `accept`, `review`, or `quarantine`; the initial system shall not automatically delete assets.
- **FR-053:** Reports shall include summary results, method coverage/capability status, limitations, and reproducibility metadata.
- **FR-054:** The frontend shall allow a user to initiate assessments, inspect findings, and export reports locally.
- **FR-055:** The application shall operate locally without mandatory external API calls.

## 2. Non-functional requirements

- **NFR-001 Security:** Validate all untrusted uploads and model files; isolate risky processing where practical.
- **NFR-002 Integrity:** Verify cryptographic signatures and hash-chain links rather than merely generating them.
- **NFR-003 Reproducibility:** Store seeds, versions, parameters, and input digests for experiments.
- **NFR-004 Auditability:** Preserve who/what initiated an assessment and the resulting state transitions, without logging unnecessary sensitive content.
- **NFR-005 Offline operation:** Core application functionality must not depend on network access.
- **NFR-006 Portability:** Support documented local development and Docker-based deployment.
- **NFR-007 Maintainability:** Separate API, business logic, ML detectors, cryptography, persistence, and UI.
- **NFR-008 Performance:** Measure CPU-only latency and peak memory on documented hardware; define acceptance targets only after baseline measurement.
- **NFR-009 Reliability:** Failed or unavailable checks must not silently produce a passing result.
- **NFR-010 Usability:** Findings must explain technical signals in understandable language without overstating certainty.

## 3. Acceptance criteria baseline

A requirement is not complete merely because a function exists. Demonstrate it with tests and reproducible evidence.

- Data detectors: evaluate on clean and controlled anomalous samples; report precision, recall, F1, and false-positive rate where ground truth exists.
- Model integrity: verify exact digest match/mismatch behavior; test behavioral fingerprint reproducibility and capability gating.
- Inference integrity: test valid chain verification and tampering, deletion, insertion, reordering, and replay cases.
- Distribution shift: test known no-shift and shifted datasets; measure false alarms and detection behavior under controlled conditions.
- Reporting: verify every finding links to evidence and records method status.
- Offline: demonstrate operation with network disabled after installation and asset provisioning.

## 4. Open requirements

- Final supported annotation dialects and edge cases.
- Dataset upload limits and maximum image dimensions.
- Exact detector list for the first release.
- Threshold calibration strategy per detector.
- Key generation, storage, backup, rotation, and revocation policies.
- Report export format(s).
- Performance targets after profiling on the target machine.
