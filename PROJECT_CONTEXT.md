# VisionGuard — Project Context

## 1. One-sentence definition

VisionGuard is an offline-first AI lifecycle integrity and security assurance platform for computer-vision systems. It examines whether the data, model artifact, inference records, and incoming data distribution can be trusted, and produces findings with traceable evidence and explicit capability status.

## 2. Problem being addressed

Computer-vision pipelines may be exposed to several integrity risks:

- **Untrusted training data:** poisoned or mislabeled samples, duplicates, suspicious clusters, or data unlike the intended training distribution.
- **Compromised model artifacts:** a model may be replaced, modified, or contain a backdoor.
- **Tampered inference records:** prediction records may be edited, deleted, reordered, or replayed.
- **Distribution shift:** incoming images may differ from trusted reference data because of changes such as illumination, camera/sensor, geography, season, or environment.

The project is not claiming that all these risks can be conclusively detected in every deployment. The system must report the evidence it has, the checks it actually ran, and the limitations that remain.

## 3. Intended users

- ML engineers and computer-vision developers who manage datasets and models.
- Organizations that deploy computer-vision systems in controlled or offline environments.
- Security, governance, and audit teams that need traceable evidence about model and inference integrity.
- Researchers who need repeatable experiments for data/model attacks and defenses.

## 4. Four core modules

### A. Data Integrity

Analyze image datasets and available annotations/metadata for signals such as near-duplicates, label/annotation anomalies, unusual samples, and selected poisoning indicators. Aggregate sample-level findings by source, contributor, batch, or dataset when those metadata fields exist.

### B. Model Integrity

Record an approved model identity using cryptographic digests; verify the current artifact against an approved digest; compare model behavior under a controlled fingerprint suite; and run selected backdoor checks only when their prerequisites are met. File-hash mismatch proves artifact difference, not malicious intent. Behavioral difference is a signal, not proof of a backdoor.

### C. Inference Integrity

Create a receipt for each inference event that binds relevant identifiers and hashes (input, model, configuration, output, timestamp/sequence metadata). Sign receipts with Ed25519 and chain records so modification, deletion, insertion, or reordering can be detected relative to a trusted checkpoint. Replay detection requires a defined event identity/nonce or monotonic sequence policy; cryptography alone does not decide whether a repeated prediction is legitimate.

### D. Distribution Shift

Compare incoming batches or windows with a trusted reference dataset/statistical baseline. Distinguish input/data drift from confirmed model performance degradation: shift detection alone does not prove accuracy has fallen. Where labels arrive later, allow delayed performance evaluation.

## 5. Differentiators

1. **Contributor/source-level risk tracing:** link suspicious samples to contributor, source, batch, and provenance metadata when available.
2. **Evidence-backed findings:** show the affected asset, signals, scores, thresholds/configuration, evidence references, severity rationale, and recommended disposition.
3. **Cryptographic provenance:** use SHA-256 digests, Ed25519 signatures, and hash chaining/checkpoints to make records tamper-evident.
4. **Capability-aware reporting:** state `available`, `unavailable`, `failed`, or `not_run` for each check. Never present an unexecuted check as a successful detection.
5. **Offline-first deployment:** all required runtime components should run locally without cloud APIs.

## 6. Scope boundaries for the initial version

### In scope

- Image-based computer-vision workflows.
- PyTorch-native model artifacts (`.pt` / `.pth`) as the initial model target.
- Common dataset layouts and annotations, initially prioritizing COCO-style and YOLO-style data where practical.
- Local file ingestion, local processing, local persistence, API, and web interface.
- Controlled attack simulation and repeatable evaluation.
- Offline packaging and deployment.

### Out of scope for the initial version

- General security auditing for all AI modalities (NLP, audio, video foundation models, etc.).
- TensorFlow/Keras model support.
- ONNX / ONNX Runtime support.
- Cloud-hosted multi-tenant SaaS, remote telemetry, or mandatory external APIs.
- Automatic claims that a sample, model, or contributor is malicious.
- Fully autonomous quarantine/deletion. Human review is required for consequential actions.
- Forensic certainty or legal conclusions.

## 7. Locked technology choices

| Area | Choice |
|---|---|
| Language | Python |
| Deep-learning framework | **PyTorch only** |
| Computer vision utilities | TorchVision |
| Scientific/data utilities | NumPy, Pandas, SciPy |
| Classical ML utilities | scikit-learn, when justified |
| Backend | FastAPI |
| Local database | SQLite |
| Frontend | React + TypeScript |
| Cryptography | SHA-256, Ed25519 via a maintained Python cryptography library |
| Packaging | Docker |

Do not introduce another deep-learning framework or ONNX in the initial architecture without an explicit scope review.

## 8. Project goals and success definition

### SIH goal

Demonstrate a coherent, working solution to the stated computer-vision integrity problem, with real controlled attack scenarios, measurable results, a clear demo, and honest limitations.

### Portfolio goal

Show that the developer can design and implement a modular Python system, understand ML-security methods, build secure APIs and persistence, write tests, evaluate detection quality, and package a reproducible offline application.

### Minimum credible success

- At least one working and tested capability in each core lifecycle area, or a clearly documented reason why a specific capability is unavailable.
- Reproducible seeded attack experiments and baseline comparisons.
- Metrics that include false positives, not only successful detections.
- Verified cryptographic record-tampering tests.
- A working local UI/API flow and repeatable setup instructions.
- Clear documentation of assumptions and unsupported cases.

## 9. Non-negotiable integrity rules

- Never fabricate a detection, score, benchmark, test result, or research claim.
- Never equate an anomaly with malicious activity.
- Never equate a digest mismatch with proof of compromise.
- Never report a method as run when its prerequisites were absent.
- Preserve experiment configs, seeds, dataset versions, model digests, and software versions.
- Do not expose private data or secrets in logs/reports.
