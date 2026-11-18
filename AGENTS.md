# Instructions for AI Coding Agents

These instructions apply to every coding agent working in this repository.

## 1. Read before coding

Before modifying code, read:
1. `PROJECT_CONTEXT.md`
2. `REQUIREMENTS.md`
3. `TECH_STACK.md`
4. `docs/ARCHITECTURE.md`
5. The relevant module specification under `docs/module_specs/`
6. Relevant testing/evaluation documents

Inspect the existing source and tests before proposing changes. Do not assume the repository is empty or overwrite existing work.

## 2. Project identity and fixed scope

VisionGuard is an offline-first AI lifecycle integrity and security assurance platform for computer-vision systems. Its four modules are:
- Data Integrity
- Model Integrity
- Inference Integrity
- Distribution Shift

The project's primary goals are SIH 2026 and a strong engineering/research portfolio.

## 3. Fixed technology constraints

- Python is the primary backend/ML language.
- **PyTorch is the only deep-learning framework.**
- TorchVision may be used for computer-vision utilities/models/transforms.
- NumPy, Pandas, SciPy, and scikit-learn are supporting libraries where justified.
- FastAPI, SQLite, React, TypeScript, SHA-256, Ed25519, and Docker are the selected application/security stack.
- Initial model artifact support is PyTorch-native `.pt` / `.pth`.

### Prohibited in the initial scope

- TensorFlow or Keras.
- ONNX or ONNX Runtime.
- Mandatory cloud services or paid APIs.
- Runtime dependence on internet access.
- New databases, brokers, orchestration systems, or frameworks without an explicit architecture decision.

Do not silently add dependencies. Explain why each new dependency is needed and identify its offline/runtime implications.

## 4. Engineering standards

- Use small modules with clear responsibilities.
- Use type hints for public Python functions and data structures.
- Validate untrusted inputs at system boundaries.
- Use explicit configuration rather than hard-coded thresholds or paths.
- Use structured errors; do not swallow exceptions.
- Keep I/O, ML logic, cryptographic logic, API routes, and persistence separate.
- Add unit tests for core logic and integration tests for component boundaries.
- Keep functions deterministic where practical; expose random seeds in experiment configs.
- Avoid global mutable state.
- Do not put secrets, private datasets, model weights, or local databases in Git.
- Add docstrings/comments for non-obvious security decisions, not comments that merely restate code.

## 5. ML and security integrity rules

- Do not call an anomaly a confirmed attack unless the experiment provides that ground truth.
- Distinguish detection score, decision threshold, calibrated confidence, and statistical p-value. They are not interchangeable.
- Report per-method capability status: `available`, `unavailable`, `failed`, or `not_run`.
- If a technique needs activations, labels, a clean reference set, model internals, or a specific architecture, validate that prerequisite before running it.
- Do not invent default thresholds. Document how thresholds are selected and validated.
- Hash the exact bytes of an artifact and record the hashing procedure.
- A hash mismatch means the bytes differ from the approved artifact; it does not by itself prove malicious modification.
- For inference provenance, define canonical receipt serialization before hashing/signing.
- Verify signatures and chain links; do not merely create them.
- Account for missing, duplicated, reordered, and replayed events in tests.
- Treat model files as untrusted. PyTorch checkpoint loading may involve pickle-based deserialization. Do not blindly load arbitrary files from untrusted sources. Prefer state dictionaries and safe/restricted loading options supported by the installed PyTorch version; document residual risk.

## 6. Offline-first rules

- The application must work without external APIs or cloud access after dependencies/assets are installed.
- Do not download weights or datasets implicitly at runtime.
- Make any pretrained weights optional and locally provisioned with documented provenance.
- Keep dependency versions pinned or constrained in reproducible environment files.
- Document how images, libraries, and containers are transferred into air-gapped environments.

## 7. API, data, and privacy rules

- Validate file size, file type, dimensions, archive paths, and metadata before processing.
- Defend against path traversal and decompression bombs when accepting archives.
- Use generated internal IDs; do not trust user-supplied IDs as filesystem paths.
- Avoid logging raw image content, secrets, private file paths, or unnecessary personal data.
- Make destructive actions (delete/quarantine/overwrite) explicit and auditable; default to review rather than automatic destruction.
- Keep database transactions and filesystem state consistent; document recovery behavior.

## 8. Required workflow for each task

1. Restate the requested scope and acceptance criteria briefly.
2. Inspect relevant code and docs.
3. Propose the smallest implementation plan and files to change.
4. Implement only the approved slice.
5. Add/update tests.
6. Run relevant tests and report exact commands and results.
7. Update docs/config/examples where behavior changed.
8. List known limitations and follow-up work.

Do not claim tests pass unless they were actually run and passed. If blocked, state the exact blocker.

## 9. Output expectations

At the end of each task, report:
- Summary of behavior implemented.
- Files added/modified.
- Tests executed and actual result.
- Any migration/configuration changes.
- Limitations, security assumptions, and unresolved risks.

Do not generate the entire application in one shot. Implement one vertical slice at a time.
