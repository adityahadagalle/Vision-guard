# VisionGuard Technology Stack

## Locked stack

| Layer | Technology | Responsibility | Status |
|---|---|---|---|
| Main language | Python | Backend, data pipeline, ML/security logic | LOCKED |
| Deep-learning framework | **PyTorch only** | Model loading/inspection and feature extraction | LOCKED |
| Vision utilities | TorchVision | Image transforms and compatible CV utilities | LOCKED |
| Numerical computing | NumPy, SciPy | Arrays, statistics, signal processing | LOCKED |
| Data handling | Pandas | Tabular metadata and reports | LOCKED |
| Classical ML | scikit-learn | Selected anomaly/statistical baselines | LOCKED as supporting utility |
| Backend | FastAPI | Local REST API | LOCKED |
| Database | SQLite | Local metadata, findings, events, jobs | LOCKED |
| Frontend | React + TypeScript | Local dashboard | LOCKED |
| Cryptography | SHA-256, Ed25519 | Artifact digests, signatures, provenance | LOCKED |
| Python crypto package | `cryptography` | Ed25519 primitives and key handling | PROPOSED package choice |
| Packaging | Docker | Repeatable offline deployment | LOCKED |
| Testing | pytest | Unit/integration tests | PROPOSED standard |

## Explicit exclusions for initial scope

- TensorFlow and Keras.
- ONNX and ONNX Runtime.
- Mandatory cloud hosting, remote databases, or external inference APIs.
- Any paid service required for core functionality.

## Model artifact policy

Initial support is for PyTorch-native model artifacts (`.pt` / `.pth`). Prefer state-dictionary-based workflows with an explicitly defined model architecture/configuration. Treat checkpoints from untrusted sources as potentially unsafe: PyTorch serialization may involve pickle. Use safe/restricted loading capabilities available in the pinned PyTorch version, validate provenance and file size, and isolate analysis when necessary. Do not claim that hashing makes an unsafe file safe to deserialize.

## Dependency management

- Pin or constrain versions in a lock/requirements file once the implementation environment is selected.
- Separate runtime, development, and optional experiment dependencies.
- Verify package and model-weight availability for offline installation.
- Record Python, PyTorch, CUDA (if used), and dependency versions in experiment reports.
- The first functional baseline should remain CPU-capable. GPU acceleration is optional and must not be required for cryptographic verification or basic metadata processing.

## Hardware assumptions

Development may use a Windows laptop with an NVIDIA RTX 3050 4 GB GPU, but the design must not assume that every experiment fits on this GPU. Use small batches, CPU fallbacks, and controlled datasets. Profile memory/latency rather than assuming capacity.

## Proposed repository environment files

- `requirements.txt` or `pyproject.toml`: package/dependency definition (choose one canonical mechanism during setup).
- `.env.example`: non-secret configuration keys and safe defaults.
- `configs/`: versioned application and experiment configs.
- Dockerfile(s): build from pinned dependencies; no runtime downloads.
