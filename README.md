# VisionGuard

**AI Lifecycle Integrity & Hardware-Bound Security Assurance for Computer-Vision Systems**  
*Official Submission for Smart India Hackathon (SIH) 2026*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6+-3178c6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4+-38bdf8.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-625%20Passed-brightgreen.svg)]()
[![Air--Gap](https://img.shields.io/badge/Air--Gap-Enforced-success.svg)]()

---

## 1. Overview

**VisionGuard** is an offline-first AI lifecycle integrity and security assurance platform engineered for mission-critical computer-vision deployments (tactical telemetry, edge drones, border surveillance, and industrial robotics).

Deployed in high-assurance environments without external cloud dependencies, VisionGuard provides cryptographic provenance, automated quarantine fencing, dual backdoor detection, and distribution drift monitoring across four key pillars:

1. **Data Integrity:** Streaming chunk-buffered SHA-256 CAS, inode exhaustion fencing, and isolated archive extraction sandboxes.
2. **Model Integrity:** Safe checkpoint deserialization, 11 forensic detectors, dual backdoor identification (Spectral Signatures & Activation Clustering), and automated Conflict Engine synthesis.
3. **Inference Integrity:** RFC 8785 canonical serialization, Ed25519 digital signatures, monotonic sequence enforcement, and continuous hash-chaining ledgers.
4. **Distribution Shift:** Two-sample Kolmogorov-Smirnov (KS) and Wasserstein distance metrics with ADR-012 epistemic honesty ($N \ge 30$ sample density gating).

---

## 2. Architecture & Technology Stack

```text
VisionGuard/
├── backend/
│   ├── app/
│   │   ├── api/v1/routes/            # REST API (Inference, Models, Datasets, Drift, Health)
│   │   ├── core/                     # Logging, Air-gap settings, and configurations
│   │   ├── db/                       # SQLAlchemy models, SQLite repositories, pragmas
│   │   ├── domain/                   # Entities, Enums (CapabilityStatus), Finding models
│   │   ├── modules/
│   │   │   ├── airgap_compliance/    # Attestation engine, hardware binding, secrets
│   │   │   ├── data_integrity/       # Archive validators, CAS, quarantine fencing
│   │   │   ├── model_integrity/      # 11 detectors, backdoor analysis, conflict engine
│   │   │   ├── inference_integrity/  # RFC 8785 canonicalizer, Ed25519 signer, audit engine
│   │   │   └── distribution_shift/   # KS and Wasserstein statistical drift detectors
│   │   └── main.py                   # FastAPI ASGI entrypoint (Loopback 127.0.0.1)
│   └── tests/
│       ├── chaos/                    # SIH 2026 Adversarial Chaos Validation Engine
│       ├── integration/              # End-to-end multi-module pipelines
│       ├── security/                 # Isolation sandbox & boundary enforcement
│       └── unit/                     # Focused component tests
├── frontend/                         # High-Agency Operator Dashboard
│   ├── src/
│   │   ├── api/                      # Client (client.ts) and custom React hooks
│   │   ├── components/               # HashChainViewer, DriftMonitor, QuarantineInspector, AttestationPanel
│   │   ├── types/                    # Cryptographic & integrity TypeScript definitions
│   │   └── App.tsx                   # Master layout with ErrorBoundary protection
│   ├── index.html
│   ├── vite.config.ts                # Loopback proxy (127.0.0.1:8000)
│   └── package.json
├── scripts/
│   └── run_chaos_benchmark.py        # CLI Chaos validation & audit report generator
├── SIH_2026_SUBMISSION_DOSSIER.md    # Master submission dossier and evaluation proof
├── Dockerfile.airgap                 # Hardened offline container definition
└── requirements.txt                  # Pinned dependencies
```

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.11 or 3.12 (virtual environment recommended)
- Node.js 18+ & npm
- Local PyTorch installation (CPU fallback supported; compatible with RTX 3050 GPUs)

### 1. Environment Setup

```bash
# Clone and enter directory
cd VisionGuard_Starter_Pack

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run the Backend API

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available locally at `http://127.0.0.1:8000/docs`.

### 3. Run the Operator Frontend

```bash
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173` in your browser to access the High-Agency Operator Dashboard.

---

## 4. Verification & Testing

### Run Comprehensive Backend Test Suite (625 Tests)
```bash
python -m pytest backend/tests -v
```

### Run SIH 2026 Adversarial Chaos Benchmark
```bash
python scripts/run_chaos_benchmark.py
```

### Run Code Quality & Linter Checks
```bash
ruff check backend scripts
```

### Build Production Frontend
```bash
cd frontend
npm run build
```

---

## 5. Security & Epistemic Honesty Principles

- **No Remote Dependencies:** Operates strictly under air-gap constraints (`HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- **Cryptographic Trust:** Edge inference receipts are signed with Ed25519 and verified monotonically before SQLite insertion.
- **ADR-012 Epistemic Honesty:** When data is insufficient (sample size $N < 30$), VisionGuard yields `CapabilityStatus.INCONCLUSIVE`. It strictly refuses to hallucinate statistical significance or declare unverified models clean.
- **Fail-Closed Containment:** Untrusted archives, modified weights, or tampered signatures trigger immediate quarantine isolation.

---

## 6. Project Documentation & Submission Dossier

For complete mathematical formulations, threat modeling, architecture decision records, and benchmark data:
- [SIH 2026 Submission Dossier](SIH_2026_SUBMISSION_DOSSIER.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Threat Model](docs/THREAT_MODEL.md)
- [Architecture Decisions (ADRs)](docs/decisions/DECISIONS.md)

---
*Developed for Smart India Hackathon (SIH) 2026.*
