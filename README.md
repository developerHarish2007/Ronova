# RONOVA (Robust Offline Network for Observation, Verification & Assurance)

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI: 0.100+](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX: Supported](https://img.shields.io/badge/ONNX-Static%20%26%20Runtime-orange.svg)](https://onnx.ai/)
[![Security: Offline-First](https://img.shields.io/badge/Security-100%25%20Offline%20%2F%20Air--Gapped-green.svg)](#threat-model--design-principles)
[![Cryptography: Ed25519](https://img.shields.io/badge/Trust%20Root-Ed25519%20Signed-purple.svg)](#cryptographic-assurance--offline-verifier)

> **RONOVA** is an offline, air-gapped AI assurance and governance framework designed to statically inspect, sandboxed-evaluate, cryptographically verify, and certify Computer Vision models and training datasets without relying on cloud APIs, online dependencies, or external telemetry.

---

## Key Capabilities & Architectural Pillars

RONOVA implements a multi-stage defense-in-depth pipeline that ensures complete traceability, deterministic reproducibility, and fail-closed security for machine learning deployments.

```
                                  [ Incoming Model / Dataset / Image ]
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. OFFLINE STATIC SAFETY GATE & ONNX GRAPH FIREWALL                                              │
│    • SHA-256 Digest Computation                                                                 │
│    • Protobuf & Graph Schema Validation                                                          │
│    • Static Operator Allowlisting (STANDARD_SAFE_OPERATORS) & Domain Verification               │
│    • Input Dimension & Memory Budget Enforcement                                                 │
│    • Out-of-Band Approved Manifest Check (manifests/approved_models.json)                        │
│    • ModelScan Static Serialization & Pickle AST Analysis                                        │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │ (Pass: Continue | Fail: Instant Hard-Gate QUARANTINE)
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. ISOLATED SUBPROCESS SANDBOX & BEHAVIORAL ANALYSIS (STRIP)                                     │
│    • Subprocess Execution Barrier (Memory Limits, Isolated CPU Execution Provider)              │
│    • Standardized Trojan Evaluation Protocol (STRIP Perturbation & Shannon Entropy)              │
│    • Kolmogorov-Smirnov (KS) Input Distribution Drift Analysis                                   │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. MULTI-LAYER GOVERNANCE & POLICY ENGINE                                                        │
│    • Level 1 Hard Gates: Manifest Mismatch, Graph Firewall Violation, Malicious Serialization,  │
│      Broken Audit Chain, Trojan Indicator Breach -> Instant QUARANTINE (Never Averaged Away)     │
│    • Level 2 Weighted Risk Evaluation: Aggregated Risk Score (0-100) -> ACCEPT / REVIEW          │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. CRYPTOGRAPHIC PROVENANCE & IMMUTABLE AUDIT LEDGER                                             │
│    • 8-Field Deterministic Provenance Manifest Hashing                                           │
│    • SHA-256 Sequential Hash-Linked Audit Event Ledger                                           │
│    • Signed Checkpoints & Canonical JSON Encoding                                                │
│    • Ed25519 Asymmetric Signed Certificates with Embedded QR Codes                              │
│    • Standalone Air-Gapped CLI Verifier (`verify_certificate.py`)                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. Offline Static ONNX Graph Firewall
- **Static Inspection Only**: Never executes model code or instantiates execution providers during structural inspection.
- **Strict Operator Allowlist**: Enforces standard safe ONNX operators (`STANDARD_SAFE_OPERATORS`), blocking custom, unapproved, or experimental operator registrations.
- **Domain Enforcement**: Disallows unapproved operator domains (`APPROVED_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}`).
- **Tensor Dimension Budgeting**: Validates input tensor dimensions against configurable bounds (`max_input_elements`, `max_dimension_size`, `max_nodes`) to protect against memory exhaustion and allocation bombs.
- **Fail-Closed Hard Gate**: Violations emit `ONNX_GRAPH_FIREWALL` hard gate findings, instantly triggering `QUARANTINE` and bypassing downstream behavioral execution.

### 2. Model Safety Gate & Manifest Verification
- **Cryptographic Digest**: Computes deterministic SHA-256 hash of uploaded artifacts.
- **Out-of-Band Manifest Validation**: Compares model digest against approved institutional manifests (`manifests/approved_models.json`).
- **ModelScan Static Serialization Inspection**: Detects unsafe Python serialization constructs (e.g., `pickle`, arbitrary code execution vectors) prior to runtime loading.

### 3. Isolated Sandbox & STRIP Behavioral Analysis
- **Subprocess Isolation**: Executes models in dedicated subprocesses with strict resource controls and sequential CPU execution provider configurations.
- **STRIP (STRong Intentional Perturbation)**: Evaluates Trojan and backdoor susceptibility by blending live inputs with clean reference overlays (`clean_overlays.npy`) and measuring output Shannon entropy distributions.

### 4. Forensic Dataset Analysis
- **Perceptual & Cryptographic Duplicate Scanning**: Identifies exact hash collisions (MD5) and near-duplicate images using difference hashing (`dHash`) and calibrated Mean Squared Error (MSE).
- **Embedding-Space Anomaly Detection**: Extracts CNN feature embeddings and computes statistical anomaly scores via Isolation Forests to identify poisoned samples or corrupt data.
- **Role-Calibrated Protocols**: Custom forensic thresholds for `training_eval`, `perturbation_overlay`, and `live_input_batch` dataset roles.

### 5. Image Sentinel Live Input Defense
- **EXIF & Metadata Sanitization**: Strips hidden tags, location coordinates, and malformed headers.
- **Structural Decoding Validation**: Enforces maximum dimension, channel count, and file size limits across configurable resource profiles (`Standard` vs `Demo`).
- **Perceptual Image Integrity**: Computes SHA-256 canonical digests and dHash fingerprints before feeding inputs to model pipelines.

### 6. Cryptographic Provenance & Offline Audit Ledger
- **8-Field Canonical Provenance Manifest**: Deterministically captures:
  1. `model_sha256`
  2. `dataset_sha256`
  3. `config_sha256`
  4. `environment_sha256`
  5. `pipeline_stage`
  6. `input_sha256`
  7. `output_sha256`
  8. `timestamp`
- **Immutable Hash-Linked Ledger**: Every evaluation is appended to a cryptographic ledger where each event is chained to the SHA-256 digest of the previous record.
- **Ed25519 Canonical Certificates**: Generates cryptographically signed JSON certificates and scannable QR verification payloads.

---

## Directory Structure

```
Ronova/
├── ronova/                     # Core Assurance Framework Package
│   ├── api/                    # FastAPI HTTP service & endpoints
│   ├── core/                   # Shared Pydantic data models & typing contracts
│   ├── dataset/                # Forensics (duplicate & anomaly detectors)
│   ├── detectors/              # STRIP behavioral detector & Image Sentinel
│   ├── engine/                 # Multi-layer governance & policy engine
│   ├── operations/             # Statistical drift analysis (Kolmogorov-Smirnov)
│   ├── provenance/             # Ed25519 crypto, ledger, and certificate builder
│   ├── safety/                 # Safety gate & static ONNX graph firewall
│   └── sandbox/                # Subprocess runner & isolation sandbox
├── frontend/                   # React + Vite + TailwindCSS Assurance Console
│   ├── src/
│   │   ├── components/         # Verdict, Perturbation, Provenance, Sentinel UI
│   │   └── App.jsx             # Multi-theme assurance dashboard
├── manifests/                  # Approved model manifests & hashes
├── models/                     # Sample clean & backdoored ONNX classifiers
├── data/                       # Reference evaluation & calibration datasets
├── scripts/                    # Pytest suites & end-to-end regression scripts
├── trust/                      # Public keys & trusted root anchors
├── secrets/                    # Local signing keys (air-gapped)
├── verify_certificate.py       # Standalone air-gapped certificate verifier CLI
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## Quickstart Guide

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10 – 3.14)
- **Node.js 18+** & `npm` (for the Frontend Console)

### 1. Installation
Clone the repository and install the backend dependencies:
```bash
git clone https://github.com/developerHarish2007/Ronova.git
cd Ronova
pip install -r requirements.txt
```

### 2. Initialize Keys & Demo Datasets (Optional / Setup)
If running for the first time or creating fresh keys:
```bash
python scripts/init_provenance_keys.py
python scripts/create_demo_datasets.py
python scripts/create_manifest.py
```

### 3. Start the Backend API Server
```bash
python -m uvicorn ronova.api.main:app --host 127.0.0.1 --port 8000
```
The REST API will be available at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.

### 4. Start the Frontend Dashboard
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to interact with the RONOVA Console.

---

## API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/scan/model` | Full model assurance pipeline: static Graph Firewall, SafetyGate, STRIP sandbox, drift, policy verdict, provenance ledger, and Ed25519 signed certificate. |
| `POST` | `/scan/dataset` | Forensic analysis of `.npy`/`.npz` datasets: perceptual duplicate detection and embedding-space anomaly scoring. |
| `POST` | `/scan/image` | Image Sentinel intake: EXIF sanitization, structural verification, and signed intake checkpointing. |
| `GET` | `/verify/certificate` | Verifies Ed25519 signatures and audit ledger payload integrity. |
| `POST` | `/log/inference` | Logs live inference records to the sequential cryptographic audit chain. |
| `POST` | `/check/input` | Evaluates input distribution drift using two-sample Kolmogorov-Smirnov tests. |
| `GET` | `/health` | Healthcheck and framework metadata. |

---

## Standalone Air-Gapped Certificate Verification

RONOVA certificates can be verified completely offline using the standalone CLI script `verify_certificate.py` without requiring network access or the API server to be running:

```bash
python verify_certificate.py --cert path/to/certificate.json --public-key trust/public_key.pem
```

---

## Running the Test Suite

RONOVA includes an exhaustive suite of unit and regression tests covering all security boundaries, firewalls, and cryptographic invariants:

```bash
# Run ONNX Graph Firewall tests
pytest scripts/test_graph_firewall.py

# Run all pytest suites
pytest scripts/

# Run complete phase 8 regression & reconciliation suite
python scripts/run_phase8_regression.py
```

---

## Threat Model & Design Principles

1. **Strictly Offline & Air-Gapped**: Zero reliance on external cloud APIs, LLM inference endpoints, or telemetry.
2. **Fail-Closed Governance**: If any Level 1 Hard Gate triggers (Graph Firewall violation, unapproved manifest hash, broken audit chain, unsafe serialization, or Trojan entropy collapse), the model is instantly assigned a `QUARANTINE` verdict that cannot be averaged away.
3. **Static-Before-Dynamic**: Malicious model structures and unapproved operators are halted statically before any runtime session is initialized.
4. **Cryptographic Proofs**: All assurance conclusions produce canonical JSON records signed via Ed25519 root keys.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
