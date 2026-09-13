# 🛡️ RONOVA — Autonomous AI Assurance & Cryptographic Provenance Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![React](https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **RONOVA** is an end-to-end, air-gapped AI assurance, security vetting, and cryptographic provenance platform designed for safety-critical machine learning deployments (Defense, Healthcare, Finance, and Enterprise Infrastructure).

---

## 🌟 Why RONOVA?

Modern machine learning supply chains are vulnerable to **data poisoning**, **Trojan backdoors**, **adversarial perturbations**, **data leakage**, and **covert weight tampering**. 

RONOVA provides automated, deterministic pre-deployment validation, runtime anomaly detection, and tamper-evident audit trails with zero reliance on cloud verification.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                RONOVA ASSURANCE PIPELINE               │
                  └───────────────────────────┬────────────────────────────┘
                                              │
      ┌─────────────────────────┬─────────────┴────────────┬────────────────────────┐
      ▼                         ▼                          ▼                        ▼
┌──────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Dataset    │       │     STRIP       │       │ Image Sentinel  │       │  Distribution   │
│  Forensics   │       │ Trojan Detector │       │  Robustness     │       │ Drift Analysis  │
└──────┬───────┘       └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
       │                        │                         │                         │
       └────────────────────────┼─────────────────────────┴─────────────────────────┘
                                ▼
               ┌─────────────────────────────────┐
               │    Deterministic Safety Gate    │  ◄── Multi-Profile Governance
               │   (Defense / Health / Corp)     │
               └────────────────┬────────────────┘
                                ▼
               ┌─────────────────────────────────┐
               │ Tamper-Evident Ledger (Ed25519) │  ──► Signed QR Certificates
               └─────────────────────────────────┘
```

---

## 🚀 Key Modules & Capabilities

### 1. 🔍 Dataset Forensics & Outlier Inspection
* **Embedding-Space Anomaly Detection**: Employs an offline CNN feature extractor and calibrated Isolation Forest clustering to detect poisoned samples and mislabeled data.
* **Dimensionality Reduction (PCA 2D)**: Visualizes high-dimensional dataset clustering in an interactive coordinate scatter space.
* **Visual Outlier Thumbnails**: Generates lightweight base64 thumbnail previews for flagged anomalous samples.
* **Exact & Near-Duplicate Analysis**: Identifies data leakage and training contamination using perceptual hashing and vector similarity thresholds.

### 2. 🧬 STRIP Trojan & Backdoor Detection
* Implements **STRIP** (*Strong Perturbation Trojan Detection*): Superimposes clean image overlays onto candidate test inputs and measures the Shannon entropy collapse across model output probability vectors.
* **Entropy Drop Isolation**: Backdoored inputs maintain low prediction entropy regardless of noise overlays, cleanly exposing Trojan triggers without needing training data access.

### 3. 🎯 Image Sentinel & Adversarial Defense
* Detects high-frequency adversarial gradient attacks (FGSM, PGD) and distribution corruption.
* Measures prediction stability across varying noise scales ($\sigma \in [0.05, 0.20]$).
* Provides visual pixel-level perturbation heatmaps and channel-wise variance maps.

### 4. 📈 Statistical Distribution Drift Monitor
* Detects inference data shift using **Wasserstein Distance** and **Kolmogorov-Smirnov (KS) tests**.
* Computes feature-level drift scores to alert teams before downstream model accuracy degrades.

### 5. ⚖️ Multi-Profile Safety Governance Gate
* Configurable risk profiles:
  * 🛡️ **Defense / Tactical**: Zero-tolerance strict gating (Rejection on any hard-gate indicator).
  * 🏥 **Healthcare / Life Sciences**: Stringent anomaly boundaries and drift limits.
  * 🏢 **Enterprise / Standard**: Balanced trade-off between throughput and assurance.
* Explains all findings with deterministic rationale, limitation disclosures, and explicit evidence strength ratings.

### 6. 🔏 Cryptographic Audit Ledger & QR Verification
* Every scan event is committed to a hash-chained audit log signed via **Ed25519** asymmetric cryptography.
* Produces portable, self-contained **Security Verification Certificates** with cryptographically signed payload QR codes.
* Includes a standalone verification tool (`verify_certificate.py`) capable of proving certificate authenticity offline without network connectivity.

---

## 🛠️ System Architecture & Stack

| Layer | Technologies Used |
|---|---|
| **Core AI & Math** | PyTorch, ONNX Runtime, NumPy, Scikit-learn, SciPy |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Cryptography** | `cryptography` (Ed25519, SHA-256), `qrcode`, `Pillow` |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide Icons, Recharts |
| **Sandboxing** | Python Subprocess Workers, Resource-Capped Memory/CPU Execution |

---

## ⚡ Quick Start Guide

### Prerequisites
* Python 3.10+
* Node.js 18+ and npm

### 1. Clone & Set Up Backend
```bash
# Clone the repository
git clone https://github.com/developerHarish2007/Ronova.git
cd Ronova

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt   # or pip install fastapi uvicorn torch onnxruntime numpy scikit-learn pillow cryptography qrcode
```

### 2. Initialize Keys & Demo Artifacts
```bash
# Generate Ed25519 provenance root keys
python scripts/init_provenance_keys.py

# Create synthetic evaluation models and sample datasets
python scripts/train_models.py
python scripts/create_demo_datasets.py
```

### 3. Launch the Backend API
```bash
python -m uvicorn ronova.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive Swagger Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### 4. Launch the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
* Web Dashboard: **[http://localhost:5173/](http://localhost:5173/)**

---

## 🧪 Testing & Verification

Run the automated test harnesses to validate individual modules:

```bash
# Test Dataset Forensics & Anomaly Isolation
python scripts/test_dataset_forensics.py

# Test Cryptographic Provenance & Ledger Chains
python scripts/test_provenance.py
python scripts/test_audit_ledger.py

# Test STRIP Backdoor Detection
python scripts/test_image_sentinel.py

# Run Complete End-to-End Regression Suite
python scripts/run_phase8_regression.py
```

---

## 📁 Repository Structure

```
Ronova/
├── ronova/
│   ├── api/             # FastAPI REST endpoints & request handlers
│   ├── core/            # Pydantic schemas, types, and finding data structures
│   ├── dataset/         # Dataset anomaly clustering & duplicate detection
│   ├── detectors/       # STRIP Trojan detection & Image Sentinel algorithms
│   ├── engine/          # Assurance policy evaluation rules
│   ├── operations/      # Statistical drift computation (KS-test / Wasserstein)
│   ├── provenance/      # Ed25519 signature engine, ledger chain & certificates
│   ├── safety/          # Verdict gating (Defense, Healthcare, Enterprise)
│   └── sandbox/         # Sandboxed ONNX runtime execution runners
├── frontend/
│   ├── src/
│   │   ├── components/  # React cards (DatasetForensics, ImageSentinel, Drift, Provenance)
│   │   ├── App.jsx      # Main interactive assurance dashboard
│   │   └── index.css    # Tailwind styling and dark mode glassmorphism
│   └── package.json
├── data/                # Synthetic datasets for test replication
├── models/              # Pre-compiled ONNX models (clean vs backdoored)
├── scripts/             # End-to-end execution and regression scripts
├── verify_certificate.py # Offline CLI tool for authenticating signed QR certificates
└── README.md
```

---

## 🔒 Security & Air-Gap Compliance

* **100% Offline Capability**: Runs completely local without third-party cloud API dependencies.
* **Deterministic Cryptography**: All security claims are backed by non-malleable Ed25519 digital signatures.
* **Memory-Safe Execution**: ONNX model evaluations run within sandboxed runner environments with bounded memory allocation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
