import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ronova.safety.gate import SafetyGate
from ronova.sandbox.runner import IsolatedSandboxRunner
from ronova.detectors.strip import STRIPDetector
from ronova.detectors.image_sentinel import ImageSentinelEngine
from ronova.dataset.duplicates import DuplicateDetector
from ronova.dataset.anomalies import AnomalyDetector
from ronova.operations.drift import InputDriftDetector
from ronova.engine.policy import PolicyEngine
from ronova.provenance.crypto import Ed25519Verifier
from ronova.provenance.engine import ProvenanceEngine
from ronova.provenance.ledger import AuditLedger
from ronova.provenance.certificate import AssuranceCertificateBuilder, compute_canonical_json_hash

app = FastAPI(
    title="RONOVA Assurance API",
    version="0.2.0",
    description="Offline AI Assurance Engine for Computer Vision Models",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BEHAVIORAL_EVAL_CONFIG = {
    "protocol_id": "standard_trojan_eval_suite_v1",
    "dataset_name": "test_samples_triggered.npy",
    "dataset_path": "data/test_samples_triggered.npy",
    "description": "Standardized Trojan behavioral evaluation suite containing probe samples evaluated under STRIP perturbation.",
}

ledger = AuditLedger()
provenance_engine = ProvenanceEngine()
policy_engine = PolicyEngine()
safety_gate = SafetyGate()
sandbox = IsolatedSandboxRunner()
verifier = Ed25519Verifier()
drift_detector = InputDriftDetector()
cert_builder = AssuranceCertificateBuilder()


class LogInferenceRequest(BaseModel):
    model_sha256: str
    input_data_summary: str
    prediction: str


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "framework": "RONOVA v0.2"}


@app.post("/scan/model")
async def scan_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only .onnx model artifacts are supported")

    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, file.filename)
    try:
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Safety Gate
        safety_result = safety_gate.scan_file(model_path)
        manifest_matched = "MANIFEST_VERIFIED_OK" in safety_result.signature_check_status

        # 2. Standardized Behavioral Evaluation Protocol (Independent of Manifest Approval)
        eval_config = BEHAVIORAL_EVAL_CONFIG
        test_data_path = eval_config["dataset_path"]
        clean_overlays_path = "data/clean_overlays.npy"
        dataset_ref_path = "data/clean_dataset.npy"

        if os.path.exists(clean_overlays_path):
            clean_overlays = np.load(clean_overlays_path)
        else:
            clean_overlays = np.random.uniform(0, 1, size=(50, 1, 28, 28)).astype(np.float32)

        if os.path.exists(test_data_path):
            test_samples = np.load(test_data_path)
        elif os.path.exists("data/test_samples_clean.npy"):
            test_samples = np.load("data/test_samples_clean.npy")
        else:
            test_samples = np.random.uniform(0, 1, size=(20, 1, 28, 28)).astype(np.float32)

        if os.path.exists(dataset_ref_path):
            ref_dataset = np.load(dataset_ref_path)
        else:
            ref_dataset = clean_overlays

        # 3. Input Distribution Drift Detector (Level 2 Soft Finding)
        clean_test_path = "data/test_samples_clean.npy"
        if os.path.exists(clean_test_path):
            drift_test_samples = np.load(clean_test_path)
        else:
            drift_test_samples = test_samples

        drift_score, drift_status, drift_finding, drift_details = drift_detector.evaluate_drift(
            ref_dataset, drift_test_samples
        )

        # 4. STRIP Behavioral Trojan Detector inside Sandbox (Only if ONNX is structurally valid)
        if safety_result.is_onnx_valid:
            strip_detector = STRIPDetector(sandbox_runner=sandbox, num_perturbations=20)
            strip_result, sandbox_meta = strip_detector.evaluate_model(
                model_path, test_samples, clean_overlays
            )
            detector_findings = [strip_result.finding, drift_finding]
            sandbox_mode = sandbox_meta.get("sandbox_mode", "subprocess_isolated")
            strip_analysis_data = {
                "behavioral_eval_config": eval_config,
                "mean_entropy": strip_result.mean_entropy,
                "trojan_indicator_triggered": strip_result.trojan_indicator_triggered,
                "finding": strip_result.finding.model_dump(),
            }
        else:
            detector_findings = [drift_finding]
            sandbox_mode = "bypassed_invalid_format"
            strip_analysis_data = {
                "behavioral_eval_config": eval_config,
                "status": "SKIPPED_INVALID_ONNX_FORMAT",
                "mean_entropy": 0.0,
                "trojan_indicator_triggered": False,
            }

        # 5. Audit Ledger Chain Validation
        chain_valid, chain_status, _ = ledger.validate_chain()

        # 6. Policy Engine Evaluation (Phase 4: Level 1 Hard Gates & Level 2 Weighted Risk)
        policy_verdict = policy_engine.evaluate(
            safety_gate_result=safety_result,
            detector_findings=detector_findings,
            sandbox_mode=sandbox_mode,
            audit_chain_valid=chain_valid,
            audit_chain_status=chain_status,
        )

        # 7. 8-Field Provenance Engine
        dummy_output = np.array([0.9, 0.1], dtype=np.float32)
        behavioral_dataset_hash = hashlib.sha256(clean_overlays.tobytes()).hexdigest()
        config_dict = {
            "num_perturbations": 20,
            "blend_alpha": 0.5,
            "entropy_threshold": 0.35,
            "behavioral_eval_protocol": eval_config["protocol_id"],
            "behavioral_dataset_name": eval_config["dataset_name"],
            "behavioral_dataset_hash": behavioral_dataset_hash,
            "drift_ks_pvalue_threshold": 0.05,
            "preprocess_info": "Resize(28,28)->ToTensor()",
        }
        provenance_manifest = provenance_engine.compute_provenance_manifest(
            model_sha256=safety_result.sha256,
            dataset_array=clean_overlays,
            config_dict=config_dict,
            preprocess_info="Resize(28,28)->ToTensor()",
            input_array=test_samples[:1],
            output_array=dummy_output,
        )

        # 8. Audit Ledger & Signed Checkpoint
        event_record = ledger.append_event(
            provenance_manifest=provenance_manifest,
            model_sha256=safety_result.sha256,
            findings_summary=policy_verdict["findings"],
        )

        try:
            signed_checkpoint = ledger.generate_signed_checkpoint(event_record)

            # 9. Phase 7 Signed Canonical Certificate Generator
            canonical_cert = cert_builder.generate_certificate(
                policy_verdict=policy_verdict,
                model_sha256=safety_result.sha256,
                provenance_manifest=provenance_manifest,
                findings=policy_verdict["findings"],
                timestamp=event_record["timestamp"],
            )
        except (FileNotFoundError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=503,
                detail=f"Certificate signing unavailable: {str(e)}",
            )

        return {
            "filename": file.filename,
            "sha256": safety_result.sha256,
            "manifest_verification": {
                "manifest_matched": manifest_matched,
                "status": safety_result.signature_check_status,
            },
            "audit_chain_verification": {
                "audit_chain_valid": chain_valid,
                "status": chain_status,
            },
            "policy": policy_verdict,
            "strip_analysis": strip_analysis_data,
            "input_distribution_drift": {
                "input_distribution_drift_score": drift_score,
                "method": "KS",
                "status": drift_status,
                "ref_means": drift_details["ref_means"],
                "live_means": drift_details["live_means"],
                "finding": drift_finding.model_dump(),
            },
            "provenance_manifest": provenance_manifest,
            "signed_checkpoint": signed_checkpoint,
            "canonical_certificate": canonical_cert,
            "graph_firewall": safety_result.onnx_validation_details.get("graph_firewall", {}),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/scan/dataset")
async def scan_dataset(
    file: UploadFile = File(...),
    role: str = Query("training_eval", description="Dataset role (training_eval, perturbation_overlay, live_input_batch)"),
) -> Dict[str, Any]:
    if not (file.filename.endswith(".npy") or file.filename.endswith(".npz")):
        raise HTTPException(status_code=400, detail="Only .npy/.npz dataset files supported")

    active_role = role if role in ["training_eval", "perturbation_overlay", "live_input_batch"] else "training_eval"

    temp_dir = tempfile.mkdtemp()
    ds_path = os.path.join(temp_dir, file.filename)
    try:
        with open(ds_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            loaded_data = np.load(ds_path)
            if isinstance(loaded_data, np.lib.npyio.NpzFile):
                first_key = loaded_data.files[0]
                dataset = loaded_data[first_key]
            else:
                dataset = loaded_data
        except Exception as e:
            return {
                "filename": file.filename,
                "dataset_role": active_role,
                "analysis_protocol": "Dataset Load Failed",
                "array_shape": [],
                "dtype": "unknown",
                "dimensions": 0,
                "channels": 0,
                "total_samples": 0,
                "dataset_risk_assessment": "UNGUARANTEED",
                "overall_risk_score": 100.0,
                "overall_evidence_strength": "HIGH",
                "duplicate_pairs_count": 0,
                "duplicate_pairs": [],
                "anomalous_samples_count": 0,
                "anomalous_indices": [],
                "anomaly_scores_summary": {},
                "class_channel_distribution": {},
                "chart_coordinates": [],
                "flagged_thumbnails": [],
                "findings": [{
                    "detector_id": "dataset_loader",
                    "finding_type": "UNGUARANTEED_CORRUPT_DATASET",
                    "risk_score": 100.0,
                    "evidence_strength": "HIGH",
                    "title": "Dataset Load Failure",
                    "explanation": f"Unable to parse dataset numpy array: {str(e)}",
                    "limitations": "File corruption or invalid numpy format.",
                    "evidence_details": {"error": str(e)},
                    "is_hard_gate": True,
                }],
                "verdict_explanation": f"Evaluation UNGUARANTEED: Dataset file could not be parsed as a valid numpy array ({str(e)}).",
                "limitations_disclaimer": "Failed dataset load prevents feature extraction or duplicate scanning.",
            }

        # Check shape suitability (must be at least 2D array with samples > 0)
        if not isinstance(dataset, np.ndarray) or dataset.ndim < 2 or len(dataset) == 0:
            return {
                "filename": file.filename,
                "dataset_role": active_role,
                "analysis_protocol": "Unsupported Shape Protocol",
                "array_shape": list(dataset.shape) if hasattr(dataset, "shape") else [],
                "dtype": str(dataset.dtype) if hasattr(dataset, "dtype") else "unknown",
                "dimensions": int(dataset.ndim) if hasattr(dataset, "ndim") else 0,
                "channels": 0,
                "total_samples": len(dataset) if hasattr(dataset, "__len__") else 0,
                "dataset_risk_assessment": "UNGUARANTEED",
                "overall_risk_score": 80.0,
                "overall_evidence_strength": "HIGH",
                "duplicate_pairs_count": 0,
                "duplicate_pairs": [],
                "anomalous_samples_count": 0,
                "anomalous_indices": [],
                "anomaly_scores_summary": {},
                "class_channel_distribution": {},
                "chart_coordinates": [],
                "flagged_thumbnails": [],
                "findings": [{
                    "detector_id": "dataset_shape_validator",
                    "finding_type": "UNGUARANTEED_UNSUPPORTED_SHAPE",
                    "risk_score": 80.0,
                    "evidence_strength": "HIGH",
                    "title": "Unsupported Dataset Array Geometry",
                    "explanation": f"Dataset array shape {dataset.shape if hasattr(dataset, 'shape') else 'invalid'} does not satisfy minimum 2D sample requirements.",
                    "limitations": "Dataset forensics requires multi-sample image/feature tensors.",
                    "evidence_details": {"shape": list(dataset.shape) if hasattr(dataset, "shape") else []},
                    "is_hard_gate": False,
                }],
                "verdict_explanation": "Evaluation UNGUARANTEED: Dataset shape is incompatible with 2D/3D/4D image array forensics.",
                "limitations_disclaimer": "Insufficient sample dimensions for CNN feature extraction.",
            }

        # Protocol naming
        if active_role == "perturbation_overlay":
            analysis_protocol = "STRIP Perturbation Overlay Reference Set Validation"
        elif active_role == "live_input_batch":
            analysis_protocol = "Live Input Batch Distribution Scan"
        else:
            analysis_protocol = "Calibrated CNN Embedding Isolation Forest + Perceptual dHash/MSE"

        # Compute Distribution Summary
        channels = int(dataset.shape[1]) if dataset.ndim == 4 else (1 if dataset.ndim == 3 else 1)
        dist_summary = {
            "mean_pixel": round(float(np.mean(dataset)), 4),
            "std_pixel": round(float(np.std(dataset)), 4),
            "min_pixel": round(float(np.min(dataset)), 4),
            "max_pixel": round(float(np.max(dataset)), 4),
            "shape_str": " × ".join(str(dim) for dim in dataset.shape),
        }

        dup_detector = DuplicateDetector(mse_threshold=0.001)
        dup_pairs, dup_finding = dup_detector.analyze(dataset)

        anom_detector = AnomalyDetector()
        anom_indices, anom_finding, chart_coords, scores_summary, flagged_thumbs = anom_detector.analyze(dataset)

        has_issues = len(dup_pairs) > 0 or len(anom_indices) > 0
        dataset_risk = "REVIEW" if has_issues else "ACCEPT"
        if has_issues:
            if len(dup_pairs) > 0 and len(anom_indices) > 0:
                overall_risk = round(min(98.0, max(dup_finding.risk_score, anom_finding.risk_score) + 0.15 * min(dup_finding.risk_score, anom_finding.risk_score)), 1)
            else:
                overall_risk = round(max(dup_finding.risk_score, anom_finding.risk_score), 1)
        else:
            overall_risk = 0.0

        if not has_issues:
            if active_role == "perturbation_overlay":
                explanation = f"Perturbation overlay dataset '{file.filename}' validated cleanly for STRIP reference use. No duplicates or anomalous features detected."
            elif active_role == "live_input_batch":
                explanation = f"Live input batch '{file.filename}' clean. Feature distribution shows no anomalies or duplicate samples."
            else:
                explanation = f"Training/Evaluation dataset '{file.filename}' passes all forensic checks cleanly under calibrated thresholds. 0 duplicate pairs and 0 anomalous sample clusters detected."
        else:
            explanation = f"Dataset '{file.filename}' (Role: {active_role}) flagged for REVIEW: {len(dup_pairs)} duplicate pair(s) and {len(anom_indices)} sample(s) anomalous in a way consistent with poisoning or labeling errors."

        limitations_disclaimer = (
            "Embedding-space anomaly detection and perceptual hash duplicate checking flag statistical and structural outliers "
            "consistent with poisoning, label noise, or data duplication; they do not constitute deterministic proof of "
            "intentional adversarial dataset poisoning."
        )

        return {
            "filename": file.filename,
            "dataset_role": active_role,
            "analysis_protocol": analysis_protocol,
            "array_shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "dimensions": int(dataset.ndim),
            "channels": channels,
            "total_samples": int(len(dataset)),
            "dataset_risk_assessment": dataset_risk,
            "overall_risk_score": float(overall_risk),
            "overall_evidence_strength": "HIGH" if not has_issues else "MEDIUM",
            "duplicate_pairs_count": len(dup_pairs),
            "duplicate_pairs": dup_pairs,
            "anomalous_samples_count": len(anom_indices),
            "anomalous_indices": anom_indices,
            "anomaly_scores_summary": scores_summary,
            "class_channel_distribution": dist_summary,
            "chart_coordinates": chart_coords,
            "flagged_thumbnails": flagged_thumbs,
            "findings": [dup_finding.model_dump(), anom_finding.model_dump()],
            "verdict_explanation": explanation,
            "limitations_disclaimer": limitations_disclaimer,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/log/inference")
def log_inference(req: LogInferenceRequest) -> Dict[str, Any]:
    chain_valid, chain_status, _ = ledger.validate_chain()
    if not chain_valid:
        return {
            "status": "REJECTED_BROKEN_CHAIN",
            "audit_chain_verification": {
                "audit_chain_valid": False,
                "status": chain_status,
            },
            "error": f"Inference logging rejected due to broken audit ledger chain: {chain_status}",
        }

    dummy_input = np.array([0.5], dtype=np.float32)
    dummy_output = np.array([0.9], dtype=np.float32)
    provenance_manifest = provenance_engine.compute_provenance_manifest(
        model_sha256=req.model_sha256,
        dataset_array=dummy_input,
        config_dict={"input_summary": req.input_data_summary},
        preprocess_info="Identity",
        input_array=dummy_input,
        output_array=dummy_output,
    )
    event_record = ledger.append_event(
        provenance_manifest=provenance_manifest,
        model_sha256=req.model_sha256,
        findings_summary=[{"prediction": req.prediction}],
    )
    checkpoint = ledger.generate_signed_checkpoint(event_record)
    return {
        "status": "LOGGED",
        "event_hash": event_record["event_hash"],
        "audit_chain_verification": {
            "audit_chain_valid": True,
            "status": chain_status,
        },
        "signed_checkpoint": checkpoint,
    }


@app.get("/verify/certificate")
def verify_certificate(
    b64_signature: Optional[str] = Query(None, alias="signature"),
    event_hash: Optional[str] = Query(None, alias="event_hash"),
    model_sha256: Optional[str] = Query(None, alias="model_sha256"),
    timestamp: Optional[float] = Query(None, alias="timestamp"),
    previous_event_hash: Optional[str] = Query("0" * 64, alias="previous_event_hash"),
) -> Dict[str, Any]:
    if not b64_signature:
        return {"verification_status": "FAILED", "reason": "Missing signature parameter"}

    checkpoint_payload = {
        "event_hash": event_hash or "",
        "previous_event_hash": previous_event_hash or ("0" * 64),
        "model_sha256": model_sha256 or "",
        "timestamp": timestamp or 0.0,
    }
    msg_bytes = json.dumps(checkpoint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    is_valid = verifier.verify_signature(msg_bytes, b64_signature)

    return {
        "verification_status": "VALID" if is_valid else "INVALID",
        "signer": "RONOVA-ROOT-ED25519",
        "checkpoint_payload": checkpoint_payload,
    }


@app.post("/check/input")
async def check_input(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.filename.endswith(".npy") or file.filename.endswith(".npz")):
        raise HTTPException(status_code=400, detail="Only .npy/.npz input batch files supported")

    temp_dir = tempfile.mkdtemp()
    live_path = os.path.join(temp_dir, file.filename)
    try:
        with open(live_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        live_dataset = np.load(live_path)
        ref_dataset_path = "data/clean_dataset.npy"

        if os.path.exists(ref_dataset_path):
            ref_dataset = np.load(ref_dataset_path)
        else:
            ref_dataset = live_dataset

        drift_score, status, finding, details = drift_detector.evaluate_drift(ref_dataset, live_dataset)

        return {
            "input_distribution_drift_score": drift_score,
            "method": "KS",
            "status": status,
            "ref_means": details["ref_means"],
            "live_means": details["live_means"],
            "finding": finding.model_dump(),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


RESOURCE_PROFILE_MAX_SIZES = {
    "Standard": 10 * 1024 * 1024,  # 10 MB
    "Demo": 5 * 1024 * 1024,       # 5 MB
}


@app.post("/scan/image")
async def scan_image(
    file: UploadFile = File(...),
    profile: str = Query("Standard", description="Active resource policy profile (Standard vs Demo)"),
) -> Dict[str, Any]:
    active_profile = profile if profile in RESOURCE_PROFILE_MAX_SIZES else "Standard"
    max_image_size = RESOURCE_PROFILE_MAX_SIZES[active_profile]

    content = b""
    while chunk := await file.read(65536):
        content += chunk
        if len(content) > max_image_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({len(content)} bytes) exceeds maximum streamed upload limit ({max_image_size} bytes, Profile: {active_profile})",
            )

    sentinel_engine = ImageSentinelEngine(profile=active_profile)
    report = sentinel_engine.process_image(raw_bytes=content, filename=file.filename)
    res_dict = report.model_dump()

    # 1. Audit Ledger Chain Validation BEFORE ledger append
    chain_valid, chain_reason, _ = ledger.validate_chain()
    if not chain_valid:
        broken_finding = {
            "detector_id": "audit_ledger_validator",
            "finding_type": "BROKEN_AUDIT_CHAIN",
            "risk_score": 100.0,
            "evidence_strength": "HIGH",
            "title": "Audit Ledger Integrity Violation",
            "explanation": f"Audit ledger hash chain is broken: {chain_reason}. Event creation blocked.",
            "limitations": "Hard gate violation. Ledger state is corrupted.",
            "evidence_details": {"error": chain_reason},
            "is_hard_gate": True,
        }
        res_dict["status"] = "REJECTED"
        res_dict["risk_score"] = 100.0
        res_dict["evidence_strength"] = "HIGH"
        res_dict["findings"].insert(0, broken_finding)
        return res_dict

    # 2. Log to audit ledger & generate signed certificate if image format and model analysis were valid
    if report.safety_result.is_valid:
        real_model_sha256 = report.safety_result.analysis_model_sha256
        input_arr = report.safety_result.get_baseline_input_array()
        output_arr = report.safety_result.get_baseline_output_array()

        if real_model_sha256 and input_arr is not None and output_arr is not None:
            config_dict = {
                "image_sentinel_protocol": report.safety_result.protocol_version,
                "active_resource_profile": report.safety_result.active_resource_profile,
                "original_sha256": report.safety_result.original_sha256,
                "canonical_sha256": report.safety_result.canonical_sha256,
                "perceptual_hash": report.safety_result.perceptual_hash,
                "original_format": report.safety_result.original_format,
                "adapter_name": report.safety_result.adapter_name,
                "calibration_sha256": report.safety_result.calibration_sha256,
                "transformation_seed": report.safety_result.transformation_seed,
            }

            prov_manifest = provenance_engine.compute_provenance_manifest(
                model_sha256=real_model_sha256,
                dataset_array=input_arr,
                config_dict=config_dict,
                preprocess_info=f"ImageSentinelDecode->StripExif->Resize(28,28)->Adapter({report.safety_result.adapter_name})",
                input_array=input_arr,
                output_array=output_arr,
            )

            event_record = ledger.append_event(
                provenance_manifest=prov_manifest,
                model_sha256=real_model_sha256,
                findings_summary=res_dict["findings"],
            )

            try:
                signed_checkpoint = ledger.generate_signed_checkpoint(event_record)
                policy_verdict = {
                    "verdict": report.status.value,
                    "risk_score": report.risk_score,
                    "evidence_strength": report.evidence_strength.value,
                    "hard_gate_fired": report.status == "REJECTED",
                    "hard_gates_triggered": [f["title"] for f in res_dict["findings"] if f.get("is_hard_gate")],
                }
                canonical_cert = cert_builder.generate_certificate(
                    policy_verdict=policy_verdict,
                    model_sha256=real_model_sha256,
                    provenance_manifest=prov_manifest,
                    findings=res_dict["findings"],
                    timestamp=event_record["timestamp"],
                )
                res_dict["provenance_manifest"] = prov_manifest
                res_dict["signed_checkpoint"] = signed_checkpoint
                res_dict["canonical_certificate"] = canonical_cert
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Signing key unavailable for certificate generation: {str(e)}",
                )

    return res_dict


# Retain /analyze endpoint for backwards compatibility
@app.post("/analyze")
async def analyze_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await scan_model(file)
