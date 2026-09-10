"""
RONOVA Image Sentinel / Image Input Assurance Layer Engine:
Executes isolated subprocess image decoding, format validation, defensive resource checks,
metadata stripping, canonical PNG re-encoding, deterministic transformation stability analysis,
and honest DL anomaly evaluation.
"""

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ronova.core.types import (
    ImageAssuranceStatus,
    ImageSafetyResult,
    ImageSentinelReport,
    Finding,
    EvidenceStrength,
)
from ronova.sandbox.runner import IsolatedSandboxRunner


def _inspect_and_select_adapter(model_file: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Inspects model input name, dtype, and shape dimensions.
    Returns (adapter_name, input_metadata, error_message).
    """
    try:
        session = ort.InferenceSession(str(model_file), providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()[0]
        input_name = inputs.name
        input_type = inputs.type
        input_shape = inputs.shape

        norm_shape = []
        for d in input_shape:
            if isinstance(d, int) and d > 0:
                norm_shape.append(d)
            else:
                norm_shape.append(-1)

        meta = {
            "name": input_name,
            "type": input_type,
            "shape": list(input_shape),
            "normalized_shape": norm_shape,
        }

        # Adapter: mnist_float32_nchw_28x28 (Demo adapter for 28x28 1-channel float32 models)
        if (norm_shape in [[-1, 1, 28, 28], [1, 1, 28, 28]]) and "float" in input_type.lower():
            return "mnist_float32_nchw_28x28", meta, None

        err = f"Model input '{input_name}: {input_type} shape={input_shape}' has no registered explicit preprocessing adapter."
        return None, meta, err
    except Exception as e:
        return None, None, f"Failed to inspect ONNX model input layout: {str(e)}"


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates Jensen-Shannon divergence between two probability distributions."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))
    js_div = 0.5 * (kl_p_m + kl_q_m)
    return float(max(0.0, js_div))


def _calculate_entropy(probs: np.ndarray) -> float:
    """Calculates Shannon entropy in nats."""
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


RESOURCE_PROFILES = {
    "Standard": {
        "max_dim": 4096,
        "max_pixels": 16000000,
        "max_file_size": 10485760,  # 10 MB
    },
    "Demo": {
        "max_dim": 2048,
        "max_pixels": 4000000,
        "max_file_size": 5242880,  # 5 MB
    },
}


class ImageSentinelEngine:
    """
    RONOVA Image Sentinel Engine:
    Manages fail-closed image upload, isolated subprocess decoding, transformation stability,
    and honest DL anomaly assessment across resource policy profiles (Standard vs Demo).
    """

    MANDATORY_DISCLAIMER = (
        "Passed configured integrity, decoding, transformation-stability, and reference-distribution checks. "
        "This does not guarantee the image is authentic, harmless, non-adversarial, or safe under all possible attack conditions."
    )

    def __init__(
        self,
        profile: str = "Standard",
        calibration_path: str = "artifacts/image_sentinel_calibration.json",
        default_model_path: str = "models/clean_classifier.onnx",
        model_path: Optional[str] = None,
        max_file_size: Optional[int] = None,
        max_dim: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        self.active_profile_name = profile if profile in RESOURCE_PROFILES else "Standard"
        profile_limits = RESOURCE_PROFILES[self.active_profile_name]

        self.max_file_size = max_file_size or profile_limits["max_file_size"]
        self.max_dim = max_dim or profile_limits["max_dim"]
        self.max_pixels = max_pixels or profile_limits["max_pixels"]
        self.calibration_path = Path(calibration_path)
        self.default_model_path = Path(model_path or default_model_path)

    def process_image(
        self,
        raw_bytes: bytes,
        filename: str,
        model_path: Optional[str] = None,
    ) -> ImageSentinelReport:
        model_file = Path(model_path).resolve() if model_path else self.default_model_path.resolve()

        # 1. Enforce streamed file size limit
        if len(raw_bytes) > self.max_file_size:
            safety_res = ImageSafetyResult(
                is_valid=False,
                status=ImageAssuranceStatus.REJECTED,
                original_sha256="",
                canonical_sha256="",
                perceptual_hash="",
                original_format="UNKNOWN",
                canonical_format="PNG",
                dimensions=[0, 0],
                color_mode="UNKNOWN",
                metadata_stripped=False,
                rejection_reasons=[f"Uploaded file size ({len(raw_bytes)} bytes) exceeds maximum limit ({self.max_file_size} bytes, Profile: {self.active_profile_name})"],
                active_resource_profile=self.active_profile_name,
                resource_profile_limits={
                    "max_dim": self.max_dim,
                    "max_pixels": self.max_pixels,
                    "max_file_size": self.max_file_size,
                },
            )
            finding = Finding(
                detector_id="image_sentinel_resource_limits",
                finding_type="REJECTED_RESOURCE_LIMIT",
                risk_score=0.0,
                evidence_strength=EvidenceStrength.HIGH,
                title="Image Rejection: Resource Processing Budget Exceeded",
                explanation=f"Image file size ({len(raw_bytes)} bytes) exceeds active processing budget ({self.max_file_size} bytes, Profile: {self.active_profile_name}).",
                limitations="Resource boundary refusal is a processing safety control, not proof of malicious intent.",
                is_hard_gate=True,
            )
            return ImageSentinelReport(
                filename=filename,
                status=ImageAssuranceStatus.REJECTED,
                risk_score=0.0,
                evidence_strength=EvidenceStrength.HIGH,
                safety_result=safety_res,
                findings=[finding],
                limitations_disclaimer=self.MANDATORY_DISCLAIMER,
            )

        # 2. Save raw bytes to temporary input file for isolated worker decoding
        temp_dir = Path(tempfile.mkdtemp())
        input_path = temp_dir / filename
        canonical_output_path = temp_dir / "canonical.png"
        worker_result_json = temp_dir / "worker_result.json"

        try:
            with open(input_path, "wb") as f:
                f.write(raw_bytes)

            # 3. Subprocess call to isolated image_worker.py
            worker_script = Path(__file__).resolve().parent.parent / "sandbox" / "image_worker.py"
            cmd = [
                sys.executable,
                str(worker_script),
                "--input", str(input_path),
                "--output", str(canonical_output_path),
                "--max-size", str(self.max_file_size),
                "--max-dim", str(self.max_dim),
                "--max-pixels", str(self.max_pixels),
                "--max-frames", "1",
                "--profile", self.active_profile_name,
                "--result-json", str(worker_result_json),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if not worker_result_json.exists():
                safety_res = ImageSafetyResult(
                    is_valid=False,
                    status=ImageAssuranceStatus.REJECTED,
                    original_sha256="",
                    canonical_sha256="",
                    perceptual_hash="",
                    original_format="UNKNOWN",
                    canonical_format="PNG",
                    dimensions=[0, 0],
                    color_mode="UNKNOWN",
                    metadata_stripped=False,
                    rejection_reasons=[f"Image decode worker failed to complete ({proc.stderr.strip()})"],
                    active_resource_profile=self.active_profile_name,
                )
                return ImageSentinelReport(
                    filename=filename,
                    status=ImageAssuranceStatus.REJECTED,
                    risk_score=100.0,
                    evidence_strength=EvidenceStrength.HIGH,
                    safety_result=safety_res,
                    findings=[
                        Finding(
                            detector_id="image_sentinel_decoder",
                            finding_type="REJECTED_WORKER_FAILURE",
                            risk_score=100.0,
                            evidence_strength=EvidenceStrength.HIGH,
                            title="Image Rejection: Isolated Decode Worker Crash",
                            explanation="Image decode worker subprocess crashed or returned non-zero code.",
                            limitations="Isolated worker prevents untrusted image decoder exploits from affecting host API.",
                            is_hard_gate=True,
                        )
                    ],
                    limitations_disclaimer=self.MANDATORY_DISCLAIMER,
                )

            with open(worker_result_json, "r", encoding="utf-8") as f:
                worker_data = json.load(f)

            if not worker_data.get("is_valid", False):
                reasons = worker_data.get("rejection_reasons", ["Unspecified decode rejection"])
                rejection_type = worker_data.get("rejection_type", "REJECTED_MALFORMED_IMAGE")

                is_resource_limit = (
                    rejection_type == "REJECTED_RESOURCE_LIMIT"
                    or any("exceed" in r.lower() or "decompression bomb" in r.lower() for r in reasons)
                )

                finding_type = "REJECTED_RESOURCE_LIMIT" if is_resource_limit else "REJECTED_MALFORMED_IMAGE"
                risk_score = 0.0 if is_resource_limit else 100.0
                title = "Image Rejection: Resource Processing Budget Exceeded" if is_resource_limit else "Image Rejection: Security / Resource Policy Breach"
                explanation = (
                    f"Image exceeded active processing budget ({self.active_profile_name}): {'; '.join(reasons)}"
                    if is_resource_limit
                    else f"Image failed defensive checks: {'; '.join(reasons)}"
                )
                limitations = (
                    "Resource boundary refusal is a processing safety control, not proof of malicious intent."
                    if is_resource_limit
                    else "Fail-closed decoding rejects non-conforming or multi-frame image binaries."
                )

                safety_res = ImageSafetyResult(
                    is_valid=False,
                    status=ImageAssuranceStatus.REJECTED,
                    original_sha256="",
                    canonical_sha256="",
                    perceptual_hash="",
                    original_format="UNKNOWN",
                    canonical_format="PNG",
                    dimensions=[0, 0],
                    color_mode="UNKNOWN",
                    metadata_stripped=False,
                    rejection_reasons=reasons,
                    active_resource_profile=self.active_profile_name,
                    resource_profile_limits={
                        "max_dim": self.max_dim,
                        "max_pixels": self.max_pixels,
                        "max_file_size": self.max_file_size,
                    },
                )
                return ImageSentinelReport(
                    filename=filename,
                    status=ImageAssuranceStatus.REJECTED,
                    risk_score=risk_score,
                    evidence_strength=EvidenceStrength.HIGH,
                    safety_result=safety_res,
                    findings=[
                        Finding(
                            detector_id="image_sentinel_resource_limits" if is_resource_limit else "image_sentinel_decoder",
                            finding_type=finding_type,
                            risk_score=risk_score,
                            evidence_strength=EvidenceStrength.HIGH,
                            title=title,
                            explanation=explanation,
                            limitations=limitations,
                            is_hard_gate=True,
                            evidence_details={"rejection_reasons": reasons, "profile": self.active_profile_name},
                        )
                    ],
                    limitations_disclaimer=self.MANDATORY_DISCLAIMER,
                )

            # Build Safety Result object
            safety_res = ImageSafetyResult(
                is_valid=True,
                status=ImageAssuranceStatus.SAFE_UNDER_CHECKS,
                original_sha256=worker_data["original_sha256"],
                canonical_sha256=worker_data["canonical_sha256"],
                perceptual_hash=worker_data["perceptual_hash"],
                original_format=worker_data["original_format"],
                canonical_format="PNG",
                dimensions=worker_data["dimensions"],
                color_mode=worker_data["color_mode"],
                metadata_stripped=True,
                decode_worker_metadata=worker_data["decode_worker_metadata"],
                configured_limits=worker_data["configured_limits"],
                protocol_version="ronova_image_sentinel_v1",
                active_resource_profile=self.active_profile_name,
                resource_profile_limits={
                    "max_dim": self.max_dim,
                    "max_pixels": self.max_pixels,
                    "max_file_size": self.max_file_size,
                },
            )

            # 4. Check Calibration Availability
            calibration = None
            if self.calibration_path.exists():
                try:
                    with open(self.calibration_path, "r", encoding="utf-8") as f:
                        calibration = json.load(f)
                except Exception:
                    pass

            if not calibration:
                # Return UNGUARANTEED if reference calibration is unavailable
                safety_res.status = ImageAssuranceStatus.UNGUARANTEED
                return ImageSentinelReport(
                    filename=filename,
                    status=ImageAssuranceStatus.UNGUARANTEED,
                    risk_score=0.0,
                    evidence_strength=EvidenceStrength.LOW,
                    safety_result=safety_res,
                    findings=[
                        Finding(
                            detector_id="image_sentinel_calibration",
                            finding_type="UNGUARANTEED_NO_CALIBRATION",
                            risk_score=0.0,
                            evidence_strength=EvidenceStrength.LOW,
                            title="Image Assurance Unguaranteed: Missing Calibration Baseline",
                            explanation="Image integrity checks passed, but reference calibration baseline is absent. Deep-learning anomaly evaluation cannot be completed.",
                            limitations="Deep-learning anomaly scoring requires an explicit clean reference calibration artifact.",
                            is_hard_gate=False,
                        )
                    ],
                    limitations_disclaimer=self.MANDATORY_DISCLAIMER,
                )

            # 5. Check Model Input Compatibility & Inspect Layout
            if not model_file.exists():
                safety_res.status = ImageAssuranceStatus.UNGUARANTEED
                return ImageSentinelReport(
                    filename=filename,
                    status=ImageAssuranceStatus.UNGUARANTEED,
                    risk_score=0.0,
                    evidence_strength=EvidenceStrength.LOW,
                    safety_result=safety_res,
                    findings=[
                        Finding(
                            detector_id="image_sentinel_model",
                            finding_type="UNGUARANTEED_MODEL_UNAVAILABLE",
                            risk_score=0.0,
                            evidence_strength=EvidenceStrength.LOW,
                            title="Image Assurance Unguaranteed: Model Artifact Unavailable",
                            explanation=f"Clean reference classifier model not found at {model_file}.",
                            limitations="Transformation stability analysis requires access to a valid clean ONNX model artifact.",
                            is_hard_gate=False,
                        )
                    ],
                    limitations_disclaimer=self.MANDATORY_DISCLAIMER,
                )

            adapter_name, input_meta, adapter_err = _inspect_and_select_adapter(model_file)
            if adapter_err or adapter_name is None:
                safety_res.status = ImageAssuranceStatus.UNGUARANTEED
                safety_res.analysis_model_path = str(model_file)
                safety_res.analysis_model_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
                return ImageSentinelReport(
                    filename=filename,
                    status=ImageAssuranceStatus.UNGUARANTEED,
                    risk_score=0.0,
                    evidence_strength=EvidenceStrength.LOW,
                    safety_result=safety_res,
                    findings=[
                        Finding(
                            detector_id="image_sentinel_model",
                            finding_type="UNGUARANTEED_UNSUPPORTED_MODEL_INPUT",
                            risk_score=0.0,
                            evidence_strength=EvidenceStrength.LOW,
                            title="Image Assurance Unguaranteed: Unsupported Model Input Layout",
                            explanation=adapter_err or "Selected model input layout cannot be satisfied by registered adapters.",
                            limitations="Model input inspection requires an explicitly registered preprocessing adapter.",
                            is_hard_gate=False,
                        )
                    ],
                    limitations_disclaimer=self.MANDATORY_DISCLAIMER,
                )

            # Record model provenance metadata
            model_bytes = model_file.read_bytes()
            model_sha256 = hashlib.sha256(model_bytes).hexdigest()
            calib_bytes = self.calibration_path.read_bytes() if self.calibration_path.exists() else b""
            calib_sha256 = hashlib.sha256(calib_bytes).hexdigest() if calib_bytes else None

            # 6. Build Deterministic Transformation Suite
            with Image.open(canonical_output_path) as canonical_img:
                rgb_canonical = canonical_img.convert("RGB")
                
                # Deterministic seed from canonical SHA-256
                seed_val = int(worker_data["canonical_sha256"][:8], 16)
                np.random.seed(seed_val % (2**32 - 1))

                # Prepare baseline 28x28 single-channel array (matching MNIST clean classifier)
                baseline_28 = rgb_canonical.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                baseline_arr = np.array(baseline_28, dtype=np.float32) / 255.0
                baseline_arr = baseline_arr.reshape(1, 1, 28, 28)

                # Variant 1: Brightness shift (+10%)
                enhancer = ImageEnhance.Brightness(rgb_canonical)
                v1_img = enhancer.enhance(1.10).convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                v1_arr = (np.array(v1_img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)

                # Variant 2: Mild Gaussian Blur (radius 0.8)
                v2_img = rgb_canonical.filter(ImageFilter.GaussianBlur(radius=0.8)).convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                v2_arr = (np.array(v2_img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)

                # Variant 3: JPEG Recompression (quality=85)
                jpeg_buf = io.BytesIO()
                rgb_canonical.save(jpeg_buf, format="JPEG", quality=85)
                jpeg_buf.seek(0)
                v3_img = Image.open(jpeg_buf).convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                v3_arr = (np.array(v3_img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)

                # Variant 4: Bounded Gaussian Noise (std=0.03)
                noise = np.random.normal(0.0, 0.03, size=baseline_arr.shape).astype(np.float32)
                v4_arr = np.clip(baseline_arr + noise, 0.0, 1.0)

                # Variant 5: Small Translation (shift 1px)
                v5_img = ImageOps.expand(rgb_canonical, border=(1, 1, 0, 0), fill=0).crop((0, 0, rgb_canonical.width, rgb_canonical.height))
                v5_img = v5_img.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                v5_arr = (np.array(v5_img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)

            transformations = [
                ("baseline", baseline_arr),
                ("t1_brightness", v1_arr),
                ("t2_blur", v2_arr),
                ("t3_jpeg", v3_arr),
                ("t4_noise", v4_arr),
                ("t5_translation", v5_arr),
            ]

            # 7. Model Inference & Transformation Stability Analysis
            batch_input = np.vstack([t[1] for t in transformations])  # shape (6, 1, 28, 28)
            runner = IsolatedSandboxRunner(timeout_seconds=10.0)
            probs, _ = runner.run_inference(str(model_file), batch_input)

            baseline_probs = probs[0]
            baseline_class = int(np.argmax(baseline_probs))
            baseline_confidence = float(baseline_probs[baseline_class])
            baseline_entropy = _calculate_entropy(baseline_probs)

            # Store real provenance fields on safety_res
            safety_res.analysis_model_path = str(model_file)
            safety_res.analysis_model_sha256 = model_sha256
            safety_res.adapter_name = adapter_name
            safety_res.calibration_sha256 = calib_sha256
            safety_res.transformation_seed = seed_val
            safety_res.set_provenance_arrays(baseline_arr, baseline_probs)

            transformed_probs = probs[1:]
            transformed_classes = [int(np.argmax(p)) for p in transformed_probs]

            # Top-class flip rate
            flips = sum(1 for c in transformed_classes if c != baseline_class)
            flip_rate = round(flips / len(transformed_classes), 4)

            # Average JS divergence
            js_divs = [_jensen_shannon_divergence(baseline_probs, p) for p in transformed_probs]
            avg_js_div = round(float(np.mean(js_divs)), 4)

            # Max confidence drop
            conf_drops = [max(0.0, baseline_confidence - float(p[baseline_class])) for p in transformed_probs]
            max_conf_drop = round(float(max(conf_drops)), 4)

            # Entropy change
            trans_entropies = [_calculate_entropy(p) for p in transformed_probs]
            entropy_change = round(float(np.mean(trans_entropies) - baseline_entropy), 4)

            # Prediction distribution L2 distance from reference mean
            ref_pred_means = np.array(calibration.get("pred_class_distribution", [0.5, 0.5]))
            pred_dist = round(float(np.linalg.norm(baseline_probs - ref_pred_means)), 4)

            metrics = {
                "baseline_class": baseline_class,
                "baseline_confidence": round(baseline_confidence, 4),
                "baseline_entropy": round(baseline_entropy, 4),
                "transformed_classes": transformed_classes,
                "top_class_flip_rate": flip_rate,
                "average_js_divergence": avg_js_div,
                "max_confidence_drop": max_conf_drop,
                "entropy_change": entropy_change,
                "prediction_distribution_distance": pred_dist,
                "embedding_l2_distance": pred_dist,  # legacy alias
                "calibration_entropy_threshold": calibration.get("anomaly_entropy_threshold", 0.75),
            }

            # 8. Verdict Determination & Finding Construction
            flip_thresh = calibration.get("max_flip_rate_threshold", 0.25)
            js_thresh = calibration.get("max_js_divergence_threshold", 0.15)
            entropy_thresh = calibration.get("anomaly_entropy_threshold", 0.75)
            pred_dist_thresh = calibration.get("max_prediction_distribution_distance", calibration.get("max_l2_feature_distance", 1.5))

            is_suspicious = (
                flip_rate > flip_thresh
                or avg_js_div > js_thresh
                or baseline_entropy > entropy_thresh
                or pred_dist > pred_dist_thresh
            )

            findings = []
            if is_suspicious:
                safety_res.status = ImageAssuranceStatus.SUSPICIOUS
                risk_score = round(min(90.0, 50.0 + flip_rate * 40.0 + avg_js_div * 100.0), 2)
                evidence_str = EvidenceStrength.HIGH if flip_rate >= 0.4 else EvidenceStrength.MEDIUM

                reasons = []
                if flip_rate > flip_thresh:
                    reasons.append(f"High Prediction Instability Indicator (flip_rate={flip_rate} > {flip_thresh})")
                if avg_js_div > js_thresh:
                    reasons.append(f"Trigger-Like Behavioral Response Indicator (js_divergence={avg_js_div} > {js_thresh})")
                if baseline_entropy > entropy_thresh:
                    reasons.append(f"Out-of-Distribution Image Indicator (baseline_entropy={baseline_entropy:.4f} > {entropy_thresh})")
                if pred_dist > pred_dist_thresh:
                    reasons.append(f"Prediction Distribution Shift Indicator (pred_dist={pred_dist} > {pred_dist_thresh})")

                findings.append(
                    Finding(
                        detector_id="image_sentinel_stability",
                        finding_type="SUSPICIOUS_IMAGE_BEHAVIOR_INDICATOR",
                        risk_score=risk_score,
                        evidence_strength=evidence_str,
                        title="Prediction Distribution Shift Indicator Fired",
                        explanation=f"Image exhibits anomalous response behavior under transformation: {'; '.join(reasons)}.",
                        limitations="Prediction instability or distribution shift indicates unusual response but does not alone prove malicious intent.",
                        evidence_details=metrics,
                        is_hard_gate=False,
                    )
                )
            else:
                safety_res.status = ImageAssuranceStatus.SAFE_UNDER_CHECKS
                risk_score = 2.5
                evidence_str = EvidenceStrength.HIGH
                findings.append(
                    Finding(
                        detector_id="image_sentinel_stability",
                        finding_type="SAFE_UNDER_CHECKS_PASSED",
                        risk_score=2.5,
                        evidence_strength=EvidenceStrength.HIGH,
                        title="Image Passed Configured Assurance Checks",
                        explanation="Image passed format decoding, metadata stripping, transformation stability, and baseline distribution checks.",
                        limitations=self.MANDATORY_DISCLAIMER,
                        evidence_details=metrics,
                        is_hard_gate=False,
                    )
                )

            return ImageSentinelReport(
                filename=filename,
                status=safety_res.status,
                risk_score=risk_score,
                evidence_strength=evidence_str,
                safety_result=safety_res,
                transformation_metrics=metrics,
                findings=findings,
                limitations_disclaimer=self.MANDATORY_DISCLAIMER,
            )

        finally:
            # Secure cleanup of temporary worker files
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
