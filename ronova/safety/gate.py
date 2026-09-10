import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import onnx
import onnxruntime as ort
from modelscan.modelscan import ModelScan

from ronova.core.types import (
    Finding,
    EvidenceStrength,
    SafetyGateResult,
)


class SafetyGate:
    """
    RONOVA Safety Gate:
    Computes SHA-256 artifact digest, verifies model against out-of-band approved manifest,
    invokes ModelScan for static malicious serialization checks, and enforces strict ONNX graph/loadability validation.
    """

    def __init__(self, manifest_path: str = "manifests/approved_models.json"):
        self.modelscanner = ModelScan()
        self.manifest_path = Path(manifest_path).resolve()

    def compute_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def validate_onnx_artifact(self, file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Performs comprehensive structural, graph, input/output, and ONNX Runtime loadability validation.
        Does NOT execute arbitrary model code or run inference.
        Returns: (is_valid, status_code, details_dict)
        """
        if not os.path.exists(file_path):
            return False, "FILE_NOT_FOUND", {"error": "File does not exist on disk."}

        if not file_path.lower().endswith(".onnx"):
            return False, "INVALID_FILE_EXTENSION", {"error": "Artifact filename extension must be .onnx."}

        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "EMPTY_FILE", {"error": "Uploaded file is 0 bytes (empty)."}
        except Exception as e:
            return False, "FILE_ACCESS_ERROR", {"error": f"Failed to inspect file size: {str(e)}"}

        # 1. Parse ONNX Protobuf structure
        try:
            model = onnx.load(file_path)
        except Exception as e:
            return False, "PROTOBUF_PARSE_ERROR", {"error": f"Failed to parse file as ONNX protobuf: {str(e)}"}

        # 2. Check ONNX Graph Schema
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            return False, "MALFORMED_GRAPH", {"error": f"ONNX graph schema check failed: {str(e)}"}

        # 3. Validate Inputs and Outputs
        graph = model.graph
        if not graph or len(graph.input) == 0:
            return False, "NO_INPUTS", {"error": "ONNX model graph defines 0 inputs."}

        if len(graph.output) == 0:
            return False, "NO_OUTPUTS", {"error": "ONNX model graph defines 0 outputs."}

        input_details = []
        for inp in graph.input:
            shape = []
            if inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape"):
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            input_details.append({
                "name": inp.name,
                "shape": shape,
                "elem_type": inp.type.tensor_type.elem_type if inp.type.HasField("tensor_type") else 0,
            })

        output_details = []
        for out in graph.output:
            shape = []
            if out.type.HasField("tensor_type") and out.type.tensor_type.HasField("shape"):
                for dim in out.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            output_details.append({
                "name": out.name,
                "shape": shape,
                "elem_type": out.type.tensor_type.elem_type if out.type.HasField("tensor_type") else 0,
            })

        # 4. ONNX Runtime Session Loadability Check (CPU provider config)
        try:
            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.enable_cpu_mem_arena = False

            session = ort.InferenceSession(file_path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            return False, "ONNX_RUNTIME_LOAD_ERROR", {
                "error": f"ONNX Runtime failed to load session: {str(e)}"
            }

        details = {
            "status": "VALID_ONNX_MODEL",
            "file_size_bytes": file_size,
            "ir_version": getattr(model, "ir_version", None),
            "producer_name": getattr(model, "producer_name", None),
            "num_inputs": len(graph.input),
            "num_outputs": len(graph.output),
            "inputs": input_details,
            "outputs": output_details,
            "providers_used": session.get_providers(),
        }
        return True, "VALID_ONNX_MODEL", details

    def verify_onnx_header(self, file_path: str) -> bool:
        valid, _, _ = self.validate_onnx_artifact(file_path)
        return valid

    def verify_manifest(self, model_sha256: str) -> Tuple[bool, str]:
        """Compares uploaded model SHA-256 against entries in approved manifest."""
        if not self.manifest_path.exists():
            return False, "MANIFEST_NOT_FOUND"

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            approved_list = manifest_data.get("approved_models", [])
            for entry in approved_list:
                if entry.get("sha256", "").lower() == model_sha256.lower():
                    return True, f"MANIFEST_VERIFIED_OK (Approved ID: {entry.get('model_id')})"

            return False, "MANIFEST_MISMATCH_UNAPPROVED_MODEL"
        except Exception as e:
            return False, f"MANIFEST_READ_ERROR: {str(e)}"

    def scan_file(self, model_path: str) -> SafetyGateResult:
        model_path = str(Path(model_path).resolve())
        sha256 = self.compute_sha256(model_path)
        is_onnx_valid, onnx_status, onnx_details = self.validate_onnx_artifact(model_path)
        
        gate_findings: List[Finding] = []
        hard_gate_triggered = False

        if not is_onnx_valid:
            hard_gate_triggered = True
            gate_findings.append(
                Finding(
                    detector_id="safety_gate_format",
                    finding_type="INVALID_FILE_FORMAT",
                    risk_score=100.0,
                    evidence_strength=EvidenceStrength.HIGH,
                    title="Invalid Artifact Format",
                    explanation=f"Uploaded artifact failed ONNX structural validation: {onnx_details.get('error', onnx_status)}",
                    limitations="Validates ONNX graph schema, input/output definitions, and ONNX Runtime loadability; does not perform dynamic behavioral analysis.",
                    evidence_details={
                        "file_path": model_path,
                        "is_onnx_valid": False,
                        "onnx_status": onnx_status,
                        "validation_error": onnx_details.get("error"),
                    },
                    is_hard_gate=True,
                )
            )

        # Approved Manifest Check (never compare a model hash to itself)
        manifest_matched, manifest_status = self.verify_manifest(sha256)
        if not manifest_matched:
            hard_gate_triggered = True
            gate_findings.append(
                Finding(
                    detector_id="safety_gate_manifest",
                    finding_type="MANIFEST_MISMATCH",
                    risk_score=100.0,
                    evidence_strength=EvidenceStrength.HIGH,
                    title="Unapproved Model Hash / Manifest Mismatch",
                    explanation=(
                        f"Model SHA-256 digest ({sha256[:16]}...) was not found in the approved model manifest. "
                        f"Status: {manifest_status}. Unapproved or substituted models trigger instant quarantine."
                    ),
                    limitations=(
                        "Compares artifact SHA-256 against out-of-band approved manifest entries; "
                        "does not infer approval for unregistered hashes."
                    ),
                    evidence_details={"sha256": sha256, "manifest_status": manifest_status},
                    is_hard_gate=True,
                )
            )

        # Run ModelScan if valid ONNX
        modelscan_passed = True
        modelscan_issues_count = 0
        raw_issues_summary: List[Dict[str, Any]] = []

        if is_onnx_valid:
            try:
                self.modelscanner.scan(model_path)
                issues_obj = self.modelscanner.issues
                all_issues = getattr(issues_obj, "all_issues", [])
                modelscan_issues_count = len(all_issues)

                if modelscan_issues_count > 0:
                    modelscan_passed = False
                    hard_gate_triggered = True
                    for issue in all_issues:
                        issue_details = {
                            "severity": str(getattr(issue, "severity", "CRITICAL")),
                            "description": str(getattr(issue, "description", "Suspicious static construct detected")),
                            "operator": str(getattr(issue, "operator_name", "UNKNOWN")),
                        }
                        raw_issues_summary.append(issue_details)

                    gate_findings.append(
                        Finding(
                            detector_id="modelscan_static_analyzer",
                            finding_type="UNSAFE_SERIALIZATION_INDICATOR",
                            risk_score=100.0,
                            evidence_strength=EvidenceStrength.HIGH,
                            title="ModelScan Unsafe Serialization Detection",
                            explanation=f"Static scanner detected {modelscan_issues_count} unsafe model serialization / execution construct(s).",
                            limitations="Static AST / pattern analysis; does not execute model code dynamically.",
                            evidence_details={"issues": raw_issues_summary},
                            is_hard_gate=True,
                        )
                    )
            except Exception as e:
                raw_issues_summary.append({"scan_error": str(e)})

        return SafetyGateResult(
            sha256=sha256,
            file_type="ONNX Classifier",
            is_onnx_valid=is_onnx_valid,
            modelscan_passed=modelscan_passed,
            modelscan_issues_count=modelscan_issues_count,
            modelscan_raw={"issues": raw_issues_summary},
            signature_check_status=manifest_status,
            hard_gate_triggered=hard_gate_triggered,
            gate_findings=gate_findings,
            onnx_validation_details=onnx_details,
        )
