import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr


class EvidenceStrength(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PolicyVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    QUARANTINE = "QUARANTINE"


class ImageAssuranceStatus(str, Enum):
    REJECTED = "REJECTED"
    SUSPICIOUS = "SUSPICIOUS"
    SAFE_UNDER_CHECKS = "SAFE_UNDER_CHECKS"
    UNGUARANTEED = "UNGUARANTEED"


class Finding(BaseModel):
    detector_id: str
    finding_type: str
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk Score between 0 and 100")
    evidence_strength: EvidenceStrength
    title: str
    explanation: str
    limitations: str
    evidence_details: Dict[str, Any] = Field(default_factory=dict)
    is_hard_gate: bool = False
    timestamp: float = Field(default_factory=time.time)


class SafetyGateResult(BaseModel):
    sha256: str
    file_type: str
    is_onnx_valid: bool
    modelscan_passed: bool
    modelscan_issues_count: int
    modelscan_raw: Dict[str, Any] = Field(default_factory=dict)
    signature_check_status: str = "STUBBED_PENDING_PHASE3"
    hard_gate_triggered: bool = False
    gate_findings: List[Finding] = Field(default_factory=list)
    onnx_validation_details: Dict[str, Any] = Field(default_factory=dict)


class STRIPResult(BaseModel):
    num_samples_evaluated: int
    num_perturbations_per_sample: int
    mean_entropy: float
    std_entropy: float
    min_entropy: float
    trojan_indicator_triggered: bool
    entropy_distribution: List[float]
    sample_summaries: List[Dict[str, Any]]
    finding: Finding


class ImageSafetyResult(BaseModel):
    is_valid: bool
    status: ImageAssuranceStatus
    original_sha256: str
    canonical_sha256: str
    perceptual_hash: str
    original_format: str
    canonical_format: str = "PNG"
    dimensions: List[int]
    color_mode: str
    metadata_stripped: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    decode_worker_metadata: Dict[str, Any] = Field(default_factory=dict)
    configured_limits: Dict[str, Any] = Field(default_factory=dict)
    protocol_version: str = "ronova_image_sentinel_v1"
    analysis_model_path: Optional[str] = None
    analysis_model_sha256: Optional[str] = None
    adapter_name: Optional[str] = None
    calibration_sha256: Optional[str] = None
    transformation_seed: Optional[int] = None
    baseline_input_shape: Optional[List[int]] = None
    baseline_output_shape: Optional[List[int]] = None
    active_resource_profile: str = "Standard"
    resource_profile_limits: Dict[str, Any] = Field(default_factory=dict)

    _baseline_input_array: Any = PrivateAttr(default=None)
    _baseline_output_array: Any = PrivateAttr(default=None)

    def set_provenance_arrays(self, input_arr: Any, output_arr: Any) -> None:
        self._baseline_input_array = input_arr
        self._baseline_output_array = output_arr
        if hasattr(input_arr, "shape"):
            self.baseline_input_shape = list(input_arr.shape)
        if hasattr(output_arr, "shape"):
            self.baseline_output_shape = list(output_arr.shape)

    def get_baseline_input_array(self) -> Any:
        return self._baseline_input_array

    def get_baseline_output_array(self) -> Any:
        return self._baseline_output_array


class ImageSentinelReport(BaseModel):
    framework: str = "RONOVA v0.2 Image Sentinel"
    filename: str
    status: ImageAssuranceStatus
    risk_score: float
    evidence_strength: EvidenceStrength
    safety_result: ImageSafetyResult
    transformation_metrics: Dict[str, Any] = Field(default_factory=dict)
    findings: List[Finding] = Field(default_factory=list)
    limitations_disclaimer: str
    timestamp: float = Field(default_factory=time.time)


class AssuranceReport(BaseModel):
    framework: str = "RONOVA v0.2"
    model_sha256: str
    sandbox_mode: str
    verdict: PolicyVerdict
    overall_risk_score: float
    overall_evidence_strength: EvidenceStrength
    hard_gates_triggered: List[str]
    safety_gate: SafetyGateResult
    findings: List[Finding]
    pipeline_timestamp: float = Field(default_factory=time.time)
