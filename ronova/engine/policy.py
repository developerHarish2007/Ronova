from typing import List, Tuple, Dict, Any
from ronova.core.types import (
    Finding,
    EvidenceStrength,
    PolicyVerdict,
    SafetyGateResult,
)


class EvidenceEngine:
    """
    RONOVA Evidence Engine:
    Normalizes findings, calculates overall Risk Score (0-100) and Evidence Strength (LOW/MEDIUM/HIGH)
    without collapsing them into a probability claim.
    """

    @staticmethod
    def aggregate_evidence(findings: List[Finding]) -> Tuple[float, EvidenceStrength]:
        if not findings:
            return 0.0, EvidenceStrength.LOW

        total_risk = sum(f.risk_score for f in findings)
        overall_risk_score = round(total_risk / len(findings), 2)

        strength_map = {EvidenceStrength.LOW: 1, EvidenceStrength.MEDIUM: 2, EvidenceStrength.HIGH: 3}
        inv_map = {1: EvidenceStrength.LOW, 2: EvidenceStrength.MEDIUM, 3: EvidenceStrength.HIGH}
        max_strength_val = max(strength_map[f.evidence_strength] for f in findings)

        return overall_risk_score, inv_map[max_strength_val]


class PolicyEngine:
    """
    RONOVA Policy Engine (Phase 4):
    Two-Layer Governance Engine:
    - Level 1 Hard Gates (Instant QUARANTINE, never averaged away):
      1. Model hash mismatch vs. approved manifest (MANIFEST_MISMATCH_UNAPPROVED_MODEL)
      2. Unsafe serialization detected in static scan (UNSAFE_SERIALIZATION_INDICATOR)
      3. Broken / invalid hash chain in audit ledger (BROKEN_AUDIT_CHAIN)
      4. High-Strength Trojan Indicator (STRIP threshold breach)
    - Level 2 Weighted Risk (evaluated only if no hard gate fired):
      -> ACCEPT or REVIEW based on weighted risk score.
    """

    HARD_GATE_TYPES = {
        "MANIFEST_MISMATCH",
        "INVALID_FILE_FORMAT",
        "UNSAFE_SERIALIZATION_INDICATOR",
        "BROKEN_AUDIT_CHAIN",
        "TROJAN_BEHAVIOR_INDICATOR",
    }

    def evaluate(
        self,
        safety_gate_result: SafetyGateResult,
        detector_findings: List[Finding],
        sandbox_mode: str = "subprocess_isolated_fallback",
        audit_chain_valid: bool = True,
        audit_chain_status: str = "",
    ) -> Dict[str, Any]:
        all_findings: List[Finding] = list(safety_gate_result.gate_findings) + detector_findings

        # Check for broken audit chain
        if not audit_chain_valid:
            exp_text = f"Audit ledger hash chain validation failed: {audit_chain_status}" if audit_chain_status else "Audit ledger hash chain validation failed."
            all_findings.append(
                Finding(
                    detector_id="audit_chain_validator",
                    finding_type="BROKEN_AUDIT_CHAIN",
                    risk_score=100.0,
                    evidence_strength=EvidenceStrength.HIGH,
                    title="Broken Audit Ledger Chain",
                    explanation=exp_text,
                    limitations="Evaluates sequential event hash linkage and payload integrity.",
                    evidence_details={"audit_chain_valid": False, "status": audit_chain_status},
                    is_hard_gate=True,
                )
            )

        hard_gates_triggered: List[str] = []

        for f in all_findings:
            if f.is_hard_gate or f.finding_type in self.HARD_GATE_TYPES or f.title == "High-Strength Trojan Indicator":
                hard_gates_triggered.append(f.title)

        overall_risk_score, overall_strength = EvidenceEngine.aggregate_evidence(all_findings)
        hard_gate_fired = len(hard_gates_triggered) > 0

        # Level 1 Evaluation
        if hard_gate_fired:
            verdict = PolicyVerdict.QUARANTINE
            overall_risk_score = max(overall_risk_score, 95.0)
        else:
            # Level 2 Evaluation
            if overall_risk_score >= 50.0:
                verdict = PolicyVerdict.REVIEW
            else:
                verdict = PolicyVerdict.ACCEPT

        return {
            "verdict": verdict.value,
            "risk_score": overall_risk_score,
            "evidence_strength": overall_strength.value,
            "hard_gate_fired": hard_gate_fired,
            "hard_gates_triggered": hard_gates_triggered,
            "sandbox_mode": sandbox_mode,
            "findings": [f.model_dump() for f in all_findings],
        }
