import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure root package directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.safety.gate import SafetyGate
from ronova.sandbox.runner import IsolatedSandboxRunner
from ronova.detectors.strip import STRIPDetector
from ronova.engine.policy import PolicyEngine


def run_pipeline_for_model(
    model_path: str,
    test_samples_path: str,
    clean_overlays_path: str,
    output_json_path: str,
) -> dict:
    print(f"\n=======================================================")
    print(f"RONOVA Core Pipeline: Assessing {model_path}")
    print(f"=======================================================")

    # Step 1: Safety Gate
    safety_gate = SafetyGate()
    print("[1/4] Running Safety Gate (SHA-256 + ONNX format + ModelScan)...")
    safety_result = safety_gate.scan_file(model_path)
    print(f"      SHA-256 Digest : {safety_result.sha256}")
    print(f"      ModelScan Passed: {safety_result.modelscan_passed} (Issues: {safety_result.modelscan_issues_count})")
    print(f"      Hard Gate Fired : {safety_result.hard_gate_triggered}")

    # Step 2: Isolated Sandbox & Data Prep
    print("[2/4] Initializing Isolated Sandbox Engine...")
    sandbox = IsolatedSandboxRunner(timeout_seconds=10.0)

    test_samples = np.load(test_samples_path)
    clean_overlays = np.load(clean_overlays_path)

    # Step 3: STRIP Trojan Detector
    print("[3/4] Running STRIP Behavioral Entropy Trojan Detector...")
    strip_detector = STRIPDetector(
        sandbox_runner=sandbox,
        num_perturbations=25,
        blend_alpha=0.5,
        entropy_threshold=0.35,
    )
    strip_result, sandbox_meta = strip_detector.evaluate_model(
        model_path, test_samples, clean_overlays
    )

    print(f"      Evaluated Samples       : {strip_result.num_samples_evaluated}")
    print(f"      Mean Entropy           : {strip_result.mean_entropy:.4f}")
    print(f"      Std Entropy            : {strip_result.std_entropy:.4f}")
    print(f"      Trojan Indicator Triggered: {strip_result.trojan_indicator_triggered}")
    print(f"      Sandbox Execution Mode : {sandbox_meta.get('sandbox_mode')}")

    # Step 4: Policy & Evidence Engine
    print("[4/4] Evaluating Policy Engine (Hard Gates -> Weighted Risk)...")
    policy_engine = PolicyEngine()
    assurance_report = policy_engine.evaluate(
        safety_gate_result=safety_result,
        detector_findings=[strip_result.finding],
        sandbox_mode=sandbox_meta.get("sandbox_mode", "subprocess_isolated_fallback"),
    )

    # Save output JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(assurance_report, f, indent=2)

    verdict_str = assurance_report.get("verdict")
    risk_score = assurance_report.get("risk_score")
    evidence_str = assurance_report.get("evidence_strength")
    hard_gates = assurance_report.get("hard_gates_triggered", [])

    print(f"\n---> VERDICT: {verdict_str}")
    print(f"---> Overall Risk Score       : {risk_score}/100")
    print(f"---> Overall Evidence Strength: {evidence_str}")
    if hard_gates:
        print(f"---> Hard Gates Triggered     : {hard_gates}")
    print(f"Assurance Certificate Saved -> {output_json_path}")
    return assurance_report


def main():
    os.makedirs("artifacts", exist_ok=True)

    clean_model = "models/clean_classifier.onnx"
    backdoored_model = "models/backdoored_classifier.onnx"

    clean_test_data = "data/test_samples_clean.npy"
    triggered_test_data = "data/test_samples_triggered.npy"
    clean_overlays = "data/clean_overlays.npy"

    try:
        print("\n--- RUNNING PIPELINE ON CLEAN CLASSIFIER ---")
        clean_report = run_pipeline_for_model(
            model_path=clean_model,
            test_samples_path=clean_test_data,
            clean_overlays_path=clean_overlays,
            output_json_path="artifacts/clean_verdict.json",
        )

        print("\n--- RUNNING PIPELINE ON BACKDOORED CLASSIFIER ---")
        backdoored_report = run_pipeline_for_model(
            model_path=backdoored_model,
            test_samples_path=triggered_test_data,
            clean_overlays_path=clean_overlays,
            output_json_path="artifacts/backdoored_verdict.json",
        )

        # Sanity Verification
        assert clean_report["verdict"] == "ACCEPT", f"Expected ACCEPT for clean model, got {clean_report['verdict']}"
        assert backdoored_report["verdict"] == "QUARANTINE", f"Expected QUARANTINE for backdoored model, got {backdoored_report['verdict']}"
        assert "High-Strength Trojan Indicator" in backdoored_report["hard_gates_triggered"], "Expected Trojan Indicator hard gate"

        print("\n=======================================================")
        print("PHASE 1 CORE PIPELINE VERIFICATION PASSED SUCCESSFULLY!")
        print("Clean Model     -> Verdict: ACCEPT (Entropy normal)")
        print("Backdoored Model -> Verdict: QUARANTINE (High-Strength Trojan Indicator fired)")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 1 Core Pipeline Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
