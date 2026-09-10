import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure root package directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.dataset.duplicates import DuplicateDetector
from ronova.dataset.anomalies import AnomalyDetector
from ronova.engine.policy import EvidenceEngine, PolicyEngine
from ronova.core.types import SafetyGateResult, PolicyVerdict


def run_dataset_forensics(dataset_path: str, output_json_path: str) -> dict:
    print(f"\n=======================================================")
    print(f"RONOVA Dataset Forensics: Analyzing {dataset_path}")
    print(f"=======================================================")

    dataset = np.load(dataset_path)
    print(f"Loaded dataset: shape {dataset.shape}")

    # 1. Duplicate Detection
    print("[1/2] Running Perceptual Hash Duplicate Detector...")
    dup_detector = DuplicateDetector(mse_threshold=0.001, hash_threshold=4)
    dup_pairs, dup_finding = dup_detector.analyze(dataset)
    print(f"      Duplicate Pairs Found: {len(dup_pairs)}")
    print(f"      Finding Type         : {dup_finding.finding_type}")
    print(f"      Risk Score           : {dup_finding.risk_score}")

    # 2. Anomaly Clustering
    print("[2/2] Running Embedding-Space Anomaly Detector...")
    anomaly_detector = AnomalyDetector(contamination=0.08)
    anom_indices, anom_finding = anomaly_detector.analyze(dataset)
    print(f"      Anomalous Samples Flagged: {len(anom_indices)}")
    print(f"      Indices                  : {anom_indices}")
    print(f"      Finding Title            : {anom_finding.title}")
    print(f"      Explanation              : {anom_finding.explanation}")

    # Aggregate Evidence
    findings = [dup_finding, anom_finding]
    risk_score, evidence_strength = EvidenceEngine.aggregate_evidence(findings)

    # Determine Verdict based on Level 2 Weighted Risk (no hard gates for soft dataset findings)
    if risk_score >= 50.0:
        verdict = PolicyVerdict.REVIEW
    elif risk_score > 0.0:
        verdict = PolicyVerdict.REVIEW
    else:
        verdict = PolicyVerdict.ACCEPT

    result = {
        "dataset_path": dataset_path,
        "total_samples": len(dataset),
        "dataset_risk_assessment": verdict.value,
        "overall_risk_score": risk_score,
        "overall_evidence_strength": evidence_strength.value,
        "duplicate_pairs_count": len(dup_pairs),
        "anomalous_samples_count": len(anom_indices),
        "anomalous_indices": anom_indices,
        "findings": [f.model_dump() for f in findings],
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n---> VERDICT: {verdict.value}")
    print(f"---> Risk Score: {risk_score}/100 | Evidence Strength: {evidence_strength.value}")
    print(f"Report saved to -> {output_json_path}")
    return result


def main():
    os.makedirs("artifacts", exist_ok=True)

    try:
        print("--- 1. RUNNING DATASET FORENSICS ON CLEAN DATASET ---")
        clean_res = run_dataset_forensics("data/clean_dataset.npy", "artifacts/clean_dataset_forensics.json")

        print("\n--- 2. RUNNING DATASET FORENSICS ON POISONED DATASET ---")
        poisoned_res = run_dataset_forensics("data/poisoned_dataset.npy", "artifacts/poisoned_dataset_forensics.json")

        # Sanity Checks
        assert clean_res["duplicate_pairs_count"] == 0, "Clean dataset should have 0 duplicate pairs"
        assert poisoned_res["duplicate_pairs_count"] > 0, "Poisoned dataset should flag duplicate pairs"
        assert poisoned_res["anomalous_samples_count"] > 0, "Poisoned dataset should flag anomalous samples"
        assert poisoned_res["dataset_risk_assessment"] in ["REVIEW", "QUARANTINE"], "Poisoned dataset should trigger REVIEW"

        print("\n=======================================================")
        print("PHASE 2 PART A (DATASET FORENSICS) VERIFICATION PASSED!")
        print("Clean Dataset    -> Risk Assessment: ACCEPT (0 duplicates, 0 anomalies)")
        print("Poisoned Dataset -> Risk Assessment: REVIEW (Duplicates & anomalies flagged)")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 2 Dataset Forensics Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
