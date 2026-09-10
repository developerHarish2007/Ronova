import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.api.main import app
from fastapi.testclient import TestClient


def main():
    client = TestClient(app)

    clean_model_path = "models/clean_classifier.onnx"
    backdoored_model_path = "models/backdoored_classifier.onnx"
    poisoned_dataset_path = "data/poisoned_dataset.npy"
    live_input_path = "data/live_input_normal.npy"

    print("=======================================================")
    print("RONOVA Phase 4 Verification: Policy Engine & API Routes")
    print("=======================================================")

    try:
        # 1. Test POST /scan/model with Clean Classifier
        print("\n[1/5] Testing POST /scan/model with clean_classifier.onnx...")
        with open(clean_model_path, "rb") as f:
            res_clean = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})

        assert res_clean.status_code == 200, f"Expected 200 OK, got {res_clean.status_code}: {res_clean.text}"
        json_clean = res_clean.json()
        policy_clean = json_clean["policy"]

        print(f"      Governed Verdict    : {policy_clean['verdict']}")
        print(f"      Risk Score          : {policy_clean['risk_score']}/100")
        print(f"      Evidence Strength   : {policy_clean['evidence_strength']}")
        print(f"      Hard Gate Fired     : {policy_clean['hard_gate_fired']}")
        print(f"      Manifest Verification: {json_clean['manifest_verification']['status']}")

        assert policy_clean["verdict"] == "ACCEPT", f"Clean model should produce ACCEPT, got {policy_clean['verdict']}"
        assert policy_clean["hard_gate_fired"] is False, "Clean model should not fire any hard gate"

        # 2. Test POST /scan/model with Backdoored Classifier
        print("\n[2/5] Testing POST /scan/model with backdoored_classifier.onnx...")
        with open(backdoored_model_path, "rb") as f:
            res_backdoored = client.post("/scan/model", files={"file": ("backdoored_classifier.onnx", f, "application/octet-stream")})

        assert res_backdoored.status_code == 200, f"Expected 200 OK, got {res_backdoored.status_code}"
        json_backdoored = res_backdoored.json()
        policy_backdoored = json_backdoored["policy"]

        print(f"      Governed Verdict    : {policy_backdoored['verdict']}")
        print(f"      Risk Score          : {policy_backdoored['risk_score']}/100")
        print(f"      Evidence Strength   : {policy_backdoored['evidence_strength']}")
        print(f"      Hard Gate Fired     : {policy_backdoored['hard_gate_fired']}")
        print(f"      Hard Gates Triggered: {policy_backdoored['hard_gates_triggered']}")

        assert policy_backdoored["verdict"] == "QUARANTINE", f"Backdoored model should produce QUARANTINE, got {policy_backdoored['verdict']}"
        assert policy_backdoored["hard_gate_fired"] is True, "Backdoored model should fire hard gate"
        assert "High-Strength Trojan Indicator" in policy_backdoored["hard_gates_triggered"], "Expected High-Strength Trojan Indicator hard gate"
        assert "Unapproved Model Hash / Manifest Mismatch" in policy_backdoored["hard_gates_triggered"], "Expected Manifest Mismatch hard gate"

        # 3. Test POST /scan/dataset with Poisoned Dataset
        print("\n[3/5] Testing POST /scan/dataset with poisoned_dataset.npy...")
        with open(poisoned_dataset_path, "rb") as f:
            res_ds = client.post("/scan/dataset", files={"file": ("poisoned_dataset.npy", f, "application/octet-stream")})

        assert res_ds.status_code == 200
        json_ds = res_ds.json()
        print(f"      Dataset Risk Assessment: {json_ds['dataset_risk_assessment']}")
        print(f"      Duplicate Pairs        : {json_ds['duplicate_pairs_count']}")
        print(f"      Anomalous Samples      : {json_ds['anomalous_samples_count']}")

        assert json_ds["dataset_risk_assessment"] == "REVIEW", "Poisoned dataset should return REVIEW"

        # 4. Test POST /log/inference & GET /verify/certificate
        print("\n[4/5] Testing POST /log/inference & GET /verify/certificate...")
        log_res = client.post("/log/inference", json={"model_sha256": json_clean["sha256"], "input_data_summary": "Test Sample 0", "prediction": "Class 0"})
        assert log_res.status_code == 200
        log_json = log_res.json()
        checkpoint = log_json["signed_checkpoint"]

        verify_res = client.get("/verify/certificate", params={
            "signature": checkpoint["signature"],
            "event_hash": checkpoint["checkpoint_payload"]["event_hash"],
            "model_sha256": checkpoint["checkpoint_payload"]["model_sha256"],
            "timestamp": checkpoint["checkpoint_payload"]["timestamp"],
            "previous_event_hash": checkpoint["checkpoint_payload"]["previous_event_hash"],
        })
        assert verify_res.status_code == 200
        verify_json = verify_res.json()
        print(f"      Certificate Verification Status: {verify_json['verification_status']}")
        assert verify_json["verification_status"] == "VALID", "Certificate verification should be VALID"

        # 5. Test POST /check/input (Live Batch Input Drift Route)
        print("\n[5/5] Testing POST /check/input on live batch data...")
        if not os.path.exists(live_input_path):
            # Create a simple valid batch array if needed
            np.save(live_input_path, np.zeros((10, 1, 28, 28), dtype=np.float32))

        with open(live_input_path, "rb") as f:
            res_input = client.post("/check/input", files={"file": ("live_input_normal.npy", f, "application/octet-stream")})

        assert res_input.status_code == 200, f"Expected 200 OK for /check/input, got {res_input.status_code}"
        json_input = res_input.json()
        print(f"      Check Input Method      : {json_input['method']}")
        print(f"      Check Input Status      : {json_input['status']}")
        assert json_input["method"] == "KS"
        assert json_input["status"] in ["NORMAL", "ELEVATED", "HIGH"]

        # Save artifact JSON verdicts
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/clean_policy_verdict.json", "w", encoding="utf-8") as f:
            json.dump(policy_clean, f, indent=2)
        with open("artifacts/backdoored_policy_verdict.json", "w", encoding="utf-8") as f:
            json.dump(policy_backdoored, f, indent=2)

        print("\n=======================================================")
        print("PHASE 4 POLICY ENGINE VERIFICATION PASSED SUCCESSFULLY!")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 4 Policy Engine Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
