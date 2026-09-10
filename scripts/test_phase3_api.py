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

    print("=======================================================")
    print("RONOVA Phase 3 Verification: Testing POST /scan/model API")
    print("=======================================================")

    try:
        # 1. Test Clean Model Upload
        print("\n[1/2] Uploading clean classifier model -> POST /scan/model...")
        with open(clean_model_path, "rb") as f:
            res_clean = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})

        assert res_clean.status_code == 200, f"Expected 200 OK, got {res_clean.status_code}: {res_clean.text}"
        json_clean = res_clean.json()

        print(f"      Filename               : {json_clean['filename']}")
        print(f"      SHA-256 Digest         : {json_clean['sha256'][:16]}...")
        print(f"      Manifest Verification  : Matched = {json_clean['manifest_verification']['manifest_matched']}")
        print(f"      STRIP Trojan Triggered : {json_clean['strip_analysis']['trojan_indicator_triggered']}")
        print(f"      Drift Status           : {json_clean['input_distribution_drift']['status']}")
        print(f"      8-Field Provenance     : {list(json_clean['provenance_manifest'].keys())}")
        print(f"      Checkpoint Signature   : {json_clean['signed_checkpoint']['verification_status']}")
        print(f"      Governed Verdict       : {json_clean['policy']['verdict']}")

        assert json_clean["manifest_verification"]["manifest_matched"] is True, "Clean model should match manifest"
        assert json_clean["strip_analysis"]["trojan_indicator_triggered"] is False, "Clean model should not trigger Trojan indicator"
        assert json_clean["signed_checkpoint"]["verification_status"] == "VALID", "Checkpoint signature should be VALID"
        assert json_clean["policy"]["verdict"] == "ACCEPT", "Clean model should produce ACCEPT verdict"

        # 2. Test Backdoored Model Upload
        print("\n[2/2] Uploading backdoored classifier model -> POST /scan/model...")
        with open(backdoored_model_path, "rb") as f:
            res_backdoored = client.post("/scan/model", files={"file": ("backdoored_classifier.onnx", f, "application/octet-stream")})

        assert res_backdoored.status_code == 200, f"Expected 200 OK, got {res_backdoored.status_code}"
        json_backdoored = res_backdoored.json()

        print(f"      Filename               : {json_backdoored['filename']}")
        print(f"      SHA-256 Digest         : {json_backdoored['sha256'][:16]}...")
        print(f"      Manifest Verification  : Matched = {json_backdoored['manifest_verification']['manifest_matched']}")
        print(f"      STRIP Trojan Triggered : {json_backdoored['strip_analysis']['trojan_indicator_triggered']}")
        print(f"      Checkpoint Signature   : {json_backdoored['signed_checkpoint']['verification_status']}")
        print(f"      Governed Verdict       : {json_backdoored['policy']['verdict']}")

        assert json_backdoored["manifest_verification"]["manifest_matched"] is False, "Backdoored model hash should mismatch manifest"
        assert json_backdoored["strip_analysis"]["trojan_indicator_triggered"] is True, "Backdoored model should trigger Trojan indicator"
        assert json_backdoored["signed_checkpoint"]["verification_status"] == "VALID", "Checkpoint signature should be VALID"
        assert json_backdoored["policy"]["verdict"] == "QUARANTINE", "Backdoored model should produce QUARANTINE verdict"

        # Save artifact JSON responses
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/api_clean_response.json", "w", encoding="utf-8") as f:
            json.dump(json_clean, f, indent=2)
        with open("artifacts/api_backdoored_response.json", "w", encoding="utf-8") as f:
            json.dump(json_backdoored, f, indent=2)

        print("\n=======================================================")
        print("PHASE 3 API & PROVENANCE VERIFICATION PASSED SUCCESSFULLY!")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 3 API Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
