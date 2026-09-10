import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.api.main import app
from fastapi.testclient import TestClient


def main():
    client = TestClient(app)
    os.makedirs("artifacts", exist_ok=True)

    clean_model_path = "models/clean_classifier.onnx"
    backdoored_model_path = "models/backdoored_classifier.onnx"

    print("=======================================================")
    print("RONOVA Phase 7 Verification: Certificates & Standalone Verifier")
    print("=======================================================")

    try:
        # 1. Generate Untampered Clean Model Certificate
        print("\n[1/5] Generating canonical certificate for Clean Model...")
        with open(clean_model_path, "rb") as f:
            res_clean = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
        assert res_clean.status_code == 200
        clean_cert = res_clean.json()["canonical_certificate"]
        clean_cert_path = "artifacts/clean_certificate.json"
        with open(clean_cert_path, "w", encoding="utf-8") as f:
            json.dump(clean_cert, f, indent=2)
        print(f"      Saved -> {clean_cert_path} (Verdict: {clean_cert['canonical_payload']['verdict']})")

        # 2. Generate Untampered Backdoored Model Certificate
        print("\n[2/5] Generating canonical certificate for Backdoored Model...")
        with open(backdoored_model_path, "rb") as f:
            res_backdoored = client.post("/scan/model", files={"file": ("backdoored_classifier.onnx", f, "application/octet-stream")})
        assert res_backdoored.status_code == 200
        backdoored_cert = res_backdoored.json()["canonical_certificate"]
        backdoored_cert_path = "artifacts/backdoored_certificate.json"
        with open(backdoored_cert_path, "w", encoding="utf-8") as f:
            json.dump(backdoored_cert, f, indent=2)
        print(f"      Saved -> {backdoored_cert_path} (Verdict: {backdoored_cert['canonical_payload']['verdict']})")

        # 3. Test Standalone Verifier on Clean Certificate
        print("\n[3/5] Running standalone verify_certificate.py on clean_certificate.json...")
        res_clean_verify = subprocess.run([sys.executable, "verify_certificate.py", clean_cert_path], capture_output=True, text=True)
        print(res_clean_verify.stdout)
        assert res_clean_verify.returncode == 0, f"Expected returncode 0 (VALID), got {res_clean_verify.returncode}"

        # 4. Test Standalone Verifier on Backdoored Certificate
        print("\n[4/5] Running standalone verify_certificate.py on backdoored_certificate.json...")
        res_backdoored_verify = subprocess.run([sys.executable, "verify_certificate.py", backdoored_cert_path], capture_output=True, text=True)
        print(res_backdoored_verify.stdout)
        assert res_backdoored_verify.returncode == 0, f"Expected returncode 0 (VALID), got {res_backdoored_verify.returncode}"

        # 5. Create Tampered Certificate & Test Failure
        print("\n[5/5] Tampering with certificate fields & testing offline detection...")
        tampered_cert = json.loads(json.dumps(clean_cert))
        tampered_cert["canonical_payload"]["risk_score"] = 99.9
        tampered_cert["canonical_payload"]["verdict"] = "QUARANTINE"

        tampered_cert_path = "artifacts/tampered_certificate.json"
        with open(tampered_cert_path, "w", encoding="utf-8") as f:
            json.dump(tampered_cert, f, indent=2)

        res_tampered_verify = subprocess.run([sys.executable, "verify_certificate.py", tampered_cert_path], capture_output=True, text=True)
        print(res_tampered_verify.stdout)
        assert res_tampered_verify.returncode != 0, f"Expected non-zero returncode for tampered cert, got {res_tampered_verify.returncode}"

        print("=======================================================")
        print("PHASE 7 OFFLINE VERIFIER TEST SUITE PASSED SUCCESSFULLY!")
        print("1. Clean Certificate       -> VALID (Ed25519 Verified)")
        print("2. Backdoored Certificate  -> VALID (Ed25519 Verified)")
        print("3. Tampered Certificate    -> FAILED (Signature Mismatch Detected)")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 7 Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
