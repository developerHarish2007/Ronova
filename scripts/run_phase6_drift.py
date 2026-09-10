import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.api.main import app
from fastapi.testclient import TestClient


def generate_live_batches():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    # 1. Normal Live Batch (sampled directly from clean reference dataset)
    if os.path.exists("data/clean_dataset.npy"):
        clean_ref = np.load("data/clean_dataset.npy")
        normal_batch = clean_ref[:50].copy()
    elif os.path.exists("data/test_samples_clean.npy"):
        clean_ref = np.load("data/test_samples_clean.npy")
        normal_batch = clean_ref[:50].copy()
    else:
        normal_batch = np.zeros((50, 1, 28, 28), dtype=np.float32)

    np.save("data/live_input_normal.npy", normal_batch)

    # 2. Domain-Shifted Live Batch (corrupted with heavy noise, inverted gradients, domain shift)
    shifted_batch = np.random.uniform(0.6, 1.0, size=(50, 1, 28, 28)).astype(np.float32)
    for i in range(50):
        shifted_batch[i, 0] = np.sin(np.linspace(0, 10, 28 * 28)).reshape(28, 28)

    np.save("data/live_input_shifted.npy", shifted_batch)
    print("Generated data/live_input_normal.npy and data/live_input_shifted.npy.")


def main():
    generate_live_batches()

    client = TestClient(app)

    print("\n=======================================================")
    print("RONOVA Phase 6 Verification: Input Distribution Drift")
    print("=======================================================")

    try:
        # 1. Test POST /check/input on Normal Live Batch
        print("\n[1/2] Testing POST /check/input on Normal Live Batch...")
        with open("data/live_input_normal.npy", "rb") as f:
            res_norm = client.post("/check/input", files={"file": ("live_input_normal.npy", f, "application/octet-stream")})

        assert res_norm.status_code == 200, f"Expected 200 OK, got {res_norm.status_code}: {res_norm.text}"
        json_norm = res_norm.json()

        print(f"      Method          : {json_norm['method']}")
        print(f"      Drift Score     : {json_norm['input_distribution_drift_score']}")
        print(f"      Status          : {json_norm['status']}")
        print(f"      Finding Title   : {json_norm['finding']['title']}")

        assert json_norm["status"] == "NORMAL", f"Normal batch should return NORMAL status, got {json_norm['status']}"
        assert json_norm["input_distribution_drift_score"] < 0.25, "Normal drift score should be < 0.25"

        # 2. Test POST /check/input on Shifted Live Batch
        print("\n[2/2] Testing POST /check/input on Shifted Live Batch...")
        with open("data/live_input_shifted.npy", "rb") as f:
            res_shift = client.post("/check/input", files={"file": ("live_input_shifted.npy", f, "application/octet-stream")})

        assert res_shift.status_code == 200, f"Expected 200 OK, got {res_shift.status_code}"
        json_shift = res_shift.json()

        print(f"      Method          : {json_shift['method']}")
        print(f"      Drift Score     : {json_shift['input_distribution_drift_score']}")
        print(f"      Status          : {json_shift['status']}")
        print(f"      Finding Title   : {json_shift['finding']['title']}")

        assert json_shift["status"] in ["ELEVATED", "HIGH"], f"Shifted batch should return ELEVATED or HIGH, got {json_shift['status']}"
        assert json_shift["input_distribution_drift_score"] >= 0.25, "Shifted drift score should be >= 0.25"

        # Save artifact JSON responses
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/drift_normal_response.json", "w", encoding="utf-8") as f:
            json.dump(json_norm, f, indent=2)
        with open("artifacts/drift_shifted_response.json", "w", encoding="utf-8") as f:
            json.dump(json_shift, f, indent=2)

        print("\n=======================================================")
        print("PHASE 6 DRIFT ENGINE VERIFICATION PASSED SUCCESSFULLY!")
        print("=======================================================")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FAILURE] Phase 6 Drift Engine Verification Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
