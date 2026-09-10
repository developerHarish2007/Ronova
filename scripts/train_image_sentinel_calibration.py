"""
RONOVA Image Sentinel Offline Calibration Script:
Computes reference distribution features and anomaly detection thresholds from clean reference dataset.
Saves calibration artifact to artifacts/image_sentinel_calibration.json.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.sandbox.runner import IsolatedSandboxRunner


def train_calibration(clean_dataset_path: str = "data/clean_dataset.npy", model_path: str = "models/clean_classifier.onnx") -> dict:
    if not os.path.exists(clean_dataset_path):
        clean_dataset_path = "data/test_samples_clean.npy"

    if not os.path.exists(clean_dataset_path):
        raise FileNotFoundError(f"Clean reference dataset not found at {clean_dataset_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Clean reference model not found at {model_path}")

    dataset = np.load(clean_dataset_path)
    dataset_sha256 = hashlib.sha256(dataset.tobytes()).hexdigest()
    with open(model_path, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Run inference via sandbox runner to get baseline predictions
    runner = IsolatedSandboxRunner(timeout_seconds=15.0)
    probs, meta = runner.run_inference(model_path, dataset)

    # Compute baseline reference distribution statistics
    ref_means = np.mean(dataset.reshape(len(dataset), -1), axis=0).tolist()
    ref_stds = np.std(dataset.reshape(len(dataset), -1), axis=0).tolist()
    pred_means = np.mean(probs, axis=0).tolist()
    pred_entropies = [-float(np.sum(p * np.log(p + 1e-12))) for p in probs]
    mean_entropy = float(np.mean(pred_entropies))
    std_entropy = float(np.std(pred_entropies))

    # Anomaly threshold: 3 std above mean
    anomaly_entropy_threshold = round(mean_entropy + 3 * std_entropy + 0.1, 4)

    calibration_dict = {
        "framework": "RONOVA Image Sentinel Calibration v1",
        "model_path": model_path,
        "model_sha256": model_sha256,
        "reference_dataset_path": clean_dataset_path,
        "reference_dataset_sha256": dataset_sha256,
        "num_samples": len(dataset),
        "mean_entropy": round(mean_entropy, 4),
        "std_entropy": round(std_entropy, 4),
        "anomaly_entropy_threshold": anomaly_entropy_threshold,
        "max_prediction_distribution_distance": 1.5,
        "max_l2_feature_distance": 1.5,  # legacy alias
        "max_flip_rate_threshold": 0.25,
        "max_js_divergence_threshold": 0.15,
        "pred_class_distribution": pred_means,
    }

    output_path = "artifacts/image_sentinel_calibration.json"
    os.makedirs("artifacts", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration_dict, f, indent=2)

    print(f"Image Sentinel calibration saved to -> {output_path}")
    return calibration_dict


if __name__ == "__main__":
    train_calibration()
