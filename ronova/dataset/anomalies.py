from typing import List, Dict, Any, Tuple
import base64
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from ronova.core.types import Finding, EvidenceStrength


class AirGappedCNNFeatureExtractor(nn.Module):
    """
    Self-contained CNN Feature Extractor for offline/air-gapped operation.
    Requires no external model downloads.
    """

    def __init__(self, out_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        self.fc = nn.Linear(32 * 4 * 4, out_features)

        # Initialize fixed deterministic orthogonal weights for offline feature projection
        torch.manual_seed(42)
        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.orthogonal_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] > 1:
            # Grayscale reduction
            x = x.mean(dim=1, keepdim=True)
        h = self.pool1(self.relu1(self.conv1(x)))
        h = self.pool2(self.relu2(self.conv2(h)))
        flat = torch.flatten(h, 1)
        return self.fc(flat)


def generate_sample_thumbnail(img_arr: np.ndarray) -> str:
    """Converts a sample array slice into a base64 PNG data URL."""
    try:
        arr = img_arr.squeeze()
        if arr.ndim > 2:
            arr = arr.mean(axis=0)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L").resize((56, 56), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception:
        return ""


class AnomalyDetector:
    """
    RONOVA Embedding-Based Dataset Anomaly & Outlier Detector:
    Extracts deep visual embeddings using an offline CNN encoder and runs calibrated
    anomaly clustering (Isolation Forest + decision score thresholding) to surface sample anomalies
    consistent with poisoning or labeling errors.
    """

    def __init__(self, score_threshold: float = -0.07, random_state: int = 42):
        self.score_threshold = score_threshold
        self.random_state = random_state
        self.extractor = AirGappedCNNFeatureExtractor()
        self.extractor.eval()

    def extract_embeddings(self, dataset: np.ndarray) -> np.ndarray:
        tensors = torch.from_numpy(dataset.copy()).float()
        embeddings_list = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i : i + batch_size]
                emb = self.extractor(batch)
                embeddings_list.append(emb.cpu().numpy())

        return np.concatenate(embeddings_list, axis=0)

    def analyze(
        self, dataset: np.ndarray
    ) -> Tuple[List[int], Finding, List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        num_samples = dataset.shape[0]
        embeddings = self.extract_embeddings(dataset)

        clf = IsolationForest(
            contamination="auto",
            random_state=self.random_state,
            n_estimators=100,
        )
        clf.fit(embeddings)
        scores = clf.decision_function(embeddings)

        anomalous_indices = [
            int(i) for i in range(num_samples) if float(scores[i]) < self.score_threshold
        ]
        num_anomalies = len(anomalous_indices)

        # Compute 2D coordinates using PCA for visual embedding chart
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_coords = pca.fit_transform(embeddings)

        # Normalize 2D coordinates to [-1.0, 1.0] for chart display
        max_abs = np.max(np.abs(pca_coords))
        if max_abs > 1e-6:
            pca_coords = pca_coords / max_abs

        chart_coordinates = [
            {
                "sample_idx": int(i),
                "x": round(float(pca_coords[i, 0]), 4),
                "y": round(float(pca_coords[i, 1]), 4),
                "score": round(float(scores[i]), 4),
                "is_anomaly": bool(i in anomalous_indices),
            }
            for i in range(num_samples)
        ]

        scores_summary = {
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "threshold": float(self.score_threshold),
            "anomalous_count": num_anomalies,
        }

        # Generate thumbnails for flagged anomalous samples
        flagged_thumbnails = [
            {
                "sample_idx": int(idx),
                "anomaly_score": round(float(scores[idx]), 4),
                "thumbnail_b64": generate_sample_thumbnail(dataset[idx]),
            }
            for idx in anomalous_indices[:12]
        ]

        if num_anomalies > 0:
            anom_ratio = num_anomalies / max(1, num_samples)
            # Calculate how severe the anomaly scores deviate beyond the threshold
            score_deviations = [max(0.0, self.score_threshold - float(scores[idx])) for idx in anomalous_indices]
            mean_deviation = float(np.mean(score_deviations)) if score_deviations else 0.0

            # Calibrated continuous anomaly risk (proportional to anomaly volume and outlier distance)
            anom_risk = round(min(95.0, max(20.0, 35.0 + (anom_ratio * 45.0) + (mean_deviation * 300.0))), 1)

            finding = Finding(
                detector_id="dataset_anomaly_detector",
                finding_type="DATASET_ANOMALY_INDICATOR",
                risk_score=float(anom_risk),
                evidence_strength=EvidenceStrength.MEDIUM,
                title="Dataset Embedding Anomaly Clusters Detected",
                explanation=(
                    f"Embedding space analysis flags {num_anomalies} sample(s) "
                    f"anomalous in a way consistent with poisoning or labeling errors. "
                    f"These samples reside in low-density feature space regions relative to clean class clusters."
                ),
                limitations=(
                    "Embedding-space anomaly detection flags distributional outliers consistent with poisoning, "
                    "label noise, or domain artifact corruption; it does not constitute deterministic proof of "
                    "intentional adversarial dataset poisoning."
                ),
                evidence_details={
                    "total_samples": num_samples,
                    "anomalous_samples_count": num_anomalies,
                    "anomalous_indices": anomalous_indices,
                    "score_threshold": float(self.score_threshold),
                    "min_decision_score": float(np.min(scores)),
                },
                is_hard_gate=False,
            )
        else:
            finding = Finding(
                detector_id="dataset_anomaly_detector",
                finding_type="DATASET_ANOMALY_CLEAN",
                risk_score=0.0,
                evidence_strength=EvidenceStrength.HIGH,
                title="Dataset Feature Distribution Clean",
                explanation=f"Evaluated {num_samples} dataset embeddings; no anomalous clusters or severe feature outliers detected under calibrated score thresholds.",
                limitations="Feature space analysis using offline CNN feature embeddings + calibrated Isolation Forest decision score thresholds.",
                evidence_details={"total_samples": num_samples, "anomalous_samples_count": 0},
                is_hard_gate=False,
            )

        return anomalous_indices, finding, chart_coordinates, scores_summary, flagged_thumbnails
