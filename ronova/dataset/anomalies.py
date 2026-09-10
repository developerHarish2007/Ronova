from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

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
        self.pool1 = nn.MaxPool2d(2, 2) # 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4)) # 4x4
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


class AnomalyDetector:
    """
    RONOVA Embedding-Based Dataset Anomaly & Outlier Detector:
    Extracts deep visual embeddings using an offline CNN encoder and runs unsupervised
    anomaly clustering (Isolation Forest) to surface sample anomalies consistent
    with poisoning or labeling errors.
    """

    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.contamination = contamination
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

    def analyze(self, dataset: np.ndarray) -> Tuple[List[int], Finding]:
        num_samples = dataset.shape[0]
        embeddings = self.extract_embeddings(dataset)

        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        preds = clf.fit_predict(embeddings) # -1 for anomaly, 1 for normal
        scores = clf.decision_function(embeddings)

        anomalous_indices = [int(i) for i in range(num_samples) if preds[i] == -1]
        num_anomalies = len(anomalous_indices)

        if num_anomalies > 0:
            # Wording discipline strictly enforced per Section 17
            finding = Finding(
                detector_id="dataset_anomaly_detector",
                finding_type="DATASET_ANOMALY_INDICATOR",
                risk_score=min(45.0 + num_anomalies * 5.0, 85.0),
                evidence_strength=EvidenceStrength.MEDIUM,
                title="Dataset Embedding Anomaly Clusters Detected",
                explanation=(
                    f"Embedding space clustering flags {num_anomalies} sample(s) "
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
                    "contamination_threshold": self.contamination,
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
                explanation=f"Evaluated {num_samples} dataset embeddings; no anomalous clusters or severe feature outliers detected.",
                limitations="Feature space analysis using offline CNN feature embeddings + Isolation Forest clustering.",
                evidence_details={"total_samples": num_samples, "anomalous_samples_count": 0},
                is_hard_gate=False,
            )

        return anomalous_indices, finding
