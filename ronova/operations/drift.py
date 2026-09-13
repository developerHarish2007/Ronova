from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp

from ronova.core.types import Finding, EvidenceStrength


class ReferenceFeatureExtractor(nn.Module):
    """
    Fixed, stable reference feature extractor for operations assurance.
    Must not be the model under assessment.
    Converts 28x28 single-channel images into 64-dim embedding vectors.
    """

    def __init__(self, out_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, out_features)

        # Initialize fixed deterministic orthogonal weights
        torch.manual_seed(99)
        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.orthogonal_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        h = self.pool1(self.relu1(self.conv1(x)))
        h = self.pool2(self.relu2(self.conv2(h)))
        flat = torch.flatten(h, 1)
        return self.fc(flat)


class InputDriftDetector:
    """
    RONOVA Operations Assurance — Input Distribution Drift Detector:
    Extracts feature embeddings from reference and live input batches using a separate,
    stable reference feature extractor. Computes 2-sample Kolmogorov-Smirnov (KS) test
    across embedding dimensions to detect Input Distribution Drift.
    """

    def __init__(self, elevated_threshold: float = 0.25, high_threshold: float = 0.50):
        self.elevated_threshold = elevated_threshold
        self.high_threshold = high_threshold
        self.extractor = ReferenceFeatureExtractor()
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

    def evaluate_drift(
        self, reference_dataset: np.ndarray, live_dataset: np.ndarray
    ) -> Tuple[float, str, Finding, Dict[str, Any]]:
        ref_embeddings = self.extract_embeddings(reference_dataset)
        live_embeddings = self.extract_embeddings(live_dataset)

        num_dims = ref_embeddings.shape[1]
        ks_stats = []

        for d in range(num_dims):
            stat = ks_2samp(ref_embeddings[:, d], live_embeddings[:, d]).statistic
            ks_stats.append(float(stat))

        drift_score = float(np.mean(ks_stats))

        if drift_score >= self.high_threshold:
            status = "HIGH"
            risk = 70.0
            strength = EvidenceStrength.HIGH
        elif drift_score >= self.elevated_threshold:
            status = "ELEVATED"
            risk = 40.0
            strength = EvidenceStrength.MEDIUM
        else:
            status = "NORMAL"
            risk = 0.0
            strength = EvidenceStrength.HIGH

        finding = Finding(
            detector_id="operations_input_drift",
            finding_type=f"INPUT_DISTRIBUTION_DRIFT_{status}",
            risk_score=risk,
            evidence_strength=strength,
            title="distribution-shift / input-anomaly indicator",
            explanation=(
                f"Evaluated statistical distribution shift between reference baseline embeddings and live batch embeddings "
                f"via 2-sample Kolmogorov-Smirnov test (Drift Score: {drift_score:.4f}, Method: KS, Status: {status})."
            ),
            limitations=(
                "This channel flags inputs that deviate from the reference embedding distribution. "
                "It is not designed to catch adversarial examples specifically optimized to evade detection — that is a stated limitation."
            ),
            evidence_details={
                "drift_score": round(drift_score, 4),
                "method": "KS",
                "status": status,
                "num_reference_samples": len(reference_dataset),
                "num_live_samples": len(live_dataset),
                "ks_stats_summary": {
                    "mean_ks": round(float(np.mean(ks_stats)), 4),
                    "max_ks": round(float(np.max(ks_stats)), 4),
                    "min_ks": round(float(np.min(ks_stats)), 4),
                },
            },
            is_hard_gate=False, # Per Section 10, drift is ALWAYS a Level 2 soft finding, never a hard gate.
        )

        details = {
            "input_distribution_drift_score": round(drift_score, 4),
            "method": "KS",
            "status": status,
            "ref_means": [round(float(m), 4) for m in np.mean(ref_embeddings, axis=0)[:10]],
            "live_means": [round(float(m), 4) for m in np.mean(live_embeddings, axis=0)[:10]],
            "finding": finding,
        }

        return round(drift_score, 4), status, finding, details
