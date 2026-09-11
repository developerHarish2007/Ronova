from typing import List, Dict, Any, Tuple
import base64
import io
import hashlib
import imagehash
import numpy as np
from PIL import Image

from ronova.core.types import Finding, EvidenceStrength


def _generate_thumbnail_b64(img_arr: np.ndarray) -> str:
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


class DuplicateDetector:
    """
    RONOVA Dataset Duplicate & Near-Duplicate Detector:
    Uses perceptual hashing (dHash/pHash) combined with structural mean squared error (MSE)
    and exact MD5 byte hashing to identify duplicate and near-duplicate image samples.
    """

    def __init__(self, mse_threshold: float = 0.001, hash_threshold: int = 4):
        self.mse_threshold = mse_threshold
        self.hash_threshold = hash_threshold

    @staticmethod
    def _to_pil_image(img_arr: np.ndarray) -> Image.Image:
        arr = img_arr.squeeze()
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="L")

    def analyze(self, dataset: np.ndarray) -> Tuple[List[Dict[str, Any]], Finding]:
        num_samples = dataset.shape[0]
        hashes = []
        raw_hashes = []

        for i in range(num_samples):
            pil_img = self._to_pil_image(dataset[i])
            h_val = imagehash.dhash(pil_img, hash_size=8)
            hashes.append(h_val)
            raw_hashes.append(hashlib.md5(dataset[i].tobytes()).hexdigest())

        duplicate_pairs: List[Dict[str, Any]] = []

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                is_exact = bool(raw_hashes[i] == raw_hashes[j])
                dist = int(hashes[i] - hashes[j])
                mse = float(np.mean((dataset[i] - dataset[j]) ** 2))

                # Flag if exact MD5 match OR (dHash match AND low MSE pixel distance)
                if is_exact or (dist <= self.hash_threshold and mse <= self.mse_threshold):
                    duplicate_pairs.append({
                        "sample_idx_1": int(i),
                        "sample_idx_2": int(j),
                        "hamming_distance": int(dist),
                        "mse_distance": round(mse, 6),
                        "is_exact_match": bool(is_exact),
                        "thumbnail_1": _generate_thumbnail_b64(dataset[i]),
                        "thumbnail_2": _generate_thumbnail_b64(dataset[j]),
                    })

        num_dups = len(duplicate_pairs)

        if num_dups > 0:
            # Calculate proportion of unique samples contaminated by duplication
            dup_indices = set()
            for pair in duplicate_pairs:
                dup_indices.add(pair["sample_idx_1"])
                dup_indices.add(pair["sample_idx_2"])
            dup_ratio = len(dup_indices) / max(1, num_samples)

            # Weight severity by match precision (Exact MD5 = 1.0, Near dHash = 1.0 - dist/8)
            weighted_severity = sum(
                1.0 if p["is_exact_match"] else max(0.2, 1.0 - p["hamming_distance"] / 8.0)
                for p in duplicate_pairs
            )
            # Continuous calibrated risk formula
            risk = round(min(95.0, max(15.0, 30.0 + (dup_ratio * 40.0) + min(30.0, weighted_severity * 2.0))), 1)
            finding = Finding(
                detector_id="dataset_duplicate_detector",
                finding_type="DATASET_DUPLICATE_SAMPLES",
                risk_score=float(risk),
                evidence_strength=EvidenceStrength.MEDIUM,
                title="Dataset Duplicate / Near-Duplicate Samples Detected",
                explanation=(
                    f"Identified {num_dups} duplicate or near-duplicate sample pair(s) in dataset via perceptual hash and MSE analysis. "
                    f"Duplicate samples can lead to train/test data leakage, artificial performance inflation, "
                    f"or sample imbalance in training pipelines."
                ),
                limitations=(
                    "Perceptual hashing (dHash + MSE) identifies visual and structural similarities under modest transformations; "
                    "semantic duplicates under heavy non-linear geometric distortions may not trigger hash matches."
                ),
                evidence_details={
                    "total_samples": int(num_samples),
                    "duplicate_pairs_count": int(num_dups),
                    "duplicate_pairs": duplicate_pairs,
                    "mse_threshold": float(self.mse_threshold),
                    "hash_threshold": int(self.hash_threshold),
                },
                is_hard_gate=False,
            )
        else:
            finding = Finding(
                detector_id="dataset_duplicate_detector",
                finding_type="DATASET_DUPLICATE_CLEAN",
                risk_score=0.0,
                evidence_strength=EvidenceStrength.HIGH,
                title="No Dataset Duplicate Pairs Detected",
                explanation=f"Evaluated {num_samples} samples; no duplicate or near-duplicate image pairs were detected.",
                limitations="Perceptual hashing (dHash + MSE) evaluated at MSE <= 0.02 and exact byte hashes.",
                evidence_details={"total_samples": int(num_samples), "duplicate_pairs_count": 0},
                is_hard_gate=False,
            )

        return duplicate_pairs, finding
