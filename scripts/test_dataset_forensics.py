import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.dataset.anomalies import AnomalyDetector
from ronova.dataset.duplicates import DuplicateDetector
from ronova.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_clean_dataset_passes_accept():
    """Verifies clean_dataset.npy receives ACCEPT with 0 duplicates and 0 anomalies."""
    clean_ds = np.load("data/clean_dataset.npy")
    
    dup_detector = DuplicateDetector()
    dup_pairs, dup_finding = dup_detector.analyze(clean_ds)
    assert len(dup_pairs) == 0
    assert dup_finding.finding_type == "DATASET_DUPLICATE_CLEAN"

    anom_detector = AnomalyDetector(score_threshold=-0.07)
    anom_indices, anom_finding, coords, scores_sum, thumbs = anom_detector.analyze(clean_ds)
    assert len(anom_indices) == 0
    assert anom_finding.finding_type == "DATASET_ANOMALY_CLEAN"

    # API Endpoint Check
    with open("data/clean_dataset.npy", "rb") as f:
        res = client.post("/scan/dataset?role=training_eval", files={"file": ("clean_dataset.npy", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_risk_assessment"] == "ACCEPT"
    assert data["anomalous_samples_count"] == 0
    assert data["duplicate_pairs_count"] == 0
    assert data["dataset_role"] == "training_eval"


def test_poisoned_dataset_triggers_review():
    """Verifies poisoned_dataset.npy produces REVIEW with flagged duplicates & calibrated anomalies."""
    with open("data/poisoned_dataset.npy", "rb") as f:
        res = client.post("/scan/dataset?role=training_eval", files={"file": ("poisoned_dataset.npy", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_risk_assessment"] == "REVIEW"
    assert data["anomalous_samples_count"] > 0
    assert data["duplicate_pairs_count"] > 0
    assert 90 in data["anomalous_indices"]
    assert len(data["chart_coordinates"]) == 100
    assert len(data["flagged_thumbnails"]) > 0


def test_clean_overlays_with_overlay_role():
    """Verifies clean_overlays.npy under perturbation_overlay role receives ACCEPT and overlay protocol."""
    with open("data/clean_overlays.npy", "rb") as f:
        res = client.post("/scan/dataset?role=perturbation_overlay", files={"file": ("clean_overlays.npy", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_risk_assessment"] == "ACCEPT"
    assert data["dataset_role"] == "perturbation_overlay"
    assert "STRIP Perturbation Overlay" in data["analysis_protocol"]


def test_clean_data_not_forced_to_review():
    """Verifies calibrated anomaly detector does not force fixed 8% contamination false positives."""
    clean_ds = np.load("data/clean_dataset.npy")
    anom_detector = AnomalyDetector(score_threshold=-0.07)
    anom_indices, finding, coords, scores_sum, thumbs = anom_detector.analyze(clean_ds)
    assert len(anom_indices) == 0, "Calibrated detector must not force false anomalies on clean dataset"


def test_unsupported_shape_returns_unguaranteed(tmp_path):
    """Verifies 1D / invalid shape dataset returns UNGUARANTEED status."""
    invalid_file = tmp_path / "invalid_1d.npy"
    np.save(invalid_file, np.array([1.0, 2.0, 3.0]))

    with open(invalid_file, "rb") as f:
        res = client.post("/scan/dataset?role=training_eval", files={"file": ("invalid_1d.npy", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_risk_assessment"] == "UNGUARANTEED"
    assert data["findings"][0]["finding_type"] == "UNGUARANTEED_UNSUPPORTED_SHAPE"


def test_duplicate_and_anomaly_separation():
    """Verifies duplicate findings and anomaly findings remain distinct."""
    poisoned_ds = np.load("data/poisoned_dataset.npy")
    dup_detector = DuplicateDetector()
    dup_pairs, dup_finding = dup_detector.analyze(poisoned_ds)

    anom_detector = AnomalyDetector(score_threshold=-0.07)
    anom_indices, anom_finding, coords, scores_sum, thumbs = anom_detector.analyze(poisoned_ds)

    assert dup_finding.detector_id == "dataset_duplicate_detector"
    assert anom_finding.detector_id == "dataset_anomaly_detector"
    assert dup_finding.finding_type == "DATASET_DUPLICATE_SAMPLES"
    assert anom_finding.finding_type == "DATASET_ANOMALY_INDICATOR"


def test_honest_limitations_wording():
    """Verifies honest wording limitations text is returned in response."""
    with open("data/clean_dataset.npy", "rb") as f:
        res = client.post("/scan/dataset?role=training_eval", files={"file": ("clean_dataset.npy", f, "application/octet-stream")})
    data = res.json()
    assert "do not constitute deterministic proof" in data["limitations_disclaimer"]
