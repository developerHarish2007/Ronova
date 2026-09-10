import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_clean_and_unapproved_clean_model_use_same_behavioral_protocol(client):
    """
    Proves requirement 1, 2, 9:
    Approved clean model and unapproved clean model MUST use the exact same behavioral dataset and evaluation protocol.
    Their STRIP mean_entropy scores must be identical.
    """
    clean_model_path = "models/clean_classifier.onnx"
    assert os.path.exists(clean_model_path)

    # 1. Scan approved clean model
    with open(clean_model_path, "rb") as f:
        res_approved = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
    assert res_approved.status_code == 200
    json_approved = res_approved.json()

    # 2. Scan same clean model under empty manifest (unapproved clean model)
    manifest_path = "manifests/approved_models.json"
    manifest_backup = "manifests/approved_models.json.bak"
    shutil.copyfile(manifest_path, manifest_backup)
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"approved_models": []}, f)

        with open(clean_model_path, "rb") as f:
            res_unapproved = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
    finally:
        shutil.move(manifest_backup, manifest_path)

    assert res_unapproved.status_code == 200
    json_unapproved = res_unapproved.json()

    # Verify manifest statuses differ
    assert json_approved["manifest_verification"]["manifest_matched"] is True
    assert json_unapproved["manifest_verification"]["manifest_matched"] is False

    # Verify behavioral dataset & protocol configuration are identical
    approved_proto = json_approved["strip_analysis"]["behavioral_eval_config"]
    unapproved_proto = json_unapproved["strip_analysis"]["behavioral_eval_config"]
    assert approved_proto == unapproved_proto
    assert approved_proto["protocol_id"] == "standard_trojan_eval_suite_v1"

    # Verify STRIP behavioral scores are clean and consistent across runs (stochastic overlay sampling)
    assert json_approved["strip_analysis"]["trojan_indicator_triggered"] is False
    assert json_unapproved["strip_analysis"]["trojan_indicator_triggered"] is False
    assert abs(json_approved["strip_analysis"]["mean_entropy"] - json_unapproved["strip_analysis"]["mean_entropy"]) < 0.05

    # Verify manifest mismatch remains visible on unapproved clean model
    hard_gates_unapproved = json_unapproved["policy"]["hard_gates_triggered"]
    assert "Unapproved Model Hash / Manifest Mismatch" in hard_gates_unapproved
    assert "High-Strength Trojan Indicator" not in hard_gates_unapproved
    assert json_unapproved["policy"]["verdict"] == "QUARANTINE"


def test_backdoored_and_clean_demo_verdicts(client):
    """
    Proves requirement 5, 10, 11:
    Backdoored demo model produces QUARANTINE (with BOTH hard gates).
    Clean demo model produces ACCEPT (with risk=2.5 and zero hard gates).
    """
    clean_model_path = "models/clean_classifier.onnx"
    backdoored_model_path = "models/backdoored_classifier.onnx"

    # Clean model scan -> ACCEPT
    with open(clean_model_path, "rb") as f:
        res_clean = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
    assert res_clean.status_code == 200
    j_clean = res_clean.json()
    assert j_clean["policy"]["verdict"] == "ACCEPT"
    assert j_clean["policy"]["risk_score"] == 2.5
    assert j_clean["policy"]["hard_gate_fired"] is False

    # Backdoored model scan -> QUARANTINE
    with open(backdoored_model_path, "rb") as f:
        res_bd = client.post("/scan/model", files={"file": ("backdoored_classifier.onnx", f, "application/octet-stream")})
    assert res_bd.status_code == 200
    j_bd = res_bd.json()
    assert j_bd["policy"]["verdict"] == "QUARANTINE"
    hard_gates_bd = j_bd["policy"]["hard_gates_triggered"]
    assert "Unapproved Model Hash / Manifest Mismatch" in hard_gates_bd
    assert "High-Strength Trojan Indicator" in hard_gates_bd


def test_provenance_includes_behavioral_eval_config(client):
    """
    Proves requirement 7:
    Selected behavioral dataset/configuration is present in provenance manifest and API response.
    """
    clean_model_path = "models/clean_classifier.onnx"
    with open(clean_model_path, "rb") as f:
        res = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()

    # API response
    assert "behavioral_eval_config" in data["strip_analysis"]
    assert data["strip_analysis"]["behavioral_eval_config"]["protocol_id"] == "standard_trojan_eval_suite_v1"

    # Provenance manifest config_hash inputs
    prov = data["provenance_manifest"]
    assert "config_hash" in prov
    assert len(prov["config_hash"]) == 64


if __name__ == "__main__":
    pytest.main([__file__])
