"""
Provenance Reproducibility and Assurance-Engine Identity Tests for RONOVA (Seventh Fix).
Tests all 6 required scenarios for 8-field provenance manifest generation.
"""

import os
import hashlib
import numpy as np
import pytest

from ronova.provenance.engine import ProvenanceEngine


@pytest.fixture
def engine():
    """Provides ProvenanceEngine instance."""
    return ProvenanceEngine()


def test_identical_inputs_produce_identical_provenance(engine):
    """1. Test that identical inputs/configuration produce identical 8-field provenance."""
    model_sha = "a" * 64
    dataset = np.ones((10, 1, 28, 28), dtype=np.float32)
    config = {"num_perturbations": 20, "blend_alpha": 0.5, "protocol": "eval_v1"}
    prep_info = "Resize(28,28)"
    inp = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = np.array([0.9, 0.1], dtype=np.float32)

    m1 = engine.compute_provenance_manifest(model_sha, dataset, config, prep_info, inp, out)
    m2 = engine.compute_provenance_manifest(model_sha, dataset, config, prep_info, inp, out)

    assert m1 == m2
    assert len(m1) == 8
    assert all(k in m1 for k in [
        "dataset_hash", "model_hash", "config_hash", "preprocess_hash",
        "runtime_hash", "input_hash", "output_hash", "assurance_engine_hash"
    ])


def test_changing_configuration_changes_config_hash(engine):
    """2. Test that changing configuration parameters changes config_hash."""
    model_sha = "a" * 64
    dataset = np.ones((10, 1, 28, 28), dtype=np.float32)
    config1 = {"num_perturbations": 20, "blend_alpha": 0.5}
    config2 = {"num_perturbations": 30, "blend_alpha": 0.5}  # modified param
    prep_info = "Resize(28,28)"
    inp = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = np.array([0.9, 0.1], dtype=np.float32)

    m1 = engine.compute_provenance_manifest(model_sha, dataset, config1, prep_info, inp, out)
    m2 = engine.compute_provenance_manifest(model_sha, dataset, config2, prep_info, inp, out)

    assert m1["config_hash"] != m2["config_hash"]
    # Other hashes remain equal
    assert m1["dataset_hash"] == m2["dataset_hash"]
    assert m1["model_hash"] == m2["model_hash"]


def test_changing_behavioral_dataset_changes_dataset_hash(engine):
    """3. Test that changing the dataset array changes dataset_hash."""
    model_sha = "a" * 64
    dataset1 = np.ones((10, 1, 28, 28), dtype=np.float32)
    dataset2 = np.zeros((10, 1, 28, 28), dtype=np.float32)  # modified dataset
    config = {"num_perturbations": 20}
    prep_info = "Resize(28,28)"
    inp = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = np.array([0.9, 0.1], dtype=np.float32)

    m1 = engine.compute_provenance_manifest(model_sha, dataset1, config, prep_info, inp, out)
    m2 = engine.compute_provenance_manifest(model_sha, dataset2, config, prep_info, inp, out)

    assert m1["dataset_hash"] != m2["dataset_hash"]


def test_changing_runtime_metadata_changes_runtime_hash(engine):
    """4. Test that changing runtime environment metadata changes runtime_hash."""
    model_sha = "a" * 64
    dataset = np.ones((10, 1, 28, 28), dtype=np.float32)
    config = {"num_perturbations": 20}
    prep_info = "Resize(28,28)"
    inp = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out = np.array([0.9, 0.1], dtype=np.float32)

    rt1 = {"python_version": "3.11.0", "onnxruntime_version": "1.17.0"}
    rt2 = {"python_version": "3.12.0", "onnxruntime_version": "1.18.0"}

    m1 = engine.compute_provenance_manifest(model_sha, dataset, config, prep_info, inp, out, custom_runtime_info=rt1)
    m2 = engine.compute_provenance_manifest(model_sha, dataset, config, prep_info, inp, out, custom_runtime_info=rt2)

    assert m1["runtime_hash"] != m2["runtime_hash"]


def test_assurance_engine_identity_is_not_old_placeholder():
    """5. Test that assurance_engine_hash is NOT derived from the old placeholder commit SHA."""
    identity, source_type = ProvenanceEngine.get_assurance_engine_identity()
    old_placeholder = ProvenanceEngine.OLD_PLACEHOLDER_COMMIT
    old_placeholder_hash = hashlib.sha256(old_placeholder.encode("utf-8")).hexdigest()

    engine = ProvenanceEngine()
    manifest = engine.compute_provenance_manifest(
        "a" * 64,
        np.ones((1, 1), dtype=np.float32),
        {},
        "",
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
    )

    assert identity != old_placeholder
    assert manifest["assurance_engine_hash"] != old_placeholder_hash
    assert source_type in ("ENV_VAR", "GIT_COMMIT", "FALLBACK_RELEASE_ID")


def test_missing_git_metadata_reported_honestly(monkeypatch):
    """6. Test that missing Git metadata is reported honestly without claiming a fake commit."""
    # Unset env vars
    monkeypatch.delenv("RONOVA_RELEASE_COMMIT", raising=False)
    monkeypatch.delenv("RONOVA_BUILD_HASH", raising=False)

    # Mock subprocess.run to raise FileNotFoundError or return error
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr("subprocess.run", mock_run)

    identity, source_type = ProvenanceEngine.get_assurance_engine_identity()
    assert identity == ProvenanceEngine.DEFAULT_FALLBACK_RELEASE_ID
    assert source_type == "FALLBACK_RELEASE_ID"
