import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.sandbox.runner import IsolatedSandboxRunner


def test_successful_inference():
    runner = IsolatedSandboxRunner(timeout_seconds=10.0, max_batch_size=128)
    clean_model = "models/clean_classifier.onnx"
    assert os.path.exists(clean_model), f"{clean_model} must exist for tests"

    input_array = np.random.randn(2, 1, 28, 28).astype(np.float32)
    probs, meta = runner.run_inference(clean_model, input_array)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 10)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)

    assert meta["sandbox_mode"] in ["subprocess_isolated", "docker_container"]
    assert meta["process_isolated"] is True
    assert meta["input_shape"] == [2, 1, 28, 28]
    assert meta["output_shape"] == [2, 10]
    assert meta["providers_used"] == ["CPUExecutionProvider"]
    assert isinstance(meta["execution_time_seconds"], float)
    assert meta["timeout_triggered"] is False


def test_timeout_handling():
    # Extremely small timeout to force TimeoutError
    runner = IsolatedSandboxRunner(timeout_seconds=0.0001, max_batch_size=128)
    clean_model = "models/clean_classifier.onnx"
    input_array = np.random.randn(20, 1, 28, 28).astype(np.float32)

    with pytest.raises(TimeoutError) as exc_info:
        runner.run_inference(clean_model, input_array)

    assert "timed out" in str(exc_info.value).lower()


def test_oversized_batch_rejection():
    runner = IsolatedSandboxRunner(timeout_seconds=10.0, max_batch_size=5)
    clean_model = "models/clean_classifier.onnx"
    input_array = np.random.randn(10, 1, 28, 28).astype(np.float32)

    with pytest.raises(ValueError) as exc_info:
        runner.run_inference(clean_model, input_array)

    assert "exceeds maximum sandbox limit" in str(exc_info.value)


def test_invalid_model_failure():
    runner = IsolatedSandboxRunner(timeout_seconds=10.0, max_batch_size=128)
    input_array = np.random.randn(1, 1, 28, 28).astype(np.float32)

    # 1. Non-existent model path
    with pytest.raises(FileNotFoundError):
        runner.run_inference("models/does_not_exist.onnx", input_array)

    # 2. Corrupted model file
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_ONNX_MODEL_HEADER")
        invalid_model_path = tmp.name

    try:
        with pytest.raises(RuntimeError) as exc_info:
            runner.run_inference(invalid_model_path, input_array)
        assert "Sandbox worker execution failed" in str(exc_info.value)
    finally:
        if os.path.exists(invalid_model_path):
            os.remove(invalid_model_path)


def test_metadata_correctness():
    runner = IsolatedSandboxRunner(timeout_seconds=15.0, max_batch_size=64)
    clean_model = "models/clean_classifier.onnx"
    input_array = np.zeros((1, 1, 28, 28), dtype=np.float32)

    probs, meta = runner.run_inference(clean_model, input_array)

    assert meta["sandbox_mode"] in ["subprocess_isolated", "docker_container"]
    assert meta["process_isolated"] is True
    assert meta["timeout_limit_seconds"] == 15.0
    assert meta["timeout_triggered"] is False
    assert meta["input_shape"] == [1, 1, 28, 28]
    assert meta["output_shape"] == [1, 10]
    assert meta["providers_used"] == ["CPUExecutionProvider"]
    assert "onnx_input_name" in meta
    assert "onnx_output_name" in meta


if __name__ == "__main__":
    pytest.main([__file__])
