import os
import sys
import tempfile
from pathlib import Path
import pytest
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.safety.gate import SafetyGate
from ronova.api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def safety_gate():
    return SafetyGate()


@pytest.fixture
def client():
    return TestClient(app)


def test_valid_clean_onnx_model(safety_gate):
    clean_model_path = "models/clean_classifier.onnx"
    assert os.path.exists(clean_model_path)

    valid, status, details = safety_gate.validate_onnx_artifact(clean_model_path)
    assert valid is True
    assert status == "VALID_ONNX_MODEL"
    assert details["num_inputs"] >= 1
    assert details["num_outputs"] >= 1
    assert len(details["inputs"]) >= 1
    assert len(details["outputs"]) >= 1

    res = safety_gate.scan_file(clean_model_path)
    assert res.is_onnx_valid is True
    assert res.onnx_validation_details["status"] == "VALID_ONNX_MODEL"


def test_valid_backdoored_onnx_model(safety_gate):
    backdoored_model_path = "models/backdoored_classifier.onnx"
    assert os.path.exists(backdoored_model_path)

    valid, status, details = safety_gate.validate_onnx_artifact(backdoored_model_path)
    assert valid is True
    assert status == "VALID_ONNX_MODEL"
    assert details["num_inputs"] >= 1
    assert details["num_outputs"] >= 1


def test_renamed_text_file(safety_gate):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(b"This is just a plain text file renamed with .onnx extension\n")
        fake_onnx_path = tmp.name

    try:
        valid, status, details = safety_gate.validate_onnx_artifact(fake_onnx_path)
        assert valid is False
        assert status in ["PROTOBUF_PARSE_ERROR", "MALFORMED_GRAPH"]

        res = safety_gate.scan_file(fake_onnx_path)
        assert res.is_onnx_valid is False
        assert res.hard_gate_triggered is True
        finding_types = [f.finding_type for f in res.gate_findings]
        assert "INVALID_FILE_FORMAT" in finding_types
    finally:
        if os.path.exists(fake_onnx_path):
            os.remove(fake_onnx_path)


def test_empty_file(safety_gate):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        empty_path = tmp.name

    try:
        valid, status, details = safety_gate.validate_onnx_artifact(empty_path)
        assert valid is False
        assert status == "EMPTY_FILE"

        res = safety_gate.scan_file(empty_path)
        assert res.is_onnx_valid is False
        assert res.hard_gate_triggered is True
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)


def test_truncated_onnx_file(safety_gate):
    clean_model_path = "models/clean_classifier.onnx"
    with open(clean_model_path, "rb") as f:
        clean_bytes = f.read(100) # Only first 100 bytes

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(clean_bytes)
        trunc_path = tmp.name

    try:
        valid, status, details = safety_gate.validate_onnx_artifact(trunc_path)
        assert valid is False
        assert status in ["PROTOBUF_PARSE_ERROR", "MALFORMED_GRAPH"]

        res = safety_gate.scan_file(trunc_path)
        assert res.is_onnx_valid is False
        assert res.hard_gate_triggered is True
    finally:
        if os.path.exists(trunc_path):
            os.remove(trunc_path)


def test_malformed_onnx_graph_no_inputs(safety_gate):
    # Construct an ONNX model graph with 0 inputs
    node = helper.make_node("Constant", inputs=[], outputs=["out"], value=helper.make_tensor("val", TensorProto.FLOAT, [1], [1.0]))
    graph = helper.make_graph([node], "no_inputs_graph", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, producer_name="test")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(model, tmp.name)
        no_inputs_path = tmp.name

    try:
        valid, status, details = safety_gate.validate_onnx_artifact(no_inputs_path)
        assert valid is False
        assert status == "NO_INPUTS"
    finally:
        if os.path.exists(no_inputs_path):
            os.remove(no_inputs_path)


def test_model_onnx_runtime_cannot_load(safety_gate):
    # Construct ONNX model with non-existent / invalid operator
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("NonExistentCustomOpXYZ", inputs=["in"], outputs=["out"], domain="invalid.domain")
    graph = helper.make_graph([node], "invalid_op_graph", [inp], [out])
    model = helper.make_model(graph, producer_name="test")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(model, tmp.name)
        invalid_op_path = tmp.name

    try:
        valid, status, details = safety_gate.validate_onnx_artifact(invalid_op_path)
        assert valid is False
        assert status in ["MALFORMED_GRAPH", "ONNX_RUNTIME_LOAD_ERROR"]
    finally:
        if os.path.exists(invalid_op_path):
            os.remove(invalid_op_path)


def test_api_invalid_file_format_hard_gate(client):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(b"CORRUPTED_NON_ONNX_DATA_BLOB")
        corrupted_path = tmp.name

    try:
        with open(corrupted_path, "rb") as f:
            res = client.post("/scan/model", files={"file": ("fake_model.onnx", f, "application/octet-stream")})

        assert res.status_code == 200
        data = res.json()
        assert data["policy"]["verdict"] == "QUARANTINE"
        assert data["policy"]["hard_gate_fired"] is True
        assert "Invalid Artifact Format" in data["policy"]["hard_gates_triggered"]
        assert data["manifest_verification"]["manifest_matched"] is False
    finally:
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)


if __name__ == "__main__":
    pytest.main([__file__])
