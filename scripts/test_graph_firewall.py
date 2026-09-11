import os
import sys
import shutil
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto
from fastapi.testclient import TestClient

from ronova.safety.graph_firewall import ONNXGraphFirewall, STANDARD_SAFE_OPERATORS, APPROVED_DOMAINS
from ronova.safety.gate import SafetyGate
from ronova.engine.policy import PolicyEngine
from ronova.api.main import app

def create_mock_onnx_model(op_type="Relu", domain="", input_shape=(1, 1, 28, 28), output_shape=(1, 10)) -> onnx.ModelProto:
    """Helper to create a synthetic ONNX model for unit testing."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_shape))
    
    node_kwargs = {}
    if domain:
        node_kwargs["domain"] = domain

    # Simple single-op or multi-op node
    opset_imports = [helper.make_operatorsetid("", 13)]
    if domain:
        opset_imports.append(helper.make_operatorsetid(domain, 1))

    if op_type == "Relu":
        node = helper.make_node("Relu", ["input"], ["output"], **node_kwargs)
    elif op_type == "RandomNormal":
        node = helper.make_node("RandomNormal", [], ["output"], shape=list(output_shape))
    elif op_type == "CustomMaliciousOp":
        # Custom domain for custom op
        if not domain:
            domain = "custom.malicious.domain"
            node_kwargs["domain"] = domain
            opset_imports.append(helper.make_operatorsetid(domain, 1))
        node = helper.make_node(op_type, ["input"], ["output"], **node_kwargs)
    elif op_type == "CustomDomainOp":
        domain = "custom.unapproved.domain"
        node_kwargs["domain"] = domain
        opset_imports.append(helper.make_operatorsetid(domain, 1))
        node = helper.make_node("Relu", ["input"], ["output"], **node_kwargs)
    else:
        node = helper.make_node(op_type, ["input"], ["output"], **node_kwargs)

    graph = helper.make_graph([node], "test_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=opset_imports, producer_name="ronova-test-firewall")
    return model


def test_clean_model_passes():
    print("--> Testing clean classifier model passes firewall...")
    firewall = ONNXGraphFirewall()
    res = firewall.scan_file("models/clean_classifier.onnx")
    assert res.passed is True, f"Clean model failed firewall: {res.violations}"
    assert len(res.disallowed_operators) == 0
    assert len(res.checked_operators) > 0
    assert res.total_input_elements > 0
    print(f"  [PASS] Clean model passed with operators: {res.checked_operators}")


def test_backdoored_model_passes():
    print("--> Testing backdoored classifier model passes structural firewall...")
    firewall = ONNXGraphFirewall()
    res = firewall.scan_file("models/backdoored_classifier.onnx")
    assert res.passed is True, f"Backdoored model failed structural firewall: {res.violations}"
    assert len(res.disallowed_operators) == 0
    assert len(res.checked_operators) > 0
    assert res.total_input_elements > 0
    print(f"  [PASS] Backdoored demo model passed structural firewall with operators: {res.checked_operators}")


def test_unapproved_operator_rejected():
    print("--> Testing unapproved operator is rejected by firewall...")
    model = create_mock_onnx_model(op_type="CustomMaliciousOp")
    firewall = ONNXGraphFirewall()
    res = firewall.inspect_graph(model)
    assert res.passed is False, "Custom operator was not blocked!"
    assert "CustomMaliciousOp" in res.disallowed_operators
    assert any("CustomMaliciousOp" in v for v in res.violations)
    print(f"  [PASS] Blocked unapproved operator as expected: {res.violations}")


def test_unapproved_domain_rejected():
    print("--> Testing unapproved domain is rejected by firewall...")
    model = create_mock_onnx_model(op_type="CustomDomainOp")
    firewall = ONNXGraphFirewall()
    res = firewall.inspect_graph(model)
    assert res.passed is False, "Custom domain was not blocked!"
    assert any("custom.unapproved.domain" in v for v in res.violations)
    print(f"  [PASS] Blocked unapproved domain as expected: {res.violations}")


def test_oversized_input_rejected():
    print("--> Testing oversized input tensor is rejected...")
    # Create model with huge input tensor (e.g. 50,000,000 elements exceeding 16M default budget)
    model = create_mock_onnx_model(op_type="Relu", input_shape=(1, 50, 1000, 1000))
    firewall = ONNXGraphFirewall(max_input_elements=16_000_000)
    res = firewall.inspect_graph(model)
    assert res.passed is False, "Oversized input tensor was not blocked!"
    assert any("exceeds maximum" in v for v in res.violations)
    print(f"  [PASS] Blocked oversized input tensor as expected: {res.violations}")


def test_oversized_single_dimension_rejected():
    print("--> Testing oversized single dimension is rejected...")
    # Single dimension > 65536
    model = create_mock_onnx_model(op_type="Relu", input_shape=(1, 100_000, 1, 1))
    firewall = ONNXGraphFirewall(max_dimension_size=65_536)
    res = firewall.inspect_graph(model)
    assert res.passed is False, "Oversized dimension was not blocked!"
    assert any("exceeds maximum dimension size" in v for v in res.violations)
    print(f"  [PASS] Blocked oversized dimension: {res.violations}")


def test_safety_gate_integration():
    print("--> Testing SafetyGate and PolicyEngine integration with firewall violation...")
    temp_dir = tempfile.mkdtemp()
    try:
        bad_model = create_mock_onnx_model(op_type="CustomMaliciousOp")
        bad_model_path = os.path.join(temp_dir, "bad_model.onnx")
        onnx.save(bad_model, bad_model_path)

        gate = SafetyGate()
        safety_res = gate.scan_file(bad_model_path)

        assert safety_res.is_onnx_valid is False
        assert safety_res.hard_gate_triggered is True
        
        firewall_findings = [f for f in safety_res.gate_findings if f.finding_type == "ONNX_GRAPH_FIREWALL"]
        assert len(firewall_findings) == 1
        assert firewall_findings[0].is_hard_gate is True
        assert firewall_findings[0].risk_score == 100.0

        # Evaluate through policy engine
        policy = PolicyEngine()
        verdict = policy.evaluate(safety_res, detector_findings=[])
        assert verdict["verdict"] == "QUARANTINE"
        assert verdict["hard_gate_fired"] is True
        assert "ONNX Graph Firewall Violation" in verdict["hard_gates_triggered"]
        print("  [PASS] SafetyGate and PolicyEngine correctly fail closed on firewall breach.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_api_compatibility():
    print("--> Testing FastAPI /scan/model endpoint compatibility...")
    client = TestClient(app)

    # 1. Test clean classifier through API
    with open("models/clean_classifier.onnx", "rb") as f:
        resp = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "policy" in data
    assert "graph_firewall" in data
    assert data["graph_firewall"]["passed"] is True
    assert "provenance_manifest" in data
    assert "signed_checkpoint" in data
    assert "canonical_certificate" in data
    print(f"  [PASS] Clean model scan API returned verdict: {data['policy']['verdict']}, Firewall passed: {data['graph_firewall']['passed']}")

    # 2. Test backdoored classifier through API
    with open("models/backdoored_classifier.onnx", "rb") as f:
        resp_bd = client.post("/scan/model", files={"file": ("backdoored_classifier.onnx", f, "application/octet-stream")})
    assert resp_bd.status_code == 200, resp_bd.text
    data_bd = resp_bd.json()
    assert "policy" in data_bd
    assert "graph_firewall" in data_bd
    assert data_bd["graph_firewall"]["passed"] is True
    assert "strip_analysis" in data_bd
    print(f"  [PASS] Backdoored model scan API returned verdict: {data_bd['policy']['verdict']}, Firewall passed: {data_bd['graph_firewall']['passed']}")

    # 3. Test malicious custom operator model through API
    temp_dir = tempfile.mkdtemp()
    try:
        bad_model = create_mock_onnx_model(op_type="CustomMaliciousOp")
        bad_path = os.path.join(temp_dir, "malicious_op.onnx")
        onnx.save(bad_model, bad_path)

        with open(bad_path, "rb") as f:
            resp_bad = client.post("/scan/model", files={"file": ("malicious_op.onnx", f, "application/octet-stream")})
        assert resp_bad.status_code == 200, resp_bad.text
        data_bad = resp_bad.json()
        assert data_bad["policy"]["verdict"] == "QUARANTINE"
        assert data_bad["policy"]["hard_gate_fired"] is True
        assert data_bad["graph_firewall"]["passed"] is False
        # Confirm STRIP behavioral inference was skipped
        assert data_bad["strip_analysis"]["status"] == "SKIPPED_INVALID_ONNX_FORMAT"
        print("  [PASS] Malicious operator model through API blocked at Graph Firewall, STRIP bypassed, and QUARANTINED.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING RONOVA ONNX GRAPH FIREWALL TEST SUITE")
    print("==================================================")
    test_clean_model_passes()
    test_backdoored_model_passes()
    test_unapproved_operator_rejected()
    test_unapproved_domain_rejected()
    test_oversized_input_rejected()
    test_oversized_single_dimension_rejected()
    test_safety_gate_integration()
    test_api_compatibility()
    print("==================================================")
    print("ALL ONNX GRAPH FIREWALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
