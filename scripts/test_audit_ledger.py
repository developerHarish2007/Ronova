"""
Audit Ledger Chain Validation Tests for RONOVA.
Tests all 11 required scenarios for ledger chain integrity and policy integration.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from ronova.provenance.ledger import AuditLedger
from ronova.api.main import app

client = TestClient(app)


@pytest.fixture
def temp_ledger():
    """Provides a temporary file path for audit ledger testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        path = tmp.name
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_missing_ledger_file(temp_ledger):
    """1. Missing ledger file should evaluate as valid genesis state."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is True
    assert msg == "VALID_GENESIS_STATE"
    assert len(events) == 0


def test_empty_ledger_file(temp_ledger):
    """2. Empty ledger file should evaluate as valid genesis state."""
    Path(temp_ledger).touch()
    ledger = AuditLedger(ledger_path=temp_ledger)
    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is True
    assert msg == "VALID_GENESIS_STATE"
    assert len(events) == 0


def test_valid_single_event_ledger(temp_ledger):
    """3. Valid single-event ledger should pass validation."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    manifest = {"step1": "hash1"}
    event = ledger.append_event(
        provenance_manifest=manifest,
        model_sha256="a" * 64,
        findings_summary=[{"rule_id": "TEST", "verdict": "PASS"}]
    )
    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is True
    assert len(events) == 1
    assert events[0]["event_hash"] == event["event_hash"]


def test_valid_multi_event_ledger(temp_ledger):
    """4. Valid multi-event ledger should pass validation with linked hashes."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    m1 = {"manifest": "1"}
    m2 = {"manifest": "2"}
    e1 = ledger.append_event(m1, "a" * 64, [])
    e2 = ledger.append_event(m2, "b" * 64, [])

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is True
    assert len(events) == 2
    assert events[1]["event_hash"] == e2["event_hash"]


def test_modified_event_payload(temp_ledger):
    """5. Modified event payload should trigger incorrect event hash error."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    ledger.append_event({"manifest": "1"}, "a" * 64, [])

    # Tamper with the ledger file: change provenance_manifest
    with open(temp_ledger, "r") as f:
        lines = f.readlines()
    data = json.loads(lines[0])
    data["provenance_manifest"]["manifest"] = "TAMPERED"
    with open(temp_ledger, "w") as f:
        f.write(json.dumps(data) + "\n")

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is False
    assert "INCORRECT_EVENT_HASH" in msg


def test_broken_previous_event_hash(temp_ledger):
    """6. Event with broken previous_event_hash should fail validation."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    ledger.append_event({"m": "1"}, "a" * 64, [])
    ledger.append_event({"m": "2"}, "b" * 64, [])

    with open(temp_ledger, "r") as f:
        lines = f.readlines()
    data2 = json.loads(lines[1])
    data2["previous_event_hash"] = "f" * 64  # Corrupt link
    with open(temp_ledger, "w") as f:
        f.write(lines[0] + json.dumps(data2) + "\n")

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is False
    assert "BROKEN_CHAIN_LINK" in msg


def test_incorrect_event_hash(temp_ledger):
    """7. Event with spoofed event_hash should fail validation."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    ledger.append_event({"m": "1"}, "a" * 64, [])

    with open(temp_ledger, "r") as f:
        lines = f.readlines()
    data = json.loads(lines[0])
    data["event_hash"] = "deadbeef" * 8
    with open(temp_ledger, "w") as f:
        f.write(json.dumps(data) + "\n")

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is False
    assert "INCORRECT_EVENT_HASH" in msg


def test_malformed_json_line(temp_ledger):
    """8. Malformed JSON line should fail validation without silent skipping."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    ledger.append_event({"m": "1"}, "a" * 64, [])

    with open(temp_ledger, "a") as f:
        f.write("{NOT_VALID_JSON}\n")

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is False
    assert "MALFORMED_JSON_LINE" in msg


def test_missing_required_event_field(temp_ledger):
    """9. Event missing a required field should fail validation."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    ledger.append_event({"m": "1"}, "a" * 64, [])

    with open(temp_ledger, "r") as f:
        lines = f.readlines()
    data = json.loads(lines[0])
    del data["previous_event_hash"]
    with open(temp_ledger, "w") as f:
        f.write(json.dumps(data) + "\n")

    is_valid, msg, events = ledger.validate_chain()
    assert is_valid is False
    assert "MISSING_REQUIRED_FIELD" in msg


def test_api_behavior_after_ledger_corruption(monkeypatch, temp_ledger):
    """10. API scan/model should force QUARANTINE with BROKEN_AUDIT_CHAIN on corrupt ledger."""
    from ronova.api import main
    monkeypatch.setattr(main, "ledger", AuditLedger(ledger_path=temp_ledger))

    # Corrupt the ledger
    with open(temp_ledger, "w") as f:
        f.write('{"invalid":"ledger_line"}\n')

    clean_model_path = Path("models/clean_classifier.onnx")
    if not clean_model_path.exists():
        pytest.skip("clean_classifier.onnx model file not available for integration test")

    with open(clean_model_path, "rb") as f:
        response = client.post(
            "/scan/model",
            files={"file": ("clean_classifier.onnx", f, "application/octet-stream")}
        )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["policy"]["verdict"] == "QUARANTINE"
    assert res_data["audit_chain_verification"]["audit_chain_valid"] is False

    findings = res_data["policy"]["findings"]
    broken_chain_finding = next((f for f in findings if f.get("finding_type") == "BROKEN_AUDIT_CHAIN"), None)
    assert broken_chain_finding is not None
    assert broken_chain_finding["is_hard_gate"] is True


def test_valid_signed_checkpoint_after_valid_chain(temp_ledger):
    """11. Signed checkpoint generation should succeed over a valid chain."""
    ledger = AuditLedger(ledger_path=temp_ledger)
    e1 = ledger.append_event({"m": "1"}, "a" * 64, [])
    checkpoint = ledger.generate_signed_checkpoint(e1)

    assert "checkpoint_payload" in checkpoint
    assert "signature" in checkpoint
    assert checkpoint["signer"] == "RONOVA-ROOT-ED25519"
