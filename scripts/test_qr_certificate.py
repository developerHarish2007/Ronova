"""
QR & Standalone Certificate Verification Tests for RONOVA (Sixth Fix).
Tests all 10 required scenarios for offline certificate and QR payload verification.
"""

import json
import os
import pytest
from pathlib import Path

from ronova.provenance.certificate import AssuranceCertificateBuilder
from verify_certificate import verify_certificate_data, verify_certificate_file


@pytest.fixture
def cert_builder():
    """Provides AssuranceCertificateBuilder instance."""
    return AssuranceCertificateBuilder()


@pytest.fixture
def valid_cert(cert_builder):
    """Generates a valid certificate dictionary."""
    policy_verdict = {
        "verdict": "ACCEPT",
        "risk_score": 2.5,
        "evidence_strength": "HIGH",
        "hard_gate_fired": False,
        "hard_gates_triggered": [],
    }
    manifest = {"manifest": "hash1"}
    findings = [{"title": "Clean Model Check", "finding_type": "CLEAN", "risk_score": 2.5, "is_hard_gate": False}]
    return cert_builder.generate_certificate(
        policy_verdict=policy_verdict,
        model_sha256="a" * 64,
        provenance_manifest=manifest,
        findings=findings,
        timestamp=1700000000.0,
    )


def test_valid_json_certificate_verification(valid_cert):
    """1. Test valid full JSON certificate verification offline."""
    is_valid, msg, details = verify_certificate_data(valid_cert)
    assert is_valid is True
    assert "VALID" in msg
    assert details["verdict"] == "ACCEPT"
    assert details["model_sha256"] == "a" * 64
    assert details["signer_identity"] == "RONOVA-ROOT-ED25519"


def test_valid_qr_payload_verification(valid_cert):
    """2. Test valid QR payload string/dictionary verification offline."""
    qr_payload = valid_cert["qr_payload"]
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is True
    assert "VALID" in msg
    assert details["verification_type"] == "QR_PAYLOAD"
    assert details["verdict"] == "ACCEPT"
    assert details["model_sha256"] == "a" * 64


def test_tampered_qr_certificate_hash(valid_cert):
    """3. Test tampered QR certificate hash fails verification."""
    qr_payload = dict(valid_cert["qr_payload"])
    qr_payload["cert_hash"] = "f" * 64  # Corrupt cert_hash
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "CERT_HASH_MISMATCH" in msg


def test_tampered_qr_signature(valid_cert):
    """4. Test tampered QR signature fails verification."""
    qr_payload = dict(valid_cert["qr_payload"])
    # Modify base64 signature
    qr_payload["signature"] = "A" * len(qr_payload["signature"])
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "INVALID_SIGNATURE" in msg


def test_tampered_model_hash(valid_cert):
    """5. Test tampered top-level model hash fails verification."""
    qr_payload = dict(valid_cert["qr_payload"])
    qr_payload["model_sha256"] = "b" * 64  # Modify model hash claim
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "MODEL_HASH_MISMATCH" in msg


def test_tampered_verdict(valid_cert):
    """6. Test tampered verdict claim (e.g., changing QUARANTINE to ACCEPT) fails verification."""
    # Create QUARANTINE cert first
    builder = AssuranceCertificateBuilder()
    cert = builder.generate_certificate(
        policy_verdict={"verdict": "QUARANTINE", "risk_score": 95.0, "evidence_strength": "HIGH", "hard_gate_fired": True},
        model_sha256="b" * 64,
        provenance_manifest={"manifest": "hash2"},
        findings=[{"title": "Trojan Indicator", "finding_type": "TROJAN", "risk_score": 95.0, "is_hard_gate": True}],
    )
    qr_payload = dict(cert["qr_payload"])
    qr_payload["verdict"] = "ACCEPT"  # Tamper claim
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "VERDICT_MISMATCH" in msg


def test_missing_qr_fields(valid_cert):
    """7. Test QR payload missing essential fields fails verification."""
    qr_payload = dict(valid_cert["qr_payload"])
    del qr_payload["signature"]
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "MISSING_SIGNATURE" in msg


def test_invalid_signer_identity(valid_cert):
    """8. Test invalid signer identity string fails verification."""
    qr_payload = dict(valid_cert["qr_payload"])
    qr_payload["signer_identity"] = "UNKNOWN-SIGNER"
    is_valid, msg, details = verify_certificate_data(qr_payload)
    assert is_valid is False
    assert "INVALID_SIGNER_IDENTITY" in msg


def test_verification_without_private_key(valid_cert, monkeypatch):
    """9. Test verification works cleanly using ONLY trust/root_public.pem (no private key needed)."""
    # Verify using verify_certificate_data with public key path only
    pub_key_path = "trust/root_public.pem"
    assert os.path.exists(pub_key_path)
    is_valid, msg, details = verify_certificate_data(valid_cert, public_key_path=pub_key_path)
    assert is_valid is True
    assert "VALID" in msg


def test_backward_compatibility_with_existing_certificates():
    """10. Test backward compatibility with existing valid JSON certificates on disk."""
    clean_cert_path = "artifacts/clean_certificate.json"
    if os.path.exists(clean_cert_path):
        is_valid = verify_certificate_file(clean_cert_path)
        assert is_valid is True
