import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.provenance.crypto import Ed25519Signer, Ed25519Verifier
from ronova.provenance.certificate import AssuranceCertificateBuilder, compute_canonical_json_hash
from ronova.api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_signing_and_verification_custom_paths():
    with tempfile.TemporaryDirectory() as tmp_dir:
        priv_path = os.path.join(tmp_dir, "custom_private.pem")
        pub_path = os.path.join(tmp_dir, "custom_public.pem")

        signer = Ed25519Signer(private_key_path=priv_path)
        signer.generate_keypair(public_key_path=pub_path)

        assert os.path.exists(priv_path)
        assert os.path.exists(pub_path)

        msg = b"RONOVA Evidence Payload 123"
        sig = signer.sign_message(msg)
        assert isinstance(sig, str) and len(sig) > 0

        verifier = Ed25519Verifier(public_key_path=pub_path)
        assert verifier.verify_signature(msg, sig) is True


def test_missing_private_key():
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent_priv = os.path.join(tmp_dir, "missing_private.pem")
        signer = Ed25519Signer(private_key_path=non_existent_priv)

        with pytest.raises(FileNotFoundError) as exc_info:
            signer.sign_message(b"test message")

        assert "not found" in str(exc_info.value).lower()


def test_invalid_private_key_file():
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_PEM_PRIVATE_KEY_DATA")
        invalid_priv_path = tmp.name

    try:
        signer = Ed25519Signer(private_key_path=invalid_priv_path)
        with pytest.raises(ValueError) as exc_info:
            signer.sign_message(b"test message")
        assert "Invalid private key" in str(exc_info.value)
    finally:
        if os.path.exists(invalid_priv_path):
            os.remove(invalid_priv_path)


def test_invalid_public_key_file():
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_PEM_PUBLIC_KEY_DATA")
        invalid_pub_path = tmp.name

    try:
        verifier = Ed25519Verifier(public_key_path=invalid_pub_path)
        with pytest.raises(ValueError) as exc_info:
            verifier.verify_signature(b"test message", "dGVzdF9zaWduYXR1cmU=")
        assert "Invalid public key" in str(exc_info.value)
    finally:
        if os.path.exists(invalid_pub_path):
            os.remove(invalid_pub_path)


def test_verification_without_private_key_access():
    with tempfile.TemporaryDirectory() as tmp_dir:
        priv_path = os.path.join(tmp_dir, "temp_private.pem")
        pub_path = os.path.join(tmp_dir, "temp_public.pem")

        signer = Ed25519Signer(private_key_path=priv_path)
        signer.generate_keypair(public_key_path=pub_path)

        msg = b"Verification test message"
        sig = signer.sign_message(msg)

        # Delete private key file to simulate isolated verification environment
        os.remove(priv_path)
        assert not os.path.exists(priv_path)

        # Verifier runs using ONLY public key
        verifier = Ed25519Verifier(public_key_path=pub_path)
        assert verifier.verify_signature(msg, sig) is True


def test_tampered_certificate_rejection():
    builder = AssuranceCertificateBuilder()
    cert = builder.generate_certificate(
        policy_verdict={"verdict": "ACCEPT", "risk_score": 2.5, "evidence_strength": "HIGH"},
        model_sha256="c2d44045fcf72ef1581c34107e34139da5b00dcce0560d7f74976ce4b0ba37a1",
        provenance_manifest={"model_hash": "c2d44045fcf72ef1581c34107e34139da5b00dcce0560d7f74976ce4b0ba37a1"},
        findings=[],
    )

    verifier = Ed25519Verifier()
    payload = cert["canonical_payload"]
    sig = cert["signature"]

    msg_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert verifier.verify_signature(msg_bytes, sig) is True

    # Tamper payload
    tampered_payload = dict(payload)
    tampered_payload["risk_score"] = 99.9
    tampered_payload["verdict"] = "QUARANTINE"

    tampered_bytes = json.dumps(tampered_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert verifier.verify_signature(tampered_bytes, sig) is False


def test_api_startup_and_signing_failure_when_private_key_missing(client, monkeypatch):
    # Set environment variable to a non-existent private key path
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_priv = os.path.join(tmp_dir, "missing_key.pem")
        monkeypatch.setenv("RONOVA_PRIVATE_KEY_PATH", missing_priv)

        # 1. API Health endpoint works cleanly without private key
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "HEALTHY"

        # 2. Scanning model fails cleanly with 503 HTTP status when signing key is missing
        clean_model_path = "models/clean_classifier.onnx"
        with open(clean_model_path, "rb") as f:
            res_scan = client.post("/scan/model", files={"file": ("clean_classifier.onnx", f, "application/octet-stream")})

        assert res_scan.status_code == 503
        assert "Certificate signing unavailable" in res_scan.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])
