#!/usr/bin/env python3
"""
RONOVA Standalone Offline Certificate Verifier:
Verifies the Ed25519 cryptographic signature of a RONOVA JSON Assurance Certificate
or QR payload against trust/root_public.pem.
Completely standalone — runs offline with NO dependency on the RONOVA FastAPI backend,
database, or private key.
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
except ImportError:
    print("ERROR: 'cryptography' library is required. Run 'pip install cryptography' first.")
    sys.exit(1)


EXPECTED_SIGNER_IDENTITY = "RONOVA-ROOT-ED25519"


def verify_certificate_data(
    cert_input: Union[Dict[str, Any], str],
    public_key_path: str = "trust/root_public.pem"
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verifies a RONOVA certificate dictionary or JSON string (JSON file content or QR payload).
    Returns (is_valid: bool, status_message: str, details_dict: dict).
    """
    pub_path = Path(public_key_path).resolve()
    if not pub_path.exists():
        return False, f"Trust Root public key not found -> {pub_path}", {}

    # Parse input string if passed as JSON string
    if isinstance(cert_input, str):
        try:
            cert_data = json.loads(cert_input)
        except Exception as e:
            return False, f"MALFORMED_INPUT_JSON: {str(e)}", {}
    elif isinstance(cert_input, dict):
        cert_data = cert_input
    else:
        return False, "INVALID_INPUT_TYPE: Input must be a dict or JSON string", {}

    details = {
        "pub_key_path": str(pub_path),
        "signer_identity": None,
        "model_sha256": None,
        "verdict": None,
        "risk_score": None,
        "cert_hash": None,
        "verification_type": "UNKNOWN",
    }

    try:
        # Load Ed25519 Public Key
        with open(pub_path, "rb") as f:
            pub_key = serialization.load_pem_public_key(f.read())

        if not isinstance(pub_key, ed25519.Ed25519PublicKey):
            return False, "INVALID_KEY_TYPE: Loaded key is not an Ed25519 public key", details

        payload = None
        b64_sig = None
        expected_cert_hash = None
        expected_model_sha = None
        expected_verdict = None
        signer_id = None

        # 1. QR Payload Format
        if "spec_version" in cert_data and cert_data.get("spec_version") == "RONOVA-QR-v1":
            details["verification_type"] = "QR_PAYLOAD"
            payload = cert_data.get("canonical_payload")
            b64_sig = cert_data.get("signature")
            expected_cert_hash = cert_data.get("cert_hash")
            expected_model_sha = cert_data.get("model_sha256")
            expected_verdict = cert_data.get("verdict")
            signer_id = cert_data.get("signer_identity")

        # 2. Canonical Certificate Format
        elif "canonical_payload" in cert_data and "signature" in cert_data:
            details["verification_type"] = "FULL_JSON_CERTIFICATE"
            payload = cert_data["canonical_payload"]
            b64_sig = cert_data["signature"]
            header = cert_data.get("certificate_header", {})
            expected_cert_hash = header.get("cert_hash") or cert_data.get("cert_hash")
            signer_id = header.get("signer_identity") or cert_data.get("signer_identity") or EXPECTED_SIGNER_IDENTITY

        # 3. Signed Checkpoint Format (backward compatibility)
        elif "signed_checkpoint" in cert_data:
            details["verification_type"] = "SIGNED_CHECKPOINT"
            checkpoint = cert_data["signed_checkpoint"]
            payload = checkpoint.get("checkpoint_payload")
            b64_sig = checkpoint.get("signature")
            signer_id = checkpoint.get("signer") or EXPECTED_SIGNER_IDENTITY

        # 4. QR payload as embedded dict
        elif "qr_payload" in cert_data and isinstance(cert_data["qr_payload"], dict):
            details["verification_type"] = "EMBEDDED_QR_PAYLOAD"
            qr = cert_data["qr_payload"]
            payload = qr.get("canonical_payload") or cert_data.get("canonical_payload")
            b64_sig = qr.get("signature") or cert_data.get("signature")
            expected_cert_hash = qr.get("cert_hash")
            expected_model_sha = qr.get("model_sha256")
            expected_verdict = qr.get("verdict")
            signer_id = qr.get("signer_identity") or EXPECTED_SIGNER_IDENTITY

        else:
            return False, "UNRECOGNIZED_FORMAT: Certificate missing required canonical_payload/signature", details

        # Verify presence of essential components
        if not payload or not isinstance(payload, dict):
            return False, "MISSING_CANONICAL_PAYLOAD: Certificate/QR payload has no valid canonical_payload", details

        if not b64_sig or not isinstance(b64_sig, str):
            return False, "MISSING_SIGNATURE: Certificate/QR payload has no signature", details

        # Signer Identity Validation
        if signer_id is not None and signer_id != EXPECTED_SIGNER_IDENTITY:
            return False, f"INVALID_SIGNER_IDENTITY: Expected '{EXPECTED_SIGNER_IDENTITY}', got '{signer_id}'", details
        details["signer_identity"] = signer_id or EXPECTED_SIGNER_IDENTITY

        # Reconstruct canonical JSON bytes
        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        recomputed_cert_hash = hashlib.sha256(canonical_bytes).hexdigest()
        details["cert_hash"] = recomputed_cert_hash

        # Cert Hash Verification
        if expected_cert_hash and expected_cert_hash != recomputed_cert_hash:
            return False, f"CERT_HASH_MISMATCH: Computed hash '{recomputed_cert_hash[:16]}' does not match expected '{expected_cert_hash[:16]}'", details

        # Top-level Model SHA-256 Claim Verification
        payload_model_sha = payload.get("model_sha256")
        if expected_model_sha and payload_model_sha != expected_model_sha:
            return False, f"MODEL_HASH_MISMATCH: Payload model_sha256 '{payload_model_sha}' does not match QR claim '{expected_model_sha}'", details
        details["model_sha256"] = payload_model_sha or "UNKNOWN"

        # Top-level Verdict Claim Verification
        payload_verdict = payload.get("verdict") or payload.get("status")
        if expected_verdict and payload_verdict != expected_verdict:
            return False, f"VERDICT_MISMATCH: Payload verdict '{payload_verdict}' does not match QR claim '{expected_verdict}'", details
        details["verdict"] = payload_verdict or "UNKNOWN"
        details["risk_score"] = payload.get("risk_score")

        # Cryptographic Signature Verification
        try:
            sig_bytes = base64.b64decode(b64_sig)
            pub_key.verify(sig_bytes, canonical_bytes)
        except (InvalidSignature, Exception) as e:
            return False, f"INVALID_SIGNATURE: Signature verification failed ({str(e)})", details

        return True, "VALID (Authentic Signed Certificate)", details

    except Exception as e:
        return False, f"VERIFICATION_ERROR: {str(e)}", details


def verify_certificate_file(cert_file_path: str, public_key_path: str = "trust/root_public.pem") -> bool:
    """Verifies a certificate file and prints verification details to console."""
    cert_path = Path(cert_file_path).resolve()
    pub_path = Path(public_key_path).resolve()

    if not cert_path.exists():
        print(f"ERROR: Certificate file not found -> {cert_path}")
        return False

    with open(cert_path, "r", encoding="utf-8") as f:
        content = f.read()

    is_valid, msg, details = verify_certificate_data(content, str(pub_path))

    print("=======================================================")
    print("RONOVA OFFLINE CERTIFICATE VERIFICATION")
    print("=======================================================")
    print(f"File/Source Verified: {cert_path.name}")
    print(f"Trust Root Key      : {pub_path}")
    if details.get("signer_identity"):
        print(f"Signer Identity     : {details['signer_identity']}")
    if details.get("model_sha256"):
        print(f"Model SHA-256       : {details['model_sha256']}")
    if details.get("verdict"):
        print(f"Governed Verdict    : {details['verdict']}")
    if details.get("risk_score") is not None:
        print(f"Risk Score          : {details['risk_score']}/100")
    print("-------------------------------------------------------")

    if is_valid:
        print(f"VERIFICATION STATUS : {msg}")
        print("NOTICE              : Cryptographic signature proves certificate authenticity")
        print("                      and payload integrity. It does NOT guarantee model safety.")
        print("=======================================================")
        return True
    else:
        print(f"VERIFICATION STATUS : FAILED ({msg})")
        print("=======================================================")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_certificate.py <path_to_certificate.json_or_raw_json_string> [path_to_root_public.pem]")
        sys.exit(1)

    input_arg = sys.argv[1]
    pub_key_file = sys.argv[2] if len(sys.argv) > 2 else "trust/root_public.pem"

    if os.path.exists(input_arg):
        is_valid = verify_certificate_file(input_arg, pub_key_file)
    else:
        # Treat input_arg as raw JSON string (e.g., pasted QR payload)
        is_valid, msg, details = verify_certificate_data(input_arg, pub_key_file)
        print("=======================================================")
        print("RONOVA OFFLINE CERTIFICATE VERIFICATION (Text / QR Input)")
        print("=======================================================")
        print(f"Trust Root Key      : {pub_key_file}")
        if details.get("signer_identity"):
            print(f"Signer Identity     : {details['signer_identity']}")
        if details.get("model_sha256"):
            print(f"Model SHA-256       : {details['model_sha256']}")
        if details.get("verdict"):
            print(f"Governed Verdict    : {details['verdict']}")
        print("-------------------------------------------------------")
        print(f"VERIFICATION STATUS : {'VALID' if is_valid else 'FAILED'} ({msg})")
        print("NOTICE              : Signature proves certificate authenticity, not model safety.")
        print("=======================================================")

    sys.exit(0 if is_valid else 2)


if __name__ == "__main__":
    main()
