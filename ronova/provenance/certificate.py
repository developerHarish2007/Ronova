import base64
import hashlib
import json
import time
from typing import Dict, Any, List, Optional

from ronova.provenance.crypto import Ed25519Signer, Ed25519Verifier


def compute_canonical_json_hash(payload_dict: Dict[str, Any]) -> str:
    """Computes SHA-256 hash of sorted, canonical JSON representation."""
    canonical_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


class AssuranceCertificateBuilder:
    """
    RONOVA Canonical Assurance Certificate Generator (Phase 7):
    Creates exportable, cryptographically signed JSON assurance certificates containing
    verdict, risk score, evidence strength, 8-field provenance hashes, findings, timestamp,
    Ed25519 signature, and compact QR payload.
    """

    def __init__(
        self,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        self.signer = Ed25519Signer(private_key_path)
        self.verifier = Ed25519Verifier(public_key_path)

    def generate_certificate(
        self,
        policy_verdict: Dict[str, Any],
        model_sha256: str,
        provenance_manifest: Dict[str, str],
        findings: List[Dict[str, Any]],
        timestamp: float = None,
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()

        # Build canonical payload for signing
        canonical_payload = {
            "framework": "RONOVA v0.2",
            "verdict": policy_verdict.get("verdict"),
            "risk_score": policy_verdict.get("risk_score"),
            "evidence_strength": policy_verdict.get("evidence_strength"),
            "hard_gate_fired": policy_verdict.get("hard_gate_fired", False),
            "hard_gates_triggered": policy_verdict.get("hard_gates_triggered", []),
            "model_sha256": model_sha256,
            "provenance_manifest": provenance_manifest,
            "findings_summary": [
                {
                    "title": f.get("title"),
                    "finding_type": f.get("finding_type"),
                    "risk_score": f.get("risk_score"),
                    "is_hard_gate": f.get("is_hard_gate", False),
                }
                for f in findings
            ],
            "timestamp": timestamp,
        }

        # Calculate canonical hash
        cert_hash = compute_canonical_json_hash(canonical_payload)
        payload_bytes = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        # Sign canonical payload with Ed25519 private key (fails closed if private key is missing/unusable)
        signature_b64 = self.signer.sign_message(payload_bytes)

        # Un-truncated Canonical QR payload encoding full hashes, signature, and canonical payload for independent offline verification
        qr_payload = {
            "spec_version": "RONOVA-QR-v1",
            "signer_identity": "RONOVA-ROOT-ED25519",
            "cert_hash": cert_hash,
            "model_sha256": model_sha256,
            "verdict": policy_verdict.get("verdict"),
            "risk_score": policy_verdict.get("risk_score"),
            "timestamp": timestamp,
            "canonical_payload": canonical_payload,
            "signature": signature_b64,
        }

        certificate = {
            "certificate_header": {
                "issuer": "RONOVA Assurance Engine",
                "spec_version": "Phase 7 Final",
                "signer_identity": "RONOVA-ROOT-ED25519",
                "cert_hash": cert_hash,
            },
            "canonical_payload": canonical_payload,
            "signature": signature_b64,
            "qr_payload": qr_payload,
        }

        return certificate
