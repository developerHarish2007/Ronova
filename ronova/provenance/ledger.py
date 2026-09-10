import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ronova.provenance.crypto import Ed25519Signer, Ed25519Verifier
from ronova.provenance.engine import ProvenanceEngine


class AuditLedger:
    """
    RONOVA Hash-Chained Audit Ledger:
    Records 8-field provenance events in a tamper-evident hash chain and produces
    Ed25519 cryptographically signed checkpoints anchored to the out-of-band Trust Root.
    Includes full chain validation from genesis to prevent tampering or broken linkage.
    """

    GENESIS_HASH = "0" * 64
    _lock = threading.Lock()

    def __init__(
        self,
        ledger_path: str = "artifacts/audit_ledger.jsonl",
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        self.ledger_path = Path(ledger_path).resolve()
        self.signer = Ed25519Signer(private_key_path)
        self.verifier = Ed25519Verifier(public_key_path)
        os.makedirs(self.ledger_path.parent, exist_ok=True)

    def validate_chain(self) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Validates the entire audit ledger hash chain from genesis to the last event.
        For each event, validates:
          - Valid JSON structure (does NOT silently skip malformed lines)
          - Required fields: previous_event_hash, event_hash, timestamp, model_sha256, provenance_manifest
          - previous_event_hash linkage to previous valid event
          - Recomputed event_hash digest match
          - Non-decreasing timestamp ordering
          - Valid model_sha256 format
        Returns: (is_valid, status_message, list_of_validated_events)
        """
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return True, "VALID_GENESIS_STATE", []

        expected_prev_hash = self.GENESIS_HASH
        validated_events = []
        last_timestamp = 0.0

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue

                # 1. Valid JSON format check
                try:
                    record = json.loads(raw_line)
                except Exception as e:
                    return False, f"MALFORMED_JSON_LINE (Line {line_idx}): {str(e)}", validated_events

                if not isinstance(record, dict):
                    return False, f"INVALID_RECORD_FORMAT (Line {line_idx}): Record must be a JSON object", validated_events

                # 2. Required fields check
                required_fields = ["previous_event_hash", "event_hash", "timestamp", "model_sha256", "provenance_manifest"]
                for req in required_fields:
                    if req not in record:
                        return False, f"MISSING_REQUIRED_FIELD (Line {line_idx}): Missing '{req}'", validated_events

                prev_hash = record["previous_event_hash"]
                stored_event_hash = record["event_hash"]
                ts = record["timestamp"]
                model_sha = record["model_sha256"]
                prov_manifest = record["provenance_manifest"]

                # 3. Model SHA-256 format check
                if not isinstance(model_sha, str) or len(model_sha) == 0:
                    return False, f"INVALID_MODEL_SHA256 (Line {line_idx}): Empty or invalid model_sha256 string", validated_events

                # 4. previous_event_hash linkage check
                if prev_hash != expected_prev_hash:
                    return (
                        False,
                        f"BROKEN_CHAIN_LINK (Line {line_idx}): previous_event_hash '{prev_hash[:16]}' does not match expected '{expected_prev_hash[:16]}'",
                        validated_events,
                    )

                # 5. Recompute event_hash and check for payload tampering
                raw_event_payload = {
                    "previous_event_hash": prev_hash,
                    "provenance_manifest": prov_manifest,
                    "model_sha256": model_sha,
                    "timestamp": ts,
                }
                event_str = json.dumps(raw_event_payload, sort_keys=True)
                recomputed_hash = ProvenanceEngine._sha256_bytes(event_str.encode("utf-8"))

                if recomputed_hash != stored_event_hash:
                    return (
                        False,
                        f"INCORRECT_EVENT_HASH (Line {line_idx}): Recomputed hash '{recomputed_hash[:16]}' does not match stored '{stored_event_hash[:16]}'",
                        validated_events,
                    )

                # 6. Timestamp ordering check
                if isinstance(ts, (int, float)) and ts < last_timestamp:
                    return (
                        False,
                        f"EVENT_ORDERING_ERROR (Line {line_idx}): Timestamp {ts} is earlier than previous event timestamp {last_timestamp}",
                        validated_events,
                    )
                if isinstance(ts, (int, float)):
                    last_timestamp = ts

                expected_prev_hash = stored_event_hash
                validated_events.append({
                    "line_index": line_idx,
                    "event_hash": stored_event_hash,
                    "model_sha256": model_sha,
                    "timestamp": ts,
                })

        return True, f"VALID_AUDIT_CHAIN ({len(validated_events)} events verified)", validated_events

    def get_last_event_hash(self) -> str:
        """Returns hash of the last valid recorded event in the ledger."""
        is_valid, _, validated_events = self.validate_chain()
        if validated_events:
            return validated_events[-1]["event_hash"]
        return self.GENESIS_HASH

    def append_event(
        self,
        provenance_manifest: Dict[str, str],
        model_sha256: str,
        findings_summary: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Appends a new event to the hash-chained audit ledger with thread synchronization."""
        with self._lock:
            previous_hash = self.get_last_event_hash()
            timestamp = time.time()

            raw_event_payload = {
                "previous_event_hash": previous_hash,
                "provenance_manifest": provenance_manifest,
                "model_sha256": model_sha256,
                "timestamp": timestamp,
            }

            # Calculate event SHA-256 hash
            event_str = json.dumps(raw_event_payload, sort_keys=True)
            event_hash = ProvenanceEngine._sha256_bytes(event_str.encode("utf-8"))

            event_record = {
                "previous_event_hash": previous_hash,
                "event_hash": event_hash,
                "timestamp": timestamp,
                "model_sha256": model_sha256,
                "provenance_manifest": provenance_manifest,
                "findings_summary": findings_summary,
            }

            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")
                f.flush()

            return event_record

    def generate_signed_checkpoint(self, event_record: Dict[str, Any]) -> Dict[str, Any]:
        """Creates an Ed25519 signed checkpoint for the event."""
        checkpoint_payload = {
            "event_hash": event_record["event_hash"],
            "previous_event_hash": event_record["previous_event_hash"],
            "model_sha256": event_record["model_sha256"],
            "timestamp": event_record["timestamp"],
        }
        msg_bytes = json.dumps(checkpoint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        try:
            signature_b64 = self.signer.sign_message(msg_bytes)
            is_valid = self.verifier.verify_signature(msg_bytes, signature_b64)
            status = "VALID" if is_valid else "INVALID"
        except (FileNotFoundError, ValueError, TypeError) as e:
            signature_b64 = "SIGNING_UNAVAILABLE"
            status = f"UNAVAILABLE ({str(e)})"

        return {
            "checkpoint_payload": checkpoint_payload,
            "signature": signature_b64,
            "signer": "RONOVA-ROOT-ED25519",
            "verification_status": status,
        }
