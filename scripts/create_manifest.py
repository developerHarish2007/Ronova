import hashlib
import json
import os
from pathlib import Path


def compute_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    os.makedirs("manifests", exist_ok=True)
    clean_model_path = "models/clean_classifier.onnx"

    if os.path.exists(clean_model_path):
        clean_sha256 = compute_sha256(clean_model_path)
    else:
        clean_sha256 = "c2d44045fcf72ef1581c34107e34139da5b00dcce0560d7f74976ce4b0ba37a1"

    manifest_data = {
        "manifest_version": "1.0",
        "approved_models": [
            {
                "model_id": "vision-classifier-clean-001",
                "version": "1.0.0",
                "sha256": clean_sha256,
                "signer": "RONOVA-ROOT-ED25519",
                "approved_at": "2026-09-10T12:00:00Z",
            }
        ],
    }

    manifest_path = "manifests/approved_models.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Created approved manifest -> {manifest_path} with clean model SHA-256: {clean_sha256}")


if __name__ == "__main__":
    main()
