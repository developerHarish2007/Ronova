import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import onnxruntime as ort


class ProvenanceEngine:
    """
    RONOVA 8-Field Provenance Engine:
    Computes 8 SHA-256 hashed fields per assurance event for reproducibility and auditability.
    Field 1: dataset_hash
    Field 2: model_hash
    Field 3: config_hash
    Field 4: preprocess_hash
    Field 5: runtime_hash = hash(canonical JSON of Python, ORT, dependencies, lockfile)
    Field 6: input_hash
    Field 7: output_hash
    Field 8: assurance_engine_hash = hash(release build Git commit or explicit release ID)
    """

    OLD_PLACEHOLDER_COMMIT = "a7e4b901f8c23d5e718910a2468d09bc12345678"
    DEFAULT_FALLBACK_RELEASE_ID = "RONOVA-v0.2.0-RELEASE-NO-GIT"

    @staticmethod
    def _sha256_bytes(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    @classmethod
    def get_assurance_engine_identity(cls) -> Tuple[str, str]:
        """
        Retrieves release identity string and source type.
        Source priority:
        1. Environment Variable (RONOVA_RELEASE_COMMIT or RONOVA_BUILD_HASH)
        2. Git Commit (via git rev-parse HEAD)
        3. Fallback Release ID ("RONOVA-v0.2.0-RELEASE-NO-GIT")
        Returns: (identity_string, source_type)
        """
        env_commit = os.environ.get("RONOVA_RELEASE_COMMIT") or os.environ.get("RONOVA_BUILD_HASH")
        if env_commit and env_commit.strip():
            return env_commit.strip(), "ENV_VAR"

        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                git_commit = res.stdout.strip()
                return git_commit, "GIT_COMMIT"
        except Exception:
            pass

        return cls.DEFAULT_FALLBACK_RELEASE_ID, "FALLBACK_RELEASE_ID"

    @staticmethod
    def _get_pkg_version(pkg_name: str) -> str:
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            return f"{pkg_name}-n/a"
        except Exception:
            return f"{pkg_name}-unknown"

    def get_runtime_environment_info(self) -> Dict[str, Any]:
        """
        Gathers runtime environment metadata canonically.
        """
        lockfile_hash = "no_lockfile"
        req_path = Path("requirements.txt")
        if req_path.exists():
            try:
                lockfile_hash = self._sha256_bytes(req_path.read_bytes())
            except Exception:
                pass

        env_info = {
            "python_version": sys.version,
            "onnxruntime_version": getattr(ort, "__version__", self._get_pkg_version("onnxruntime")),
            "numpy_version": self._get_pkg_version("numpy"),
            "scipy_version": self._get_pkg_version("scipy"),
            "torch_version": self._get_pkg_version("torch"),
            "modelscan_version": self._get_pkg_version("modelscan"),
            "fastapi_version": self._get_pkg_version("fastapi"),
            "lockfile_sha256": lockfile_hash,
        }
        return env_info

    def build_runtime_hash(self, custom_runtime_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Builds deterministic runtime_hash using canonical JSON serialization.
        """
        env_info = custom_runtime_info if custom_runtime_info is not None else self.get_runtime_environment_info()
        canonical_json = json.dumps(env_info, sort_keys=True, separators=(",", ":"))
        return self._sha256_bytes(canonical_json.encode("utf-8"))

    def compute_provenance_manifest(
        self,
        model_sha256: str,
        dataset_array: np.ndarray,
        config_dict: Dict[str, Any],
        preprocess_info: str,
        input_array: np.ndarray,
        output_array: np.ndarray,
        custom_runtime_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        dataset_hash = self._sha256_bytes(dataset_array.tobytes())
        model_hash = model_sha256

        # Canonical JSON serialization for config_hash
        canonical_config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        config_hash = self._sha256_bytes(canonical_config_json.encode("utf-8"))

        preprocess_hash = self._sha256_bytes(preprocess_info.encode("utf-8"))
        runtime_hash = self.build_runtime_hash(custom_runtime_info)
        input_hash = self._sha256_bytes(input_array.tobytes())
        output_hash = self._sha256_bytes(output_array.tobytes())

        # Release identity-based assurance engine hash
        engine_id, _ = self.get_assurance_engine_identity()
        assurance_engine_hash = self._sha256_bytes(engine_id.encode("utf-8"))

        return {
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "config_hash": config_hash,
            "preprocess_hash": preprocess_hash,
            "runtime_hash": runtime_hash,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "assurance_engine_hash": assurance_engine_hash,
        }
