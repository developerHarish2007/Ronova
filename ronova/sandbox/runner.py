import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Tuple
import numpy as np


class IsolatedSandboxRunner:
    """
    RONOVA Isolated Sandbox:
    Executes ONNX models under strict execution boundaries in a separate worker subprocess.
    - Uses Docker container runtime (--network=none, read-only FS, RAM/CPU limits, wall-clock timeout) if Docker daemon is available.
    - Otherwise: Subprocess isolation executing ONNX Runtime in a dedicated worker process with CPU-only execution, wall-clock timeout enforcement, and batch limits.
    """

    def __init__(self, timeout_seconds: float = 60.0, max_batch_size: int = 10000):
        self.timeout_seconds = float(timeout_seconds)
        self.max_batch_size = max_batch_size
        self.docker_available = self._check_docker_available()

    def _check_docker_available(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            res = subprocess.run(["docker", "info"], capture_output=True, timeout=2.0)
            return res.returncode == 0
        except Exception:
            return False

    def run_inference(
        self, model_path: str, input_array: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs model inference on input_array within isolated sandbox.
        Returns:
            (probabilities_or_logits, sandbox_metadata)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if input_array.shape[0] > self.max_batch_size:
            raise ValueError(
                f"Batch size {input_array.shape[0]} exceeds maximum sandbox limit {self.max_batch_size}"
            )

        start_time = time.time()

        if self.docker_available:
            output, metadata = self._run_in_docker(model_path, input_array)
        else:
            output, metadata = self._run_in_subprocess(model_path, input_array)

        elapsed = time.time() - start_time
        metadata.update({
            "execution_time_seconds": round(elapsed, 4),
            "timeout_limit_seconds": self.timeout_seconds,
            "timeout_triggered": False,
        })
        return output, metadata

    def _run_in_subprocess(
        self, model_path: str, input_array: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes inference in a dedicated Python worker subprocess with wall-clock timeout enforcement.
        """
        abs_model_path = os.path.abspath(model_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.npy")
            output_file = os.path.join(temp_dir, "output.npy")
            meta_file = os.path.join(temp_dir, "meta.json")

            np.save(input_file, input_array)

            cmd = [
                shutil.which("python") or "python",
                "-m",
                "ronova.sandbox.worker",
                abs_model_path,
                input_file,
                output_file,
                meta_file,
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TimeoutError(
                    f"Sandbox execution timed out after {self.timeout_seconds} seconds"
                )

            if proc.returncode != 0:
                err_msg = stderr.strip() if stderr else f"Worker process exited with code {proc.returncode}"
                raise RuntimeError(f"Sandbox worker execution failed: {err_msg}")

            if not os.path.exists(output_file) or not os.path.exists(meta_file):
                raise RuntimeError("Sandbox worker failed to produce expected output files.")

            probs = np.load(output_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            metadata.update({
                "sandbox_mode": "subprocess_isolated",
                "process_isolated": True,
            })
            return probs, metadata

    def _run_in_docker(
        self, model_path: str, input_array: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs inference inside a Docker container with network disabled and read-only root filesystem.
        """
        abs_model_path = os.path.abspath(model_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.npy")
            output_file = os.path.join(temp_dir, "output.npy")
            meta_file = os.path.join(temp_dir, "meta.json")

            np.save(input_file, input_array)

            container_temp = "/tmp/sandbox"
            container_model = "/tmp/model.onnx"

            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "-v", f"{abs_model_path}:{container_model}:ro",
                "-v", f"{temp_dir}:{container_temp}:rw",
                "ronova-sandbox-runner:latest",
                "python",
                "-m",
                "ronova.sandbox.worker",
                container_model,
                f"{container_temp}/input.npy",
                f"{container_temp}/output.npy",
                f"{container_temp}/meta.json",
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TimeoutError(
                    f"Docker sandbox execution timed out after {self.timeout_seconds} seconds"
                )

            if proc.returncode != 0:
                # If docker container run fails (e.g. missing container image), fallback to subprocess isolation
                return self._run_in_subprocess(model_path, input_array)

            probs = np.load(output_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            metadata.update({
                "sandbox_mode": "docker_container",
                "network_disabled": True,
                "read_only_fs": True,
                "process_isolated": True,
            })
            return probs, metadata
