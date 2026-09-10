import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_script(cmd_list: list, description: str, cwd: str = ".") -> bool:
    print(f"\n[*] Executing: {description} ({' '.join(cmd_list)})")
    start = time.time()
    use_shell = sys.platform == "win32"
    res = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True, shell=use_shell)
    duration = round(time.time() - start, 2)
    if res.returncode == 0:
        print(f"[PASS] {description} ({duration}s)")
        return True
    else:
        print(f"[FAIL] {description} ({duration}s)")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        return False


def main():
    print("=======================================================")
    print("RONOVA PHASE 8: COMPREHENSIVE REGRESSION & RECONCILIATION")
    print("=======================================================")

    results = []

    # 1. Component Phase Scripts
    phase_scripts = [
        ([sys.executable, "scripts/run_phase1_pipeline.py"], "Phase 1: Core Safety & Sandbox Pipeline"),
        ([sys.executable, "scripts/run_phase2_dataset.py"], "Phase 2: Dataset Forensics Engine"),
        ([sys.executable, "scripts/test_phase3_api.py"], "Phase 3: Model Scan API & Provenance"),
        ([sys.executable, "scripts/test_phase4_policy.py"], "Phase 4: Policy Engine & Input Shield"),
        ([sys.executable, "scripts/run_phase6_drift.py"], "Phase 6: Input Distribution Drift Engine"),
        ([sys.executable, "scripts/run_phase7_verification.py"], "Phase 7: Signed Certificates & Offline Verifier"),
    ]

    for cmd, desc in phase_scripts:
        ok = run_script(cmd, desc)
        results.append((desc, ok))

    # 2. Focused Pytest Unit Test Suites
    test_suites = [
        ([sys.executable, "-m", "pytest", "scripts/test_provenance.py"], "Pytest: Provenance Reproducibility & Identity (6 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_qr_certificate.py"], "Pytest: QR & Certificate Offline Verifier (10 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_audit_ledger.py"], "Pytest: Audit Ledger Hash Chain Integrity (11 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_sandbox.py"], "Pytest: Sandbox Subprocess Isolation (5 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_onnx_validation.py"], "Pytest: ONNX Structural Validation (8 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_model_scan_independence.py"], "Pytest: Model Scan Evidence Independence (3 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_key_management.py"], "Pytest: Signing Key Lifecycle & Trust Root (7 tests)"),
        ([sys.executable, "-m", "pytest", "scripts/test_image_sentinel.py"], "Pytest: Image Sentinel / Image Assurance Layer (23 tests)"),
    ]

    for cmd, desc in test_suites:
        ok = run_script(cmd, desc)
        results.append((desc, ok))

    # 3. Frontend Production Build
    frontend_dir = str(Path("frontend").resolve())
    ok_build = run_script(["npm", "run", "build"], "Frontend Production Build (npm run build)", cwd=frontend_dir)
    results.append(("Frontend Production Build (Zero errors)", ok_build))

    # Final Summary Table
    print("\n=======================================================")
    print("FINAL REGRESSION SUITE RECONCILIATION SUMMARY TABLE")
    print("=======================================================")
    all_passed = True
    for name, ok in results:
        status_str = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"[{status_str}] {name}")

    print("=======================================================")
    if all_passed:
        print("ALL 15 REGRESSION SUITES PASSED CLEANLY (73 Focused Unit Tests + 6 Component Scripts + Frontend Build)!")
        print("RONOVA MVP BUILD IS OFFICIALLY LOCKED & FULLY RECONCILED.")
        sys.exit(0)
    else:
        print("REGRESSION PASS FAILED FOR ONE OR MORE SUITES.")
        sys.exit(1)


if __name__ == "__main__":
    main()
