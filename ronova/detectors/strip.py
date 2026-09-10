import time
from typing import List, Dict, Any, Tuple
import numpy as np

from ronova.core.types import (
    Finding,
    EvidenceStrength,
    STRIPResult,
)
from ronova.sandbox.runner import IsolatedSandboxRunner


class STRIPDetector:
    """
    STRIP Behavioral Trojan Detector (STRong Intent Perturbation):
    Ports core entropy-under-perturbation analysis.
    Overlays test samples with reference clean patterns.
    Clean models display high prediction entropy across perturbations.
    Backdoored models exhibit suppressed entropy because the model locks onto
    the trigger pattern regardless of overlay noise.
    """

    def __init__(
        self,
        sandbox_runner: IsolatedSandboxRunner,
        num_perturbations: int = 20,
        blend_alpha: float = 0.5,
        entropy_threshold: float = 0.35,
    ):
        self.sandbox = sandbox_runner
        self.num_perturbations = num_perturbations
        self.blend_alpha = blend_alpha
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def compute_shannon_entropy(probs: np.ndarray) -> np.ndarray:
        """
        Computes Shannon entropy across class probabilities for each batch sample.
        H = - sum(p * log2(p))
        """
        eps = 1e-12
        clipped_probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(clipped_probs * np.log2(clipped_probs), axis=-1)
        return entropy

    def evaluate_model(
        self,
        model_path: str,
        test_samples: np.ndarray,
        clean_overlays: np.ndarray,
    ) -> Tuple[STRIPResult, Dict[str, Any]]:
        """
        Evaluates test_samples against model_path using clean_overlays.
        test_samples: shape (N, C, H, W) or (N, H, W, C)
        clean_overlays: shape (M, C, H, W) or (M, H, W, C)
        """
        num_test_samples = test_samples.shape[0]
        num_overlays = clean_overlays.shape[0]

        all_entropies: List[float] = []
        sample_summaries: List[Dict[str, Any]] = []

        total_sandbox_meta = {}

        for i in range(num_test_samples):
            test_img = test_samples[i : i + 1] # shape (1, ...)

            # Generate N perturbed variants by blending test_img with random clean overlays
            rand_indices = np.random.choice(num_overlays, size=self.num_perturbations, replace=True)
            overlays_selected = clean_overlays[rand_indices] # shape (N_pert, ...)

            # Linear superimposition: I_blend = alpha * I_test + (1 - alpha) * I_overlay
            blended_variants = (
                self.blend_alpha * test_img + (1.0 - self.blend_alpha) * overlays_selected
            ).astype(np.float32)

            # Pass blended variants through Isolated Sandbox
            probs, sandbox_meta = self.sandbox.run_inference(model_path, blended_variants)
            total_sandbox_meta = sandbox_meta

            # Calculate entropy per variant
            variant_entropies = self.compute_shannon_entropy(probs)
            mean_sample_entropy = float(np.mean(variant_entropies))
            all_entropies.extend([float(e) for e in variant_entropies])

            # Top predicted classes for variant summary
            top_classes = np.argmax(probs, axis=-1).tolist()

            sample_summaries.append({
                "sample_index": i,
                "mean_entropy": mean_sample_entropy,
                "top_predicted_classes": top_classes[:5],
                "min_variant_entropy": float(np.min(variant_entropies)),
                "max_variant_entropy": float(np.max(variant_entropies)),
            })

        mean_overall_entropy = float(np.mean(all_entropies))
        std_overall_entropy = float(np.std(all_entropies))
        min_overall_entropy = float(np.min(all_entropies))

        # Backdoor / Trojan decision logic
        trojan_triggered = mean_overall_entropy < self.entropy_threshold

        if trojan_triggered:
            finding = Finding(
                detector_id="strip_behavioral_analyzer",
                finding_type="TROJAN_BEHAVIOR_INDICATOR",
                risk_score=95.0,
                evidence_strength=EvidenceStrength.HIGH,
                title="High-Strength Trojan Indicator",
                explanation=(
                    f"STRIP behavioral analysis detected suppressed entropy under perturbation "
                    f"(Mean Entropy: {mean_overall_entropy:.4f} < Threshold: {self.entropy_threshold:.4f}). "
                    f"Model prediction locked onto a single target state despite heavy visual overlay blending, "
                    f"behavior consistent with a Trojan / backdoor trigger."
                ),
                limitations=(
                    "STRIP behavioral entropy analysis detects backdoor trigger suppression under perturbation; "
                    "it does not localize or reconstruct white-box trigger heatmaps."
                ),
                evidence_details={
                    "mean_entropy": mean_overall_entropy,
                    "std_entropy": std_overall_entropy,
                    "min_entropy": min_overall_entropy,
                    "threshold": self.entropy_threshold,
                    "num_samples_evaluated": num_test_samples,
                    "num_perturbations_per_sample": self.num_perturbations,
                },
                is_hard_gate=True, # High-Strength Trojan Indicator triggers Hard Gate per Section 10
            )
        else:
            finding = Finding(
                detector_id="strip_behavioral_analyzer",
                finding_type="TROJAN_BEHAVIOR_CLEAN",
                risk_score=5.0,
                evidence_strength=EvidenceStrength.HIGH,
                title="Trojan Behavioral Analysis Passed",
                explanation=(
                    f"STRIP behavioral analysis measured normal entropy under perturbation "
                    f"(Mean Entropy: {mean_overall_entropy:.4f} >= Threshold: {self.entropy_threshold:.4f}). "
                    f"Model predictions varied across overlay blends as expected for an uncompromised classifier."
                ),
                limitations=(
                    "STRIP behavioral entropy analysis detects backdoor trigger suppression under perturbation; "
                    "it does not guarantee absolute immunity against highly sophisticated clean-label backdoors."
                ),
                evidence_details={
                    "mean_entropy": mean_overall_entropy,
                    "std_entropy": std_overall_entropy,
                    "min_entropy": min_overall_entropy,
                    "threshold": self.entropy_threshold,
                },
                is_hard_gate=False,
            )

        strip_result = STRIPResult(
            num_samples_evaluated=num_test_samples,
            num_perturbations_per_sample=self.num_perturbations,
            mean_entropy=mean_overall_entropy,
            std_entropy=std_overall_entropy,
            min_entropy=min_overall_entropy,
            trojan_indicator_triggered=trojan_triggered,
            entropy_distribution=all_entropies,
            sample_summaries=sample_summaries,
            finding=finding,
        )

        return strip_result, total_sandbox_meta
