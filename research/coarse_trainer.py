"""Multi-seed trainer and CEF evaluator for the C1 coarse interaction baseline."""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from diagnostics.metrics import MetricEvaluator
from research.coarse_seed_fitter import CoarseSeedFitter


class CoarseBaselineTrainer:
    """Trains C1 across multiple seeds, calibrates tau_b, and reports CEF."""

    @staticmethod
    def _calibrated_sigmoid(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply overflow-safe logistic sigmoid after temperature scaling."""
        scaled = np.clip(logits / max(temperature, 1e-6), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-scaled))

    @classmethod
    def evaluate(
        cls,
        splits: Dict[str, pd.DataFrame],
        drug_vectors: Dict[str, np.ndarray],
        protein_vectors: Dict[str, np.ndarray],
        reference_auroc: float,
        output_dir: str,
        seeds: int = 5,
    ) -> Dict[str, Any]:
        """Train C1 across seeds, compute CEF, and persist seed-0 coarse logits."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_tr, p_tr, y_tr = CoarseSeedFitter.build_split_tensors(splits["train"], drug_vectors, protein_vectors, device)
        d_va, p_va, y_va = CoarseSeedFitter.build_split_tensors(splits["validation"], drug_vectors, protein_vectors, device)
        d_te, p_te, y_te = CoarseSeedFitter.build_split_tensors(splits["test"], drug_vectors, protein_vectors, device)

        runs: List[Dict[str, float]] = []
        temperatures: List[float] = []
        for seed in range(seeds):
            b_tr, b_va, b_te, tau, state = CoarseSeedFitter.fit_seed(d_tr, p_tr, y_tr, d_va, p_va, y_va, d_te, p_te, seed)
            temperatures.append(tau)
            va_prob = cls._calibrated_sigmoid(b_va, tau)
            te_prob = cls._calibrated_sigmoid(b_te, tau)
            threshold = MetricEvaluator.find_best_mcc_threshold(y_va, va_prob)
            runs.append(MetricEvaluator.compute_classification_metrics(y_te, te_prob, threshold))
            if seed == 0:
                os.makedirs(output_dir, exist_ok=True)
                np.savez(os.path.join(output_dir, "c1_seed0_logits.npz"), b_train=b_tr, b_val=b_va, b_test=b_te, tau_b=tau)
                torch.save(state, os.path.join(output_dir, "c1_seed0.pt"))

        mean_auroc = float(np.mean([r["auroc"] for r in runs]))
        cef = float((mean_auroc - 0.5) / max(reference_auroc - 0.5, 1e-6))
        summary = {
            "auroc": mean_auroc, "auroc_std": float(np.std([r["auroc"] for r in runs])),
            "auprc": float(np.mean([r["auprc"] for r in runs])),
            "acc": float(np.mean([r["accuracy"] for r in runs])),
            "mcc": float(np.mean([r["mcc"] for r in runs])),
            "tau_b_mean": float(np.mean(temperatures)), "reference_auroc": reference_auroc, "cef": cef,
        }
        print(f"[C1] AUROC={mean_auroc:.4f}+/-{summary['auroc_std']:.4f} AUPRC={summary['auprc']:.4f} ACC={summary['acc']:.4f} MCC={summary['mcc']:.4f} | CEF={cef:.3f}", flush=True)
        return summary
