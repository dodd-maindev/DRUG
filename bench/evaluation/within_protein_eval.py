"""H-A1 evaluation: within-protein ranking AUROC/AUPRC and paired bootstrap."""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class WithinProteinEvaluator:
    """Computes per-protein AUROC/AUPRC across proteins with >=1 pos and >=1 neg."""

    @staticmethod
    def per_protein_scores(
        proteins: np.ndarray, labels: np.ndarray, scores: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Compute (AUROC, AUPRC) keyed by UniProt ID for mixed-label proteins."""
        results: Dict[str, Tuple[float, float]] = {}
        for pid in sorted(set(proteins.tolist())):
            mask = proteins == pid
            y = labels[mask]
            if (y == 1).any() and (y == 0).any():
                s = scores[mask]
                results[pid] = (float(roc_auc_score(y, s)), float(average_precision_score(y, s)))
        return results

    @classmethod
    def evaluate_preds(cls, npz_paths: List[str]) -> Dict[str, Any]:
        """Average within-protein AUROC/AUPRC across seed prediction files."""
        seed_aurocs, seed_auprcs, seed_r_aurocs = [], [], []
        merged_prot_scores: Dict[str, List[float]] = {}
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            scores = cls.per_protein_scores(data["protein"], data["label"], data["prob"])
            seed_aurocs.append(float(np.mean([v[0] for v in scores.values()])))
            seed_auprcs.append(float(np.mean([v[1] for v in scores.values()])))
            for pid, (auc_val, _) in scores.items():
                merged_prot_scores.setdefault(pid, []).append(auc_val)
            if "r" in data and np.any(data["r"] != data["logit"]):
                r_scores = cls.per_protein_scores(data["protein"], data["label"], data["r"])
                seed_r_aurocs.append(float(np.mean([v[0] for v in r_scores.values()])))

        per_protein_mean = {pid: float(np.mean(vals)) for pid, vals in merged_prot_scores.items()}
        summary = {
            "n_proteins": len(per_protein_mean),
            "within_auroc": float(np.mean(seed_aurocs)),
            "within_auroc_std": float(np.std(seed_aurocs)),
            "within_auprc": float(np.mean(seed_auprcs)),
            "per_protein_auroc": per_protein_mean,
        }
        if seed_r_aurocs:
            summary["within_auroc_r_only"] = float(np.mean(seed_r_aurocs))
        return summary

    @staticmethod
    def paired_bootstrap(
        scores_a: Dict[str, float], scores_b: Dict[str, float], n_boot: int = 2000, seed: int = 42
    ) -> Dict[str, float]:
        """Compute paired bootstrap difference (A - B), 95% CI, and two-sided p-value."""
        common = sorted(set(scores_a) & set(scores_b))
        if not common:
            return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
        diffs = np.array([scores_a[p] - scores_b[p] for p in common], dtype=np.float64)
        rng = np.random.default_rng(seed)
        boots = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
        p_val = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
        return {
            "delta": float(diffs.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_value": min(1.0, p_val),
        }
