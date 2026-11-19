"""Binary classification metrics, threshold selection, and bootstrap intervals."""

from typing import Dict, Tuple

import numpy as np


class MetricEvaluator:
    """Computes AUROC, AUPRC, accuracy, MCC, and bootstrap intervals."""

    @staticmethod
    def compute_classification_metrics(
        labels: np.ndarray,
        probabilities: np.ndarray,
        threshold: float | None = None,
    ) -> Dict[str, float | int]:
        """Compute AUROC, AUPRC, accuracy, and MCC for binary predictions."""
        from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

        targets = np.asarray(labels)
        scores = np.asarray(probabilities)
        result: Dict[str, float | int] = {
            "count": int(len(targets)),
            "positive_rate": float(targets.mean()) if len(targets) else float("nan"),
        }
        if len(set(targets.tolist())) < 2:
            result.update(auroc=float("nan"), auprc=float("nan"), accuracy=float("nan"), mcc=float("nan"))
            return result
        cutoff = 0.5 if threshold is None else float(threshold)
        predictions = (scores >= cutoff).astype(int)
        result.update(
            auroc=float(roc_auc_score(targets, scores)),
            auprc=float(average_precision_score(targets, scores)),
            accuracy=float((predictions == targets).mean()),
            mcc=float(matthews_corrcoef(targets, predictions)),
            threshold=cutoff,
        )
        return result

    @staticmethod
    def find_best_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
        """Select the decision threshold maximizing Matthews correlation coefficient."""
        from sklearn.metrics import matthews_corrcoef

        targets = np.asarray(labels)
        scores = np.asarray(probabilities)
        candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
        best_score = -2.0
        best_threshold = 0.5
        for candidate in candidates:
            score = matthews_corrcoef(targets, (scores >= candidate).astype(int))
            if score > best_score:
                best_score = score
                best_threshold = float(candidate)
        return best_threshold

    @staticmethod
    def bootstrap_auroc_interval(
        labels: np.ndarray,
        probabilities: np.ndarray,
        iterations: int = 300,
        seed: int = 0,
    ) -> Tuple[float, float]:
        """Compute a 95% bootstrap confidence interval for AUROC."""
        from sklearn.metrics import roc_auc_score

        generator = np.random.default_rng(seed)
        targets = np.asarray(labels)
        scores = np.asarray(probabilities)
        if len(targets) < 30 or len(set(targets.tolist())) < 2:
            return (float("nan"), float("nan"))
        samples = []
        for _ in range(iterations):
            indices = generator.integers(0, len(targets), len(targets))
            if len(set(targets[indices].tolist())) < 2:
                continue
            samples.append(roc_auc_score(targets[indices], scores[indices]))
        if not samples:
            return (float("nan"), float("nan"))
        return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
