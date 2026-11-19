"""Degree-only shortcut baseline evaluator for DTI splits."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from diagnostics.metrics import MetricEvaluator


class DegreeBaselineEvaluator:
    """Trains degree-only HistGradientBoosting baselines to detect popularity bias."""

    @staticmethod
    def _build_counts(dataframe: pd.DataFrame) -> Tuple[dict, dict]:
        """Count positive and negative training occurrences per drug and protein."""
        drug_counts = defaultdict(lambda: [0, 0])
        protein_counts = defaultdict(lambda: [0, 0])
        for drug_id, protein_id, label in zip(dataframe["drug"], dataframe["protein"], dataframe["label"]):
            drug_counts[drug_id][label] += 1
            protein_counts[protein_id][label] += 1
        return drug_counts, protein_counts

    @staticmethod
    def _extract_degree_features(
        dataframe: pd.DataFrame,
        drug_counts: dict,
        protein_counts: dict,
        leave_one_out: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log1p positive/negative degree features with optional LOO subtraction."""
        def encode(entity_id: str, label: int, counts: dict) -> List[float]:
            pair_count = list(counts.get(entity_id, [0, 0]))
            if leave_one_out:
                pair_count[label] -= 1
            return [math.log1p(pair_count[0]), math.log1p(pair_count[1]), float(pair_count[0] + pair_count[1] > 0)]

        drug_features = np.array([encode(d, y, drug_counts) for d, y in zip(dataframe["drug"], dataframe["label"])])
        protein_features = np.array([encode(p, y, protein_counts) for p, y in zip(dataframe["protein"], dataframe["label"])])
        return drug_features, protein_features

    @classmethod
    def evaluate(cls, splits: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Fit degree-only classifiers on train and evaluate on test."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        train_df = splits["train"]
        drug_counts, protein_counts = cls._build_counts(train_df)
        print("\n=== degree structure of TRAIN list ===", flush=True)
        for label, counts in (("drug", drug_counts), ("protein", protein_counts)):
            table = pd.DataFrame([(v[0], v[1]) for v in counts.values()], columns=["neg", "pos"])
            rho = table["neg"].corr(table["pos"], method="spearman")
            print(f"{label}: n={len(table)}, mean pos={table.pos.mean():.2f}, mean neg={table.neg.mean():.2f}, spearman={rho:.3f}", flush=True)
        for split_name in ("validation", "test"):
            if split_name in splits:
                subset = splits[split_name]
                print(f"{split_name}: seen drugs={subset.drug.isin(set(train_df.drug)).mean():.3f}, seen proteins={subset.protein.isin(set(train_df.protein)).mean():.3f}", flush=True)

        train_feats = cls._extract_degree_features(train_df, drug_counts, protein_counts, leave_one_out=True)
        val_feats = cls._extract_degree_features(splits["validation"], drug_counts, protein_counts, leave_one_out=False) if "validation" in splits else None
        test_feats = cls._extract_degree_features(splits["test"], drug_counts, protein_counts, leave_one_out=False)

        selectors = {"drug_degree": lambda pair: pair[0], "protein_degree": lambda pair: pair[1], "both_degree": lambda pair: np.hstack(pair)}
        rows: List[Dict[str, Any]] = []
        for model_name, selector in selectors.items():
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
            clf.fit(selector(train_feats), train_df["label"].values)
            val_prob = clf.predict_proba(selector(val_feats))[:, 1] if val_feats is not None else None
            test_prob = clf.predict_proba(selector(test_feats))[:, 1]
            threshold = MetricEvaluator.find_best_mcc_threshold(splits["validation"]["label"].values, val_prob) if val_prob is not None else 0.5
            scores = MetricEvaluator.compute_classification_metrics(splits["test"]["label"].values, test_prob, threshold)
            rows.append({"model": model_name, "auroc": scores["auroc"], "auprc": scores["auprc"], "acc": scores["accuracy"], "mcc": scores["mcc"]})
        return rows
