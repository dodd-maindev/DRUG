"""Non-parametric Pair-kNN neighborhood lookup audit for DTI benchmarks."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from diagnostics.metrics import MetricEvaluator
from research.split_smiles_loader import SplitSmilesLoader


class PairKnnAuditor:
    """Evaluates non-parametric Pair-kNN lookup using drug Tanimoto x protein cosine."""

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Normalize rows of a 2-D float32 array to unit L2 norm."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return (matrix / norms).astype(np.float32)

    @staticmethod
    def _tanimoto_matrix(query_fps: np.ndarray, train_fps: np.ndarray) -> np.ndarray:
        """Compute exact pairwise Tanimoto similarity matrix via matrix multiplication."""
        intersection = query_fps @ train_fps.T
        sum_query = query_fps.sum(axis=1, keepdims=True)
        sum_train = train_fps.sum(axis=1, keepdims=True).T
        return intersection / (sum_query + sum_train - intersection + 1e-8)

    @classmethod
    def _score_split(
        cls,
        eval_df: pd.DataFrame,
        train_df: pd.DataFrame,
        fps: Dict[str, np.ndarray],
        prot_vecs: Dict[str, np.ndarray],
        top_k: Optional[int],
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Compute Pair-kNN interaction probabilities for an evaluation split."""
        train_fps = np.stack([fps[d] for d in train_df["drug"]])
        train_prot = cls._l2_normalize(np.stack([prot_vecs[p] for p in train_df["protein"]]))
        train_labels = train_df["label"].values.astype(np.float32)

        eval_fps = np.stack([fps[d] for d in eval_df["drug"]])
        eval_prot = cls._l2_normalize(np.stack([prot_vecs[p] for p in eval_df["protein"]]))

        scores = np.zeros(len(eval_df), dtype=np.float32)
        for start in range(0, len(eval_df), chunk_size):
            end = min(start + chunk_size, len(eval_df))
            sim_drug = cls._tanimoto_matrix(eval_fps[start:end], train_fps)
            sim_prot = np.clip(eval_prot[start:end] @ train_prot.T, 0.0, 1.0)
            weights = sim_drug * sim_prot
            if top_k is not None and top_k < weights.shape[1]:
                idx = np.argpartition(weights, -top_k, axis=1)[:, -top_k:]
                rows = np.arange(end - start)[:, None]
                top_w = weights[rows, idx]
                top_y = train_labels[idx]
                scores[start:end] = (top_w * top_y).sum(axis=1) / (top_w.sum(axis=1) + 1e-8)
            else:
                scores[start:end] = (weights @ train_labels) / (weights.sum(axis=1) + 1e-8)
        return scores

    @classmethod
    def evaluate(
        cls,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        prot_vecs: Dict[str, np.ndarray],
        top_k_values: List[Optional[int]],
    ) -> List[Dict[str, Any]]:
        """Run Pair-kNN evaluation across top-K settings using val threshold tuning."""
        tr = SplitSmilesLoader.load_split(train_csv)
        va = SplitSmilesLoader.load_split(val_csv)
        te = SplitSmilesLoader.load_split(test_csv)
        smiles_map = SplitSmilesLoader.build_drug_smiles_dict([tr, va, te])
        fps = SplitSmilesLoader.build_ecfp4_map(smiles_map)

        filter_valid = lambda df: df[[d in fps and p in prot_vecs for d, p in zip(df.drug, df.protein)]].reset_index(drop=True)  # noqa: E731
        tr, va, te = filter_valid(tr), filter_valid(va), filter_valid(te)
        print(f"[pair-knn] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

        rows: List[Dict[str, Any]] = []
        for top_k in top_k_values:
            val_prob = cls._score_split(va, tr, fps, prot_vecs, top_k)
            test_prob = cls._score_split(te, tr, fps, prot_vecs, top_k)
            thr = MetricEvaluator.find_best_mcc_threshold(va["label"].values, val_prob)
            scores = MetricEvaluator.compute_classification_metrics(te["label"].values, test_prob, thr)
            rows.append({
                "model": f"pair_knn(k={'all' if top_k is None else top_k})",
                "auroc": scores["auroc"], "auprc": scores["auprc"],
                "acc": scores["accuracy"], "mcc": scores["mcc"],
            })
        return rows
