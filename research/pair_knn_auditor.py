"""Non-parametric Pair-kNN neighborhood lookup audit for DTI benchmarks."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from diagnostics.metrics import MetricEvaluator
from research.split_smiles_loader import SplitSmilesLoader


class PairKnnAuditor:
    """Evaluates non-parametric Pair-kNN lookup using drug Tanimoto x protein cosine."""

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Normalize rows of a 2-D float32 array to unit L2 norm."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return (matrix / norms).astype(np.float32)

    @classmethod
    def _score_multi_k(
        cls,
        eval_df: pd.DataFrame,
        train_df: pd.DataFrame,
        fps: Dict[str, np.ndarray],
        prot_vecs: Dict[str, np.ndarray],
        top_k_values: List[Optional[int]],
        chunk_size: int = 512,
    ) -> Dict[Optional[int], np.ndarray]:
        """Compute Pair-kNN scores for all K settings in one GPU/CPU matrix pass."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_fp = torch.from_numpy(np.stack([fps[d] for d in train_df["drug"]])).to(device)
        tr_fp_sum = tr_fp.sum(dim=1, keepdim=True).T
        tr_pr = torch.from_numpy(cls._l2_normalize(np.stack([prot_vecs[p] for p in train_df["protein"]]))).to(device)
        tr_y = torch.tensor(train_df["label"].values, dtype=torch.float32, device=device)

        ev_fp_all = np.stack([fps[d] for d in eval_df["drug"]])
        ev_pr_all = cls._l2_normalize(np.stack([prot_vecs[p] for p in eval_df["protein"]]))
        results = {k: np.zeros(len(eval_df), dtype=np.float32) for k in top_k_values}

        with torch.no_grad():
            for start in range(0, len(eval_df), chunk_size):
                end = min(start + chunk_size, len(eval_df))
                ev_fp = torch.from_numpy(ev_fp_all[start:end]).to(device)
                ev_pr = torch.from_numpy(ev_pr_all[start:end]).to(device)
                inter = ev_fp @ tr_fp.T
                sim_drug = inter / (ev_fp.sum(dim=1, keepdim=True) + tr_fp_sum - inter + 1e-8)
                weights = sim_drug * torch.clamp(ev_pr @ tr_pr.T, min=0.0, max=1.0)
                for top_k in top_k_values:
                    if top_k is not None and top_k < weights.shape[1]:
                        top_w, idx = torch.topk(weights, top_k, dim=1)
                        score = (top_w * tr_y[idx]).sum(dim=1) / (top_w.sum(dim=1) + 1e-8)
                    else:
                        score = (weights @ tr_y) / (weights.sum(dim=1) + 1e-8)
                    results[top_k][start:end] = score.cpu().numpy()
        return results

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
        fps = SplitSmilesLoader.build_ecfp4_map(SplitSmilesLoader.build_drug_smiles_dict([tr, va, te]))

        filter_valid = lambda df: df[[d in fps and p in prot_vecs for d, p in zip(df.drug, df.protein)]].reset_index(drop=True)  # noqa: E731
        tr, va, te = filter_valid(tr), filter_valid(va), filter_valid(te)
        print(f"[pair-knn] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

        va_scores = cls._score_multi_k(va, tr, fps, prot_vecs, top_k_values)
        te_scores = cls._score_multi_k(te, tr, fps, prot_vecs, top_k_values)
        rows: List[Dict[str, Any]] = []
        for top_k in top_k_values:
            thr = MetricEvaluator.find_best_mcc_threshold(va["label"].values, va_scores[top_k])
            scores = MetricEvaluator.compute_classification_metrics(te["label"].values, te_scores[top_k], thr)
            rows.append({
                "model": f"pair_knn(k={'all' if top_k is None else top_k})",
                "auroc": scores["auroc"], "auprc": scores["auprc"],
                "acc": scores["accuracy"], "mcc": scores["mcc"],
            })
        return rows
