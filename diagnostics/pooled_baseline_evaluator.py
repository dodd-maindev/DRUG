"""Pooled-embedding single-modality and pair MLP baselines for shortcut testing."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from diagnostics.embedding_store import EmbeddingStore
from diagnostics.metrics import MetricEvaluator
from diagnostics.pooled_feature_cache import PooledFeatureCache


class PooledBaselineEvaluator:
    """Evaluates drug-only, protein-only, and pooled-pair MLPs across seeds."""

    @staticmethod
    def _train_mlp(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        seed: int,
        device: torch.device,
        epochs: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train a standardized 2-layer MLP selecting best checkpoint by val AUROC."""
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(seed)
        mean = x_train.mean(0, keepdims=True)
        std = x_train.std(0, keepdims=True) + 1e-6
        norm = lambda arr: torch.tensor((arr - mean) / std, dtype=torch.float32, device=device)  # noqa: E731
        tr_t, va_t, te_t = norm(x_train), norm(x_val), norm(x_test)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        net = nn.Sequential(nn.Linear(x_train.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        best_auc, best_val, best_test = -1.0, None, None
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(tr_t), device=device)
            for start in range(0, len(perm), 256):
                idx = perm[start: start + 256]
                loss = nn.functional.binary_cross_entropy_with_logits(net(tr_t[idx]).squeeze(1), y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vp = torch.sigmoid(net(va_t).squeeze(1)).cpu().numpy()
                tp = torch.sigmoid(net(te_t).squeeze(1)).cpu().numpy()
            auc = roc_auc_score(y_val, vp)
            if auc > best_auc:
                best_auc, best_val, best_test = auc, vp, tp
        return best_val, best_test

    @classmethod
    def evaluate(cls, splits: Dict[str, pd.DataFrame], store: EmbeddingStore, out_dir: str, trim: bool, seeds: int) -> List[Dict[str, Any]]:
        """Run pooled drug-only, protein-only, and pair MLP baselines."""
        if "validation" not in splits:
            print("[pooled] need --val-csv; skipped", flush=True)
            return []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tag = "trim" if trim else "notrim"
        proteins = sorted(set(pd.concat([s["protein"] for s in splits.values()])))
        drugs = sorted(set(pd.concat([s["drug"] for s in splits.values()])))
        pv = PooledFeatureCache.load_or_compute(store, "protein", proteins, os.path.join(out_dir, f"pooled_protein_{tag}.npz"))
        dv = PooledFeatureCache.load_or_compute(store, "drug", drugs, os.path.join(out_dir, f"pooled_drug_{tag}.npz"))
        to_mats = lambda df: (lambda sub: (np.stack([dv[d] for d in sub.drug]), np.stack([pv[p] for p in sub.protein]), sub.label.values))(df[[d in dv and p in pv for d, p in zip(df.drug, df.protein)]])  # noqa: E731
        d_tr, p_tr, y_tr = to_mats(splits["train"])
        d_va, p_va, y_va = to_mats(splits["validation"])
        d_te, p_te, y_te = to_mats(splits["test"])
        variants = {"drug_only(pooled)": lambda d, p: d, "protein_only(pooled)": lambda d, p: p, "pair_concat(pooled)": lambda d, p: np.hstack([d, p]), "pair_concat+prod(pooled)": lambda d, p: np.hstack([d, p, d * p])}
        rows: List[Dict[str, Any]] = []
        for name, fn in variants.items():
            runs = []
            for seed in range(seeds):
                vp, tp = cls._train_mlp(fn(d_tr, p_tr), y_tr, fn(d_va, p_va), y_va, fn(d_te, p_te), seed, device)
                runs.append(MetricEvaluator.compute_classification_metrics(y_te, tp, MetricEvaluator.find_best_mcc_threshold(y_va, vp)))
            rows.append({"model": name, **{k: float(np.mean([r[k if k != "acc" else "accuracy"] for r in runs])) for k in ("auroc", "auprc", "acc", "mcc")}})
        return rows
