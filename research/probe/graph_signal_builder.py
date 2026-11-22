"""Leak-free inductive transfer and 5-fold OOF bipartite SVD graph feature builder."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from research.probe.inductive_transfer import InductiveTransferBuilder


class LeakFreeGraphFeatureBuilder:
    """Combines inductive SAR/Homology transfer with 5-fold OOF bipartite graph features."""

    @staticmethod
    def _svd_scores(a_pos: np.ndarray, d_idx: np.ndarray, p_idx: np.ndarray, rank: int = 64) -> np.ndarray:
        """Compute normalized SVD dot-product, 3-hop path density, and log degree product."""
        out = np.zeros((len(d_idx), 4), dtype=np.float32)
        deg_d = a_pos.sum(axis=1, keepdims=True)
        deg_p = a_pos.sum(axis=0, keepdims=True)
        norm_a = a_pos / np.sqrt(np.clip(deg_d, 1.0, None) * np.clip(deg_p, 1.0, None))
        k = min(rank, min(norm_a.shape) - 1)
        u, s, vt = np.linalg.svd(norm_a, full_matrices=False)
        u_k = u[:, :k] * np.sqrt(s[:k])
        v_k = vt[:k, :].T * np.sqrt(s[:k])
        valid = (d_idx >= 0) & (p_idx >= 0)
        if valid.any():
            dv, pv = d_idx[valid], p_idx[valid]
            out[valid, 0] = (u_k[dv] * v_k[pv]).sum(axis=1)
            out[valid, 1] = ((u_k[dv] * s[:k]) * v_k[pv]).sum(axis=1)
            d_sim = norm_a[dv] @ norm_a.T
            out[valid, 2] = (d_sim * norm_a[:, pv].T).sum(axis=1)
            out[valid, 3] = np.log1p(deg_d[dv, 0] * deg_p[0, pv]) / 10.0
        return out

    @classmethod
    def build_split_graph_features(
        cls,
        splits: Dict[str, pd.DataFrame],
        drug_vecs: Dict[str, np.ndarray],
        prot_vecs: Dict[str, np.ndarray],
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Build (inductive_10d, oof_transductive_4d) matrices per split without leakage."""
        tr_df = splits["train"]
        tr_drugs = sorted(set(tr_df["drug"]))
        tr_prots = sorted(set(tr_df["protein"]))
        d_map = {d: i for i, d in enumerate(tr_drugs)}
        p_map = {p: i for i, p in enumerate(tr_prots)}
        a_pos = np.zeros((len(tr_drugs), len(tr_prots)), dtype=np.float32)
        for d, p, y in zip(tr_df["drug"], tr_df["protein"], tr_df["label"]):
            if int(y) == 1:
                a_pos[d_map[d], p_map[p]] = 1.0

        tr_d_mat = InductiveTransferBuilder.l2_normalize(np.stack([drug_vecs[d] for d in tr_drugs]))
        tr_p_mat = InductiveTransferBuilder.l2_normalize(np.stack([prot_vecs[p] for p in tr_prots]))

        rng = np.random.default_rng(seed)
        tr_d_idx = np.array([d_map[d] for d in tr_df["drug"]], dtype=np.int64)
        tr_p_idx = np.array([p_map[p] for p in tr_df["protein"]], dtype=np.int64)
        fold_ids = rng.integers(0, 5, size=len(tr_df))
        tr_trans = np.zeros((len(tr_df), 4), dtype=np.float32)
        for fold in range(5):
            mask_f = fold_ids == fold
            a_sub = a_pos.copy()
            for idx in np.where(mask_f & (tr_df["label"].values == 1))[0]:
                a_sub[tr_d_idx[idx], tr_p_idx[idx]] = 0.0
            tr_trans[mask_f] = cls._svd_scores(a_sub, tr_d_idx[mask_f], tr_p_idx[mask_f])

        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for split_name, df in splits.items():
            d_idx = np.array([d_map.get(d, -1) for d in df["drug"]], dtype=np.int64)
            p_idx = np.array([p_map.get(p, -1) for p in df["protein"]], dtype=np.int64)
            ind = InductiveTransferBuilder.compute_for_split(
                df, tr_drugs, tr_prots, tr_d_mat, tr_p_mat, a_pos, drug_vecs, prot_vecs
            )
            trans = tr_trans if split_name == "train" else cls._svd_scores(a_pos, d_idx, p_idx)
            results[split_name] = (ind, trans)
        return results
