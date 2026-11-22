"""Leak-free inductive SAR (drug->target) and Homology (target->drug) transfer."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class InductiveTransferBuilder:
    """Computes 10-D multi-temperature SAR and Homology similarity transfer over train positives."""

    @staticmethod
    def l2_normalize(mat: np.ndarray) -> np.ndarray:
        """L2-normalize each row safely."""
        return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)

    @classmethod
    def compute_for_split(
        cls,
        df: pd.DataFrame,
        tr_drugs: List[str],
        tr_prots: List[str],
        tr_d_mat: np.ndarray,
        tr_p_mat: np.ndarray,
        a_pos: np.ndarray,
        drug_vecs: Dict[str, np.ndarray],
        prot_vecs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute 10-D SAR/Homology features masking exact self (d'==d, p'==p) across all splits."""
        d_tr_map = {d: i for i, d in enumerate(tr_drugs)}
        p_tr_map = {p: i for i, p in enumerate(tr_prots)}
        q_drugs = sorted(set(df["drug"]))
        q_prots = sorted(set(df["protein"]))
        qd_map = {d: i for i, d in enumerate(q_drugs)}
        qp_map = {p: i for i, p in enumerate(q_prots)}

        s_d = cls.l2_normalize(np.stack([drug_vecs[d] for d in q_drugs])) @ tr_d_mat.T
        s_p = cls.l2_normalize(np.stack([prot_vecs[p] for p in q_prots])) @ tr_p_mat.T
        for qi, d in enumerate(q_drugs):
            if d in d_tr_map:
                s_d[qi, d_tr_map[d]] = -1e4
        for qi, p in enumerate(q_prots):
            if p in p_tr_map:
                s_p[qi, p_tr_map[p]] = -1e4

        row_qd = np.array([qd_map[d] for d in df["drug"]], dtype=np.int64)
        row_qp = np.array([qp_map[p] for p in df["protein"]], dtype=np.int64)
        row_td = np.array([d_tr_map.get(d, -1) for d in df["drug"]], dtype=np.int64)
        row_tp = np.array([p_tr_map.get(p, -1) for p in df["protein"]], dtype=np.int64)
        in_p, in_d = row_tp >= 0, row_td >= 0

        feats = np.zeros((len(df), 10), dtype=np.float32)
        for t_idx, tau in enumerate((0.05, 0.12, 0.25)):
            wd = np.exp((s_d - s_d.max(axis=1, keepdims=True)) / tau)
            wd /= np.clip(wd.sum(axis=1, keepdims=True), 1e-8, None)
            score_dp = wd @ a_pos
            feats[in_p, t_idx] = score_dp[row_qd[in_p], row_tp[in_p]]

            wp = np.exp((s_p - s_p.max(axis=1, keepdims=True)) / tau)
            wp /= np.clip(wp.sum(axis=1, keepdims=True), 1e-8, None)
            score_pd = wp @ a_pos.T
            feats[in_d, 3 + t_idx] = score_pd[row_qp[in_d], row_td[in_d]]

        pos_drugs_per_p = [np.where(a_pos[:, j] > 0.5)[0] for j in range(a_pos.shape[1])]
        pos_prots_per_d = [np.where(a_pos[i, :] > 0.5)[0] for i in range(a_pos.shape[0])]
        for i in range(len(df)):
            tp, td = int(row_tp[i]), int(row_td[i])
            if tp >= 0 and len(pos_drugs_per_p[tp]) > 0:
                sims = s_d[row_qd[i], pos_drugs_per_p[tp]]
                feats[i, 6] = float(sims.max())
                feats[i, 7] = float(np.sort(sims)[-min(5, len(sims)):].mean())
            if td >= 0 and len(pos_prots_per_d[td]) > 0:
                sims = s_p[row_qp[i], pos_prots_per_d[td]]
                feats[i, 8] = float(sims.max())
                feats[i, 9] = float(np.sort(sims)[-min(5, len(sims)):].mean())
        return feats
