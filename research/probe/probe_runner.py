"""Runs P0..P5 multi-modal signal probes across DrugBank standard and cold-start splits."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from research.probe.graph_signal_builder import LeakFreeGraphFeatureBuilder
from research.probe.probe_mlp import ProbeEvaluator


class SignalProbeRunner:
    """Evaluates progressive modality ablations against 3DICE Table 1 benchmarks."""

    PAPER_REF = {
        "db_std": {"accuracy": 0.8198, "ppv": 0.8086, "tpr": 0.8385, "f1": 0.8231, "mcc": 0.6403, "auroc": 0.8933, "auprc": 0.8852},
        "drug_cs": {"accuracy": 0.7474, "ppv": 0.7471, "tpr": 0.7434, "f1": 0.7445, "mcc": 0.4913, "auroc": 0.8195, "auprc": 0.8182},
        "prot_cs": {"accuracy": 0.7476, "ppv": 0.7737, "tpr": 0.7053, "f1": 0.7375, "mcc": 0.5048, "auroc": 0.8243, "auprc": 0.8297},
    }

    @staticmethod
    def _assemble(df: pd.DataFrame, vec_map: Dict[str, np.ndarray], col: str) -> np.ndarray:
        """Stack entity vectors aligned with split DataFrame rows."""
        return np.stack([vec_map[k] for k in df[col]]).astype(np.float32)

    @staticmethod
    def _l2(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a 1-D vector safely."""
        return vec / max(float(np.linalg.norm(vec)), 1e-8)

    @classmethod
    def run_split(
        cls,
        name: str,
        raw_splits: Dict[str, pd.DataFrame],
        p3d: Dict[str, np.ndarray],
        d3d: Dict[str, np.ndarray],
        esm2: Dict[str, np.ndarray],
        chem2d: Dict[str, np.ndarray],
        seeds: List[int],
    ) -> List[Dict[str, object]]:
        """Evaluate P0..P5 progressive feature arms on one benchmark split."""
        splits = {
            k: df[df["protein"].isin(p3d) & df["drug"].isin(d3d) & df["protein"].isin(esm2) & df["drug"].isin(chem2d)].reset_index(drop=True)
            for k, df in raw_splits.items()
        }
        p_both = {k: np.concatenate([p3d[k], esm2[k]]) for k in p3d if k in esm2}
        d_both = {k: np.concatenate([d3d[k], chem2d[k]]) for k in d3d if k in chem2d}
        p_sim = {k: np.concatenate([cls._l2(p3d[k]), cls._l2(esm2[k])]) for k in p_both}
        d_sim = {k: np.concatenate([cls._l2(d3d[k]), cls._l2(chem2d[k])]) for k in d_both}
        g_feats = LeakFreeGraphFeatureBuilder.build_split_graph_features(splits, d_sim, p_sim)

        order = ("train", "val", "test")
        y_arrs = [splits[s]["label"].values for s in order]
        z_pair = [np.zeros((len(splits[s]), 0), dtype=np.float32) for s in order]
        ind_pair = [g_feats[s][0] for s in order]
        full_pair = [np.concatenate([g_feats[s][0], g_feats[s][1]], axis=1) for s in order]

        arms = [
            ("P0_3DICE_pool", p3d, d3d, z_pair),
            ("P1_+ESM2_650M", p_both, d3d, z_pair),
            ("P2_+Chem2D", p3d, d_both, z_pair),
            ("P2b_Seq+Chem2D", esm2, chem2d, z_pair),
            ("P3_DualView", p_both, d_both, z_pair),
            ("P4_+SAR_Homol", p_both, d_both, ind_pair),
            ("P5_+OOF_Graph", p_both, d_both, full_pair),
        ]
        rows: List[Dict[str, object]] = [{"split": name, "arm": "3DICE_Paper", **cls.PAPER_REF[name]}]
        for arm_name, p_map, d_map, pair_x in arms:
            p_arrs = [cls._assemble(splits[s], p_map, "protein") for s in order]
            d_arrs = [cls._assemble(splits[s], d_map, "drug") for s in order]
            metrics = ProbeEvaluator.evaluate_arm(p_arrs, d_arrs, pair_x, y_arrs, seeds)
            rows.append({"split": name, "arm": arm_name, **metrics})
            print(
                f"  [{name} | {arm_name:14s}] AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
                f"MCC={metrics['mcc']:.4f} F1={metrics['f1']:.4f} ACC={metrics['accuracy']:.4f}",
                flush=True,
            )
        return rows
