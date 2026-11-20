"""Multi-seed benchmark runner handling caching, OOF C1 generation, and reporting."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from bench.data.entity_bundle_cache import EntityBundleCache
from bench.data.pair_dataset import BenchmarkPairDataset
from bench.data.track_b_builder import TrackBBuilder
from bench.models.coarse_oof_helper import CoarseOofGenerator
from bench.training.arm_trainer import ArmTrainer
from bench.training.train_config import ArmRegistry
from diagnostics.common import DiagnosticUtilities
from diagnostics.dataset_reader import PairDatasetReader


class MultiSeedBenchmarkRunner:
    """Runs multi-seed training and persistence for any experimental arm."""

    @staticmethod
    def _aggregate(seed_metrics: List[Dict[str, float]], r_aurocs: List[float], ref_auroc: Optional[float]) -> Dict[str, Any]:
        """Compute mean and standard deviation across random seeds."""
        summary: Dict[str, Any] = {"seeds": len(seed_metrics)}
        for key in ("auroc", "auprc", "accuracy", "mcc"):
            vals = [m[key] for m in seed_metrics]
            summary[key] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
        if r_aurocs:
            summary["r_only_auroc"] = float(np.mean(r_aurocs))
        if ref_auroc is not None:
            summary["cef_vs_c1"] = float((ref_auroc - 0.5) / max(summary["auroc"] - 0.5, 1e-6))
        return summary

    @classmethod
    def run(
        cls, arm_name: str, train_csv: str, val_csv: str, test_csv: str,
        protein_dir: str, drug_dir: str, ca_dir: Optional[str], out_dir: str,
        bundle_path: Optional[str], oof_dir: str, seeds: List[int],
        track: str = "A", epochs: Optional[int] = None, c1_auroc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Train specified seeds on Track A or Track B and save checkpoints + predictions."""
        cfg = ArmRegistry.get(arm_name, epochs_override=epochs)
        splits = PairDatasetReader().load_splits(train_csv, val_csv, test_csv)
        if track.upper() == "B":
            splits = TrackBBuilder.convert_splits(splits, seed=42)

        entities = EntityBundleCache.load_or_build(
            [splits["train"], splits["validation"], splits["test"]], protein_dir, drug_dir, ca_dir, bundle_path
        )
        os.makedirs(out_dir, exist_ok=True)
        seed_metrics, r_aurocs = [], []

        for seed in seeds:
            coarse_payload = None
            b_train = None
            if cfg.family == "delta":
                oof_path = os.path.join(oof_dir, f"oof_c1_track{track.upper()}_seed{seed}.pt")
                coarse_payload = CoarseOofGenerator.compute_or_load(
                    splits["train"], splits["validation"], splits["test"], entities, seed, oof_path
                )
                b_train = coarse_payload["b_train"]

            tr_ds = BenchmarkPairDataset(splits["train"], entities, cfg.trim_protein_token, b_train)
            va_ds = BenchmarkPairDataset(splits["validation"], entities, cfg.trim_protein_token)
            te_ds = BenchmarkPairDataset(splits["test"], entities, cfg.trim_protein_token)
            result, state = ArmTrainer.train_seed(cfg, tr_ds, va_ds, te_ds, coarse_payload, seed)
            seed_metrics.append(result["metrics"])

            if cfg.family == "delta":
                from sklearn.metrics import roc_auc_score
                r_prob = 1.0 / (1.0 + np.exp(-np.clip(result["test"]["r"], -60.0, 60.0)))
                r_aurocs.append(float(roc_auc_score(result["test"]["labels"], r_prob)))

            torch.save(state, os.path.join(out_dir, f"ckpt_seed{seed}.pt"))
            np.savez(
                os.path.join(out_dir, f"preds_seed{seed}.npz"),
                protein=te_ds.df["protein"].values, drug=te_ds.df["drug"].values,
                label=result["test"]["labels"], prob=result["test"]["probs"],
                logit=result["test"]["logits"], r=result["test"]["r"], b=result["test"]["b"],
                threshold=result["threshold"],
            )

        summary = cls._aggregate(seed_metrics, r_aurocs, c1_auroc)
        summary["arm"] = arm_name
        summary["track"] = track.upper()
        DiagnosticUtilities.save_json(summary, out_dir, "summary.json")
        print(f"[{arm_name} Track {track.upper()}] AUROC={summary['auroc']:.4f}+/-{summary['auroc_std']:.4f} "
              f"MCC={summary['mcc']:.4f}+/-{summary['mcc_std']:.4f}", flush=True)
        return summary
