#!/usr/bin/env python3
"""CLI entrypoint for 5-seed Omni3D-DTI and full ablation experiments across splits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.omni.omni_config import OmniRegistry


class OmniBenchmarkCli:
    """Orchestrates cached feature loading, multi-seed training, resume, and reporting."""

    @classmethod
    def main(cls) -> None:
        """Parse CLI options and run Omni3D-DTI + ablations across selected DrugBank splits."""
        ap = argparse.ArgumentParser(description="Run 5-seed Omni3D-DTI and ablations on DrugBank splits.")
        ap.add_argument("--list-dir", required=True)
        ap.add_argument("--protein-dir", required=True)
        ap.add_argument("--drug-dir", required=True)
        ap.add_argument("--bundle-cache", default="")
        ap.add_argument("--cache-dir", required=True)
        ap.add_argument("--out-dir", required=True)
        ap.add_argument("--splits", nargs="+", default=["db_std", "drug_cs", "prot_cs"])
        ap.add_argument("--arms", nargs="+", default=OmniRegistry.ALL_ARMS)
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        ap.add_argument("--epochs", type=int, default=None)
        ap.add_argument("--overwrite", action="store_true")
        args = ap.parse_args()

        import numpy as np
        import torch

        from bench.omni.omni_dataset import OmniFeaturePacker
        from bench.omni.omni_reporter import OmniReporter
        from bench.omni.omni_trainer import OmniTrainer
        from diagnostics.metrics import MetricEvaluator
        from research.probe.drug_chem2d_extractor import DrugChem2DExtractor
        from research.probe.esm2_extractor import Esm2FeatureExtractor
        from research.probe.pooled_3d_loader import Pooled3DLoader
        from research.probe.sequence_drug_collector import SplitMetadataCollector

        benchmarks, seqs, smiles = SplitMetadataCollector.load_all_benchmarks(args.list_dir)
        p3d = Pooled3DLoader.load_or_build(seqs.keys(), smiles.keys(), args.protein_dir, args.drug_dir,
                                           args.bundle_cache or None, os.path.join(args.cache_dir, "pooled_3d.pt"))
        esm2 = Esm2FeatureExtractor.load_or_extract(seqs, os.path.join(args.cache_dir, "esm2_650m_pooled.pt"))
        chem = DrugChem2DExtractor.load_or_extract(smiles, os.path.join(args.cache_dir, "drug_chem2d.pt"))

        os.makedirs(args.out_dir, exist_ok=True)
        all_rows: List[Dict[str, Any]] = []
        active_splits = [s for s in args.splits if s in benchmarks]

        for split_name in active_splits:
            print(f"\n=== Packing Split: {split_name} ===", flush=True)
            packed, dims = OmniFeaturePacker.pack_split(benchmarks[split_name], p3d, esm2, chem)
            for arm_name in args.arms:
                cfg = OmniRegistry.get(arm_name, epochs_override=args.epochs)
                seed_records: List[Dict[str, float]] = []
                for seed in args.seeds:
                    npz_p = os.path.join(args.out_dir, f"preds_{split_name}_{arm_name}_seed{seed}.npz")
                    ckpt_p = os.path.join(args.out_dir, f"best_{split_name}_{arm_name}_seed0.pt")
                    if not args.overwrite and os.path.exists(npz_p):
                        z = np.load(npz_p)
                        m = MetricEvaluator.compute_classification_metrics(z["y"], z["prob"], float(z["threshold"]))
                        seed_records.append({**m, "gate_p": float(z["gate_p"]), "gate_d": float(z["gate_d"])})
                        continue
                    m, preds, state = OmniTrainer.train_seed(cfg, packed, dims, seed)
                    np.savez(npz_p, **preds)
                    if seed == 0:
                        torch.save(state, ckpt_p)
                    seed_records.append(m)
                row = OmniReporter.aggregate_seeds(split_name, arm_name, seed_records)
                all_rows.append(row)
                print(f"  [{split_name:7s} | {arm_name:18s}] AUROC={row['auroc_fmt']} AUPRC={row['auprc_fmt']} "
                      f"MCC={row['mcc_fmt']} F1={row['f1_fmt']} ACC={row['accuracy_fmt']} (gp={row['gate_p']:.2f}, gd={row['gate_d']:.2f})",
                      flush=True)

        OmniReporter.emit(all_rows, active_splits, args.out_dir)


if __name__ == "__main__":
    OmniBenchmarkCli.main()
