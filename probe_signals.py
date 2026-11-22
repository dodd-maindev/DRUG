#!/usr/bin/env python3
"""CLI entrypoint probing ESM-2, Chem2D, SAR/Homology, and OOF graph signal ceilings."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SignalProbeCli:
    """Extracts reusable multi-modal caches and benchmarks P0..P5 across DrugBank splits."""

    DEFAULT_ESM2 = "facebook/esm2_t33_650M_UR50D"

    @classmethod
    def main(cls) -> None:
        """Parse CLI flags, cache ESM-2/Chem2D/3D features, and run all split probes."""
        ap = argparse.ArgumentParser(description="Probe multi-modal signal ceilings across DrugBank splits.")
        ap.add_argument("--list-dir", required=True)
        ap.add_argument("--protein-dir", required=True)
        ap.add_argument("--drug-dir", required=True)
        ap.add_argument("--bundle-cache", default="")
        ap.add_argument("--cache-dir", required=True)
        ap.add_argument("--out-dir", required=True)
        ap.add_argument("--esm2-model", default=cls.DEFAULT_ESM2)
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        args = ap.parse_args()

        import numpy as np
        import pandas as pd

        from diagnostics.common import DiagnosticUtilities
        from research.probe.drug_chem2d_extractor import DrugChem2DExtractor
        from research.probe.esm2_extractor import Esm2FeatureExtractor
        from research.probe.pooled_3d_loader import Pooled3DLoader
        from research.probe.probe_runner import SignalProbeRunner
        from research.probe.sequence_drug_collector import SplitMetadataCollector

        benchmarks, seqs, smiles = SplitMetadataCollector.load_all_benchmarks(args.list_dir)
        p3d_raw = Pooled3DLoader.load_or_build(
            seqs.keys(), smiles.keys(), args.protein_dir, args.drug_dir,
            args.bundle_cache or None, os.path.join(args.cache_dir, "pooled_3d.pt"),
        )
        esm2_raw = Esm2FeatureExtractor.load_or_extract(seqs, os.path.join(args.cache_dir, "esm2_650m_pooled.pt"), args.esm2_model)
        chem_raw = DrugChem2DExtractor.load_or_extract(smiles, os.path.join(args.cache_dir, "drug_chem2d.pt"))

        p3d = {k: v.float().numpy() for k, v in p3d_raw["proteins"].items()}
        d3d = {k: v.float().numpy() for k, v in p3d_raw["drugs"].items()}
        esm2 = {k: np.concatenate([v["mean"].float().numpy(), v["max"].float().numpy()]) for k, v in esm2_raw.items()}
        chem2d = {k: np.concatenate([v["fp"].float().numpy(), v["desc"].float().numpy()]) for k, v in chem_raw.items()}

        all_rows: List[Dict[str, object]] = []
        for split_name in ("db_std", "drug_cs", "prot_cs"):
            if split_name in benchmarks:
                print(f"\n=== Probing Split: {split_name} ===", flush=True)
                all_rows.extend(SignalProbeRunner.run_split(split_name, benchmarks[split_name], p3d, d3d, esm2, chem2d, args.seeds))

        df = pd.DataFrame(all_rows)
        cols = ["split", "arm", "accuracy", "ppv", "tpr", "f1", "mcc", "auroc", "auprc"]
        print("\n=== FULL SIGNAL PROBE SUMMARY (PAPER TABLE 1 METRICS) ===", flush=True)
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        DiagnosticUtilities.save_json({"results": all_rows}, args.out_dir, "probe_signals_summary.json")


if __name__ == "__main__":
    SignalProbeCli.main()
