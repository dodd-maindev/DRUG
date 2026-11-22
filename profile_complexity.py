#!/usr/bin/env python3
"""CLI tool profiling GPU memory and speed of 3DICE vs Omni3D-DTI at batch size 32."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ComplexityProfileCli:
    """Profiles 3DICE and Omni3D-DTI on identical hardware and renders Table 2."""

    @staticmethod
    def main() -> None:
        """Parse CLI flags, run GPU timing/memory benchmarks, and save complexity JSON."""
        ap = argparse.ArgumentParser(description="Profile GPU memory, forward ms, and epoch time at batch_size=32.")
        ap.add_argument("--list-dir", required=True)
        ap.add_argument("--protein-dir", required=True)
        ap.add_argument("--drug-dir", required=True)
        ap.add_argument("--bundle-cache", default="/content/cache/entity_bundle.pt")
        ap.add_argument("--cache-dir", required=True)
        ap.add_argument("--out-dir", required=True)
        ap.add_argument("--summary", default="omni3d_summary.json")
        ap.add_argument("--figures-dir", default="figures")
        ap.add_argument("--batch-size", type=int, default=32)
        args = ap.parse_args()

        import pandas as pd
        import torch

        from bench.data.entity_bundle_cache import EntityBundleCache
        from bench.data.pair_dataset import BenchmarkPairDataset
        from bench.omni.complexity_profiler import ComplexityProfiler
        from bench.omni.complexity_table_renderer import ComplexityTableRenderer
        from bench.omni.omni_config import OmniRegistry
        from bench.omni.omni_dataset import OmniFeaturePacker
        from diagnostics.dataset_reader import PairDatasetReader
        from research.probe.drug_chem2d_extractor import DrugChem2DExtractor
        from research.probe.esm2_extractor import Esm2FeatureExtractor
        from research.probe.pooled_3d_loader import Pooled3DLoader
        from research.probe.sequence_drug_collector import SplitMetadataCollector

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        benchmarks, seqs, smiles = SplitMetadataCollector.load_all_benchmarks(args.list_dir)
        p3d = Pooled3DLoader.load_or_build(seqs.keys(), smiles.keys(), args.protein_dir, args.drug_dir,
                                           args.bundle_cache or None, os.path.join(args.cache_dir, "pooled_3d.pt"))
        esm2 = Esm2FeatureExtractor.load_or_extract(seqs, os.path.join(args.cache_dir, "esm2_650m_pooled.pt"))
        chem = DrugChem2DExtractor.load_or_extract(smiles, os.path.join(args.cache_dir, "drug_chem2d.pt"))
        packed, dims = OmniFeaturePacker.pack_split(benchmarks["db_std"], p3d, esm2, chem)

        splits_3dice = PairDatasetReader().load_splits(
            os.path.join(args.list_dir, "db_train.csv"),
            os.path.join(args.list_dir, "db_val.csv"),
            os.path.join(args.list_dir, "db_test.csv"),
        )
        entities = EntityBundleCache.load_or_build(list(splits_3dice.values()), args.protein_dir, args.drug_dir, None, args.bundle_cache or None)
        tr_ds_3dice = BenchmarkPairDataset(splits_3dice["train"], entities, trim_protein_token=True)

        results = [
            ComplexityProfiler.profile_3dice(tr_ds_3dice, device, args.batch_size),
            ComplexityProfiler.profile_omni(OmniRegistry.get("Omni3D_DTI"), packed, dims, device, args.batch_size),
            ComplexityProfiler.profile_omni(OmniRegistry.get("Omni3D_wo_Gate"), packed, dims, device, args.batch_size),
        ]
        os.makedirs(args.out_dir, exist_ok=True)
        out_json = os.path.join(args.out_dir, "complexity_summary.json")
        payload = {"batch_size": args.batch_size, "results": results}
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        print("\n=== Computational Complexity Comparison ===", flush=True)
        print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
        os.makedirs(args.figures_dir, exist_ok=True)
        ComplexityTableRenderer.render(os.path.join(args.figures_dir, "table2_complexity.png"), payload)
        print(f"[saved] {out_json} & {args.figures_dir}/table2_complexity.png", flush=True)


if __name__ == "__main__":
    ComplexityProfileCli.main()
