#!/usr/bin/env python3
"""CLI entrypoint for training and evaluating benchmark arms (A0..M1)."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class BenchmarkCli:
    """Parses training flags and executes MultiSeedBenchmarkRunner."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Register model arm, dataset split, cache, and hyperparameter flags."""
        parser = argparse.ArgumentParser(description="Train and evaluate benchmark arm A0..M1.")
        parser.add_argument("--arm", required=True)
        parser.add_argument("--train-csv", required=True)
        parser.add_argument("--val-csv", required=True)
        parser.add_argument("--test-csv", required=True)
        parser.add_argument("--protein-dir", required=True)
        parser.add_argument("--drug-dir", required=True)
        parser.add_argument("--ca-dir", required=True)
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--bundle-cache", default="/content/cache/entity_bundle.pt")
        parser.add_argument("--oof-dir", default="/content/cache/oof_c1")
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--track", choices=["A", "B"], default="A")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--c1-auroc", type=float, default=None)
        parser.add_argument("--overwrite", action="store_true")
        return parser

    @classmethod
    def run(cls) -> None:
        """Execute deterministic multi-seed benchmark training for one arm."""
        args = cls.build_parser().parse_args()
        from bench.training.multi_seed_runner import MultiSeedBenchmarkRunner
        from diagnostics.common import DiagnosticUtilities

        DiagnosticUtilities.configure_determinism()
        MultiSeedBenchmarkRunner.run(
            arm_name=args.arm, train_csv=args.train_csv, val_csv=args.val_csv, test_csv=args.test_csv,
            protein_dir=args.protein_dir, drug_dir=args.drug_dir, ca_dir=args.ca_dir,
            out_dir=args.out_dir, bundle_path=args.bundle_cache, oof_dir=args.oof_dir,
            seeds=args.seeds, track=args.track, epochs=args.epochs, c1_auroc=args.c1_auroc,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    BenchmarkCli.run()
