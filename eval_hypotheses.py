#!/usr/bin/env python3
"""CLI entrypoint for computing the H-A1 through H-A5 scientific report."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class HypothesisEvalCli:
    """Parses evaluation arguments and runs HypothesisReporter."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Register paths for trained arm runs, splits, caches, and pocket CSV."""
        parser = argparse.ArgumentParser(description="Evaluate H-A1..H-A5 across saved arm runs.")
        parser.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "C2", "M1"])
        parser.add_argument("--runs-root", required=True)
        parser.add_argument("--train-csv", required=True)
        parser.add_argument("--val-csv", required=True)
        parser.add_argument("--test-csv", required=True)
        parser.add_argument("--protein-dir", required=True)
        parser.add_argument("--drug-dir", required=True)
        parser.add_argument("--ca-dir", required=True)
        parser.add_argument("--bundle-cache", default="/content/cache/entity_bundle.pt")
        parser.add_argument("--oof-dir", default="/content/cache/oof_c1")
        parser.add_argument("--overlap-csv", default=None)
        parser.add_argument("--track", choices=["A", "B"], default="A")
        parser.add_argument("--out-dir", required=True)
        return parser

    @classmethod
    def run(cls) -> None:
        """Execute cross-arm evaluation and write hypothesis_report.json."""
        args = cls.build_parser().parse_args()
        from bench.evaluation.hypothesis_reporter import HypothesisReporter
        from diagnostics.common import DiagnosticUtilities

        DiagnosticUtilities.configure_determinism()
        HypothesisReporter.evaluate_all(
            arms=args.arms, runs_root=args.runs_root, train_csv=args.train_csv,
            val_csv=args.val_csv, test_csv=args.test_csv, protein_dir=args.protein_dir,
            drug_dir=args.drug_dir, ca_dir=args.ca_dir, bundle_path=args.bundle_cache,
            oof_dir=args.oof_dir, overlap_csv=args.overlap_csv, track=args.track, out_dir=args.out_dir,
        )


if __name__ == "__main__":
    HypothesisEvalCli.run()
