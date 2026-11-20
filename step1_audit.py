#!/usr/bin/env python3
"""CLI entrypoint for Step 1 audits: Pair-kNN, C1 CEF baseline, and PDB overlap."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class StepOneAuditCli:
    """Builds CLI subcommands and dispatches Step 1 audit checks."""

    @staticmethod
    def _add_common(parser: argparse.ArgumentParser) -> None:
        """Register split paths, directories, and cache arguments."""
        parser.add_argument("--train-csv", required=True)
        parser.add_argument("--val-csv", required=True)
        parser.add_argument("--test-csv", required=True)
        parser.add_argument("--protein-dir")
        parser.add_argument("--drug-dir")
        parser.add_argument("--struct-dir")
        parser.add_argument("--protein-npz")
        parser.add_argument("--drug-npz")
        parser.add_argument("--trim-protein", action="store_true", help="Default False (fixed Stage 1 protein length).")
        parser.add_argument("--out", default="step1_out")

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Create parser with pair-knn, c1-coarse, pdb-overlap, and all commands."""
        parser = argparse.ArgumentParser(description="Step 1 scientific audits for 3DICE.")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p_knn = sub.add_parser("pair-knn")
        cls._add_common(p_knn)

        p_c1 = sub.add_parser("c1-coarse")
        cls._add_common(p_c1)
        p_c1.add_argument("--ref-auroc", type=float, default=0.8933)
        p_c1.add_argument("--seeds", type=int, default=5)

        p_pdb = sub.add_parser("pdb-overlap")
        cls._add_common(p_pdb)
        p_pdb.add_argument("--no-rcsb", action="store_true")
        p_pdb.add_argument("--tanimoto", type=float, default=0.70)

        p_all = sub.add_parser("all")
        cls._add_common(p_all)
        p_all.add_argument("--ref-auroc", type=float, default=0.8933)
        p_all.add_argument("--seeds", type=int, default=5)
        p_all.add_argument("--no-rcsb", action="store_true")
        p_all.add_argument("--tanimoto", type=float, default=0.70)
        return parser

    @classmethod
    def run(cls) -> None:
        """Execute selected Step 1 audit routine."""
        args = cls.build_parser().parse_args()
        from diagnostics.common import DiagnosticUtilities
        from research.step1_runner import StepOneAuditRunner

        DiagnosticUtilities.configure_determinism()
        if args.cmd in ("pair-knn", "all"):
            StepOneAuditRunner.run_pair_knn(args)
        if args.cmd in ("c1-coarse", "all"):
            StepOneAuditRunner.run_c1_coarse(args)
        if args.cmd in ("pdb-overlap", "all"):
            StepOneAuditRunner.run_pdb_overlap(args)


if __name__ == "__main__":
    StepOneAuditCli.run()
