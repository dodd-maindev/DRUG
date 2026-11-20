#!/usr/bin/env python3
"""CLI entrypoint for extracting aligned C-alpha coordinates and pLDDT masks."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class CaExtractionCli:
    """Parses CLI arguments and runs CaCoordinateExtractor."""

    @staticmethod
    def run() -> None:
        """Execute C-alpha coordinate extraction and save summary JSON."""
        parser = argparse.ArgumentParser(description="Extract (L_p, 3) C-alpha coordinates aligned with ESM-IF1.")
        parser.add_argument("--protein-dir", required=True)
        parser.add_argument("--struct-dir", required=True)
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args()

        from diagnostics.common import DiagnosticUtilities
        from research.ca_coordinate_extractor import CaCoordinateExtractor

        stats = CaCoordinateExtractor.run(args.protein_dir, args.struct_dir, args.out_dir, args.limit)
        DiagnosticUtilities.save_json(stats, args.out_dir, "ca_extraction_summary.json")


if __name__ == "__main__":
    CaExtractionCli.run()
