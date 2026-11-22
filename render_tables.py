#!/usr/bin/env python3
"""CLI tool rendering publication-grade Table 1 and Table 2 PNGs from omni3d_summary.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.omni.booktabs_renderer import BooktabsTableRenderer
from bench.omni.table_specs import PaperTableData


class TableRenderCli:
    """Loads summary JSON and produces Table 1 comparison and Table 2 ablation figures."""

    @staticmethod
    def main() -> None:
        """Parse CLI arguments and render Table 1 & Table 2 300-DPI figures."""
        ap = argparse.ArgumentParser(description="Render Booktabs Table 1 and Table 2 figures from Omni3D JSON.")
        ap.add_argument("--summary", default="omni3d_summary.json")
        ap.add_argument("--out-dir", default="figures")
        args = ap.parse_args()

        with open(args.summary, "r", encoding="utf-8") as handle:
            results = json.load(handle)["results"]
        os.makedirs(args.out_dir, exist_ok=True)

        t1_path = os.path.join(args.out_dir, "table1_comparison.png")
        t2_path = os.path.join(args.out_dir, "table2_ablation.png")

        BooktabsTableRenderer.render(
            table_num="Table 1",
            caption="Performance comparison among benchmarks on different datasets (mean (std) over n = 5 seeds).",
            sections=PaperTableData.build_table1_sections(results),
            footnote="Note: Row 5 in each block (shaded blue) reports Omni3D-DTI (Ours). Bold denotes the best mean per metric in each dataset; '—' indicates KIBA pending.",
            out_path=t1_path,
        )
        BooktabsTableRenderer.render(
            table_num="Table 2",
            caption="Structural ablation study of Omni3D-DTI across DrugBank standard and cold-start splits (n = 5 seeds).",
            sections=PaperTableData.build_table2_sections(results),
            footnote="Note: Shaded row marks the default Omni3D-DTI configuration; bold indicates the highest mean per column within each split.",
            out_path=t2_path,
            name_col_header="Variant",
        )
        print(f"[rendered] {t1_path} & {t2_path}", flush=True)


if __name__ == "__main__":
    TableRenderCli.main()
