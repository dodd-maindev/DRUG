#!/usr/bin/env python3
"""CLI tool rendering Oxford-style Table 1, Table 2 (Complexity), and Table 3 (Ablation)."""

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
from bench.omni.complexity_profiler import ComplexityProfiler
from bench.omni.complexity_table_renderer import ComplexityTableRenderer
from bench.omni.table_specs import PaperTableData


class TableRenderCli:
    """Loads summary and optional complexity JSON to produce all three paper tables."""

    @staticmethod
    def main() -> None:
        """Parse CLI arguments and render Table 1, Table 2, and Table 3 300-DPI PNGs."""
        ap = argparse.ArgumentParser(description="Render Table 1, Table 2, and Table 3 PNG figures.")
        ap.add_argument("--summary", default="omni3d_summary.json")
        ap.add_argument("--complexity", default="")
        ap.add_argument("--out-dir", default="figures")
        args = ap.parse_args()

        with open(args.summary, "r", encoding="utf-8") as handle:
            results = json.load(handle)["results"]
        complexity_payload = None
        if args.complexity and os.path.exists(args.complexity):
            with open(args.complexity, "r", encoding="utf-8") as handle:
                complexity_payload = json.load(handle)

        os.makedirs(args.out_dir, exist_ok=True)
        t1_path = os.path.join(args.out_dir, "table1_comparison.png")
        t2_path = os.path.join(args.out_dir, "table2_complexity.png")
        t3_path = os.path.join(args.out_dir, "table3_ablation.png")

        BooktabsTableRenderer.render(
            table_num="Table 1",
            caption="Performance comparison among benchmarks on different datasets (mean (std) over n = 5 seeds).",
            sections=PaperTableData.build_table1_sections(results),
            footnote="Note: Row 5 in each block (shaded blue) reports Omni3D-DTI (Ours). Bold denotes best mean per metric in each dataset; '—' marks KIBA pending.",
            out_path=t1_path,
        )
        default_dims = {"p_seq": 2560, "p_3d": 1024, "d_2d": 3263, "d_3d": 1024}
        ComplexityTableRenderer.render(
            out_path=t2_path,
            complexity_payload=complexity_payload,
            default_params_m=ComplexityProfiler.exact_param_count_m(default_dims, use_gate=True),
        )
        BooktabsTableRenderer.render(
            table_num="Table 3",
            caption="Structural ablation study of Omni3D-DTI across DrugBank standard and cold-start splits (n = 5 seeds).",
            sections=PaperTableData.build_table2_sections(results),
            footnote="Note: Shaded row marks the default Omni3D-DTI configuration; bold indicates the highest mean per column within each split.",
            out_path=t3_path,
            name_col_header="Variant",
        )
        print(f"[rendered] {t1_path}, {t2_path}, {t3_path}", flush=True)


if __name__ == "__main__":
    TableRenderCli.main()
