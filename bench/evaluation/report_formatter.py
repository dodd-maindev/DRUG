"""Formatting and persistence helpers for Paper Table 1 and H-A1..H-A5 reports."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from diagnostics.common import DiagnosticUtilities


class ReportFormatter:
    """Formats Table 1 classification metrics and cross-arm hypothesis summaries."""

    @staticmethod
    def build_row(
        arm: str,
        base: Dict[str, Any],
        ha1: Dict[str, float],
        ha2: Dict[str, float],
        ha3: Dict[str, float],
        ha3_te: Dict[str, float],
    ) -> Dict[str, Any]:
        """Combine Table 1 summary metrics with H-A1, H-A2, and H-A3 diagnostics."""
        nan = float("nan")
        return {
            "arm": arm,
            "acc": base.get("accuracy", nan),
            "ppv": base.get("ppv", nan),
            "tpr": base.get("tpr", nan),
            "f1": base.get("f1", nan),
            "mcc": base["mcc"],
            "auroc": base["auroc"],
            "auprc": base["auprc"],
            "r_auroc": base.get("r_only_auroc", nan),
            "within_auroc": ha1["within_auroc"],
            "within_r": ha1.get("within_auroc_r_only", nan),
            "js_div": ha2["js_pos_neg_mean"],
            "spread_A": ha2["pos_spatial_spread_A"],
            "leak": ha2["pad_leak_max_abs"],
            "pocket_auprc": ha3.get("pocket_auprc", nan),
            "test_pocket_auprc": ha3_te.get("pocket_auprc", nan),
            "auprc_lift": ha3.get("auprc_lift", nan),
            "top10_hit": ha3.get("top10_hit", nan),
        }

    @staticmethod
    def emit_reports(rows: List[Dict[str, Any]], boot: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        """Print formatted Paper Table 1 and H-A1..H-A5 tables and save JSON report."""
        report = {"table": rows, "bootstrap_within_auroc": boot}
        if rows:
            df = pd.DataFrame(rows)
            t1_cols = ["arm", "acc", "ppv", "tpr", "f1", "mcc", "auroc", "auprc"]
            ha_cols = [
                "arm", "auroc", "mcc", "r_auroc", "within_auroc", "within_r",
                "js_div", "spread_A", "leak", "pocket_auprc", "test_pocket_auprc", "auprc_lift", "top10_hit",
            ]
            print("\n=== Paper Table 1 Metrics ===", flush=True)
            print(df[t1_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
            print("\n=== Cross-Arm Hypothesis Matrix (H-A1 .. H-A5) ===", flush=True)
            print(df[ha_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        if boot:
            print("\n=== Paired Bootstrap Within-Protein AUROC ===", boot, flush=True)
        DiagnosticUtilities.save_json(report, out_dir, "hypothesis_report.json")
        return report
