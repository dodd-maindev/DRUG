"""Publication-grade Table 1 & Table 2 (mean +/- std) formatter for Omni3D-DTI."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np


class OmniReporter:
    """Aggregates 5-seed metrics and writes formatted Paper Table 1 & Table 2 outputs."""

    KEYS = ("accuracy", "ppv", "tpr", "f1", "mcc", "auroc", "auprc")
    PAPER_REFS: Dict[str, List[Dict[str, Any]]] = {
        "db_std": [
            {"arm": "MolTrans", "accuracy": (0.7780, 0.0063), "ppv": (0.7437, 0.0128), "tpr": (0.8723, 0.0202), "f1": (0.8027, 0.0025), "mcc": (0.5587, 0.0045), "auroc": (0.8650, 0.0031), "auprc": (0.8520, 0.0073)},
            {"arm": "DLM-DTI", "accuracy": (0.7906, 0.0016), "ppv": (0.7854, 0.0078), "tpr": (0.8000, 0.0154), "f1": (0.7920, 0.0042), "mcc": (0.5818, 0.0029), "auroc": (0.8651, 0.0035), "auprc": (0.8605, 0.0030)},
            {"arm": "MocFormer", "accuracy": (0.7972, 0.0021), "ppv": (0.7741, 0.0030), "tpr": (0.8463, 0.0045), "f1": (0.8081, 0.0027), "mcc": (0.5976, 0.0029), "auroc": (0.8816, 0.0019), "auprc": (0.8780, 0.0071)},
            {"arm": "3DICE (Paper)", "accuracy": (0.8198, 0.0040), "ppv": (0.8086, 0.0118), "tpr": (0.8385, 0.0100), "f1": (0.8231, 0.0018), "mcc": (0.6403, 0.0074), "auroc": (0.8933, 0.0021), "auprc": (0.8852, 0.0037)},
        ],
        "drug_cs": [{"arm": "3DICE (Paper)", "accuracy": (0.7474, 0.0058), "ppv": (0.7471, 0.0207), "tpr": (0.7434, 0.0351), "f1": (0.7445, 0.0101), "mcc": (0.4913, 0.0130), "auroc": (0.8195, 0.0044), "auprc": (0.8182, 0.0052)}],
        "prot_cs": [{"arm": "3DICE (Paper)", "accuracy": (0.7476, 0.0103), "ppv": (0.7737, 0.0218), "tpr": (0.7053, 0.0211), "f1": (0.7375, 0.0053), "mcc": (0.5048, 0.0095), "auroc": (0.8243, 0.0032), "auprc": (0.8297, 0.0035)}],
    }

    @classmethod
    def aggregate_seeds(cls, split: str, arm: str, seed_records: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute mean, std, formatted cells, and deltas vs 3DICE Paper baseline."""
        row: Dict[str, Any] = {"split": split, "arm": arm, "seeds": len(seed_records)}
        for key in cls.KEYS:
            vals = [float(r[key]) for r in seed_records]
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            row[key] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_fmt"] = f"{mu:.4f} ({sd:.4f})"
        row["gate_p"] = float(np.mean([r["gate_p"] for r in seed_records]))
        row["gate_d"] = float(np.mean([r["gate_d"] for r in seed_records]))
        dice_ref = cls.PAPER_REFS[split][-1]
        row["delta_auroc"] = row["auroc"] - dice_ref["auroc"][0]
        row["delta_mcc"] = row["mcc"] - dice_ref["mcc"][0]
        return row

    @classmethod
    def _ref_fmt_row(cls, split: str, ref: Dict[str, Any]) -> List[str]:
        """Format one literature reference entry with zero delta vs 3DICE."""
        cells = [f"{ref[k][0]:.4f} ({ref[k][1]:.4f})" for k in cls.KEYS]
        return [split, ref["arm"], *cells, "-", "-"]

    @classmethod
    def build_markdown_tables(cls, rows: List[Dict[str, Any]], splits: List[str]) -> str:
        """Render publication-ready Markdown Table 1 (SOTA) and Table 2 (Ablations)."""
        hdr = "| Split | Model | ACC | PPV | TPR | F1 | MCC | AUROC | AUPRC | gate_p / gate_d |"
        sep = "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        lines = ["## Table 1: Paper Comparison (mean (std))", "", hdr, sep]
        for sp in splits:
            for ref in cls.PAPER_REFS.get(sp, []):
                r_cells = cls._ref_fmt_row(sp, ref)
                lines.append(f"| {' | '.join(r_cells[:-1])} | - |")
            for r in [x for x in rows if x["split"] == sp and x["arm"] == "Omni3D_DTI"]:
                c = [r[f"{k}_fmt"] for k in cls.KEYS]
                lines.append(f"| {sp} | **Omni3D_DTI** | {' | '.join(c)} | {r['gate_p']:.2f} / {r['gate_d']:.2f} |")

        lines.extend(["", "## Table 2: Structural Ablation Study (mean (std))", "", hdr, sep])
        for sp in splits:
            for r in [x for x in rows if x["split"] == sp]:
                c = [r[f"{k}_fmt"] for k in cls.KEYS]
                lines.append(f"| {sp} | {r['arm']} | {' | '.join(c)} | {r['gate_p']:.2f} / {r['gate_d']:.2f} |")
        return "\n".join(lines) + "\n"

    @classmethod
    def emit(cls, rows: List[Dict[str, Any]], splits: List[str], out_dir: str) -> None:
        """Print tables to stdout and save omni3d_summary.json and omni3d_tables.md."""
        os.makedirs(out_dir, exist_ok=True)
        md = cls.build_markdown_tables(rows, splits)
        print("\n" + md, flush=True)
        with open(os.path.join(out_dir, "omni3d_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"results": rows}, f, indent=2)
        with open(os.path.join(out_dir, "omni3d_tables.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[saved] {out_dir}/omni3d_summary.json & omni3d_tables.md", flush=True)
