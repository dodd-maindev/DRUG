"""Oxford Bioinformatics booktabs-style Table 2 (Computational Complexity) renderer."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ComplexityTableRenderer:
    """Renders Table 2 comparing trainable parameters, peak memory, forward time, and epoch time."""

    COL_X = [0.012, 0.255, 0.445, 0.640, 0.855]
    BASELINES = [
        ("MolTrans", 235.0, 26366.0, 2.747, 450.0, ("235", "26 366", "2.747", "450")),
        ("DLM-DTI", 49.4, 5212.0, 1.961, 174.0, ("49.4", "5212", "1.961", "174")),
        ("MocFormer", 349.0, 15292.0, 0.185, 13.1, ("349", "15 292", "0.185", "13.1")),
        ("3DICE", 12.9, 1758.0, 0.051, 7.73, ("12.9", "1758", "0.051", "7.73")),
    ]

    @classmethod
    def _build_rows(cls, payload: Optional[Dict[str, Any]], default_params_m: float) -> List[Dict[str, Any]]:
        """Assemble Rows 1-4 (paper baselines) and Row 5 (Omni3D-DTI) for Table 2."""
        rows = [{"label": n, "vals": [p, m, f, e], "fmt": list(fmt), "accent": False} for n, p, m, f, e, fmt in cls.BASELINES]
        recs = {r["model"]: r for r in payload.get("results", [])} if payload else {}
        ours = recs.get("Omni3D_DTI")
        if ours:
            vals = [float(ours["params_m"]), float(ours["peak_mem_mb"]), float(ours["forward_ms"]), float(ours["epoch_s"])]
            fmt = [f"{vals[0]:.2f}", f"{vals[1]:.0f}", f"{vals[2]:.3f}", f"{vals[3]:.2f}"]
        else:
            vals, fmt = [float(default_params_m), math.nan, math.nan, math.nan], [f"{default_params_m:.2f}", "—", "—", "—"]
        rows.append({"label": "Omni3D-DTI (Ours)", "vals": vals, "fmt": fmt, "accent": True})
        return rows

    @staticmethod
    def _footnote(payload: Optional[Dict[str, Any]]) -> str:
        """Build accurate hardware footnote distinguishing paper RTX 5090 vs measured GPU."""
        recs = {r["model"]: r for r in payload.get("results", [])} if payload else {}
        ours, dice_gpu = recs.get("Omni3D_DTI"), recs.get("3DICE")
        if ours and dice_gpu:
            return (f"Rows 1–4 from paper (RTX 5090, BS=32). Row 5 measured on {ours['gpu']} (BS=32); "
                    f"on same {dice_gpu['gpu']}, 3DICE uses {dice_gpu['peak_mem_mb']:.0f} MB, "
                    f"{dice_gpu['forward_ms']:.3f} ms/sample, {dice_gpu['epoch_s']:.2f} s/epoch.")
        return "Rows 1–4 performed on RTX 5090 (BS=32). Row 5 reports exact trainable parameters; run profile_complexity.py on GPU for memory/timing."

    @classmethod
    def render(cls, out_path: str, complexity_payload: Optional[Dict[str, Any]] = None, default_params_m: float = 5.63) -> None:
        """Render publication-grade 300-DPI PNG for Table 2."""
        rows = cls._build_rows(complexity_payload, default_params_m)
        minima = [min(r["vals"][c] for r in rows if not math.isnan(r["vals"][c])) for c in range(4)]

        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(9.2, 4.1), dpi=300)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.35, 7.8)
        ax.axis("off")

        ax.text(0.010, 7.40, "Table 2  ", fontsize=10.5, fontweight="bold", va="center", color="#111111")
        ax.text(0.115, 7.40, "Computational complexity comparisons.", fontsize=10.2, va="center", color="#222222")
        ax.plot([0.005, 0.995], [7.00, 7.00], color="#111111", lw=1.35)
        ax.plot([0.005, 0.995], [5.45, 5.45], color="#111111", lw=0.90)

        headers = ["Model", "Trainable\nparameter\ncount (M)", "Peak\ntraining\nmemory (MB)", "Forward\npass time\n(ms)", "Training\ntime\n(epoch) (s)"]
        for idx, hdr in enumerate(headers):
            ax.text(cls.COL_X[idx], 6.22, hdr, fontsize=9.2, fontweight="bold", ha=("left" if idx == 0 else "center"), va="center", color="#111111", linespacing=1.18)

        cur_y = 4.92
        for row in rows:
            is_ours = row["accent"]
            if is_ours:
                ax.add_patch(Rectangle((0.005, cur_y - 0.42), 0.990, 0.84, facecolor="#EDF4FF", edgecolor="none", zorder=0))
            ax.text(cls.COL_X[0], cur_y, row["label"], fontsize=9.3, fontweight=("bold" if is_ours else "normal"),
                    ha="left", va="center", color=("#0B3B8C" if is_ours else "#1E293B"), zorder=2)
            for c_idx, txt in enumerate(row["fmt"]):
                val = row["vals"][c_idx]
                is_best = (not math.isnan(val)) and (val <= minima[c_idx] + 1e-9)
                col = "#0B3B8C" if (is_ours and is_best) else ("#0F172A" if is_best else "#334155")
                ax.text(cls.COL_X[1 + c_idx], cur_y, txt, fontsize=9.3, fontweight=("bold" if is_best else "normal"),
                        ha="center", va="center", color=col, zorder=2)
            cur_y -= 0.92

        y_bot = cur_y + 0.44
        ax.plot([0.005, 0.995], [y_bot, y_bot], color="#111111", lw=1.35)
        ax.text(0.010, y_bot - 0.25, cls._footnote(complexity_payload), fontsize=7.8, color="#475569", ha="left", va="top")

        plt.tight_layout(pad=0.25)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
