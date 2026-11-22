"""Oxford Bioinformatics booktabs-style 300-DPI PNG table renderer."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class BooktabsTableRenderer:
    """Renders structured Table 1 and Table 2 benchmark sections to crisp PNG images."""

    COL_X = [0.010, 0.235, 0.345, 0.455, 0.565, 0.675, 0.785, 0.895]
    HEADERS = ("Method", "ACC", "PPV", "TPR", "F1", "MCC", "AUROC", "AUPRC")

    @staticmethod
    def _section_maxima(rows: List[Dict[str, Any]]) -> List[float]:
        """Compute rounded 4-decimal maximum per metric column within a dataset section."""
        maxima: List[float] = []
        for c_idx in range(7):
            vals = [round(r["vals"][c_idx], 4) for r in rows if not math.isnan(r["vals"][c_idx])]
            maxima.append(max(vals) if vals else math.nan)
        return maxima

    @classmethod
    def render(
        cls,
        table_num: str,
        caption: str,
        sections: List[Dict[str, Any]],
        footnote: str,
        out_path: str,
        name_col_header: str = "",
    ) -> None:
        """Render a publication-grade horizontal-rule table to a 300-DPI PNG file."""
        n_rows = 1 + sum(1 + len(sec["rows"]) for sec in sections)
        fig_w, fig_h = 14.4, max(4.2, 1.35 + n_rows * 0.255)
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, float(n_rows) + 2.15)
        ax.axis("off")

        y_title = float(n_rows) + 1.55
        ax.text(0.008, y_title, f"{table_num}  ", fontsize=10.5, fontweight="bold", va="center", color="#111111")
        ax.text(0.068, y_title, caption, fontsize=10.2, va="center", color="#222222")

        y_top = float(n_rows) + 1.05
        y_hdr = float(n_rows) + 0.55
        y_mid = float(n_rows) + 0.05
        ax.plot([0.004, 0.996], [y_top, y_top], color="#111111", lw=1.35)
        ax.plot([0.004, 0.996], [y_mid, y_mid], color="#111111", lw=0.90)

        headers = (name_col_header,) + cls.HEADERS[1:]
        ax.text(cls.COL_X[0], y_hdr, headers[0], fontsize=9.5, fontweight="bold", ha="left", va="center", color="#111111")
        for c_idx, h in enumerate(headers[1:], start=1):
            ax.text(cls.COL_X[c_idx], y_hdr, h, fontsize=9.5, fontweight="bold", ha="center", va="center", color="#111111")

        cur_y = float(n_rows) - 0.45
        for sec_i, sec in enumerate(sections):
            if sec_i > 0:
                ax.plot([0.004, 0.996], [cur_y + 0.48, cur_y + 0.48], color="#E2E8F0", lw=0.65)
            ax.text(0.008, cur_y, sec["title"], fontsize=9.4, fontweight="bold", ha="left", va="center", color="#0F172A")
            cur_y -= 1.0
            maxima = cls._section_maxima(sec["rows"])
            for row in sec["rows"]:
                is_ours = bool(row["accent"])
                if is_ours:
                    rect = Rectangle((0.004, cur_y - 0.46), 0.992, 0.92, facecolor="#EDF4FF", edgecolor="none", zorder=0)
                    ax.add_patch(rect)
                lbl_color = "#0B3B8C" if is_ours else "#1E293B"
                ax.text(0.024, cur_y, row["label"], fontsize=9.1, fontweight=("bold" if is_ours else "normal"),
                        ha="left", va="center", color=lbl_color, zorder=2)
                for m_idx, text_val in enumerate(row["fmt"]):
                    val = row["vals"][m_idx]
                    is_best = (not math.isnan(val)) and (round(val, 4) >= maxima[m_idx] - 1e-9)
                    txt_color = "#0B3B8C" if (is_ours and is_best) else ("#0F172A" if is_best else "#334155")
                    ax.text(cls.COL_X[1 + m_idx], cur_y, text_val, fontsize=9.0,
                            fontweight=("bold" if is_best else "normal"), ha="center", va="center", color=txt_color, zorder=2)
                cur_y -= 1.0

        y_bot = cur_y + 0.52
        ax.plot([0.004, 0.996], [y_bot, y_bot], color="#111111", lw=1.35)
        ax.text(0.008, y_bot - 0.35, footnote, fontsize=8.1, color="#475569", ha="left", va="top")

        plt.tight_layout(pad=0.25)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
