"""300-DPI Matplotlib diagram comparing BASE_3DICE vs IMPROVED Omni3D-DTI."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


class ArchitectureDiagramPlotter:
    """Renders a clean EPINTLM-inspired visual comparison of 3DICE vs Omni3D-DTI."""

    @staticmethod
    def _box(ax, x: float, y: float, w: float, h: float, title: str, sub: str, fc: str, ec: str, fs: float = 8.4) -> None:
        """Draw a rounded colored block with centered bold title and subtitle."""
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14", fc=fc, ec=ec, lw=1.6, zorder=2)
        ax.add_patch(patch)
        txt = f"{title}\n{sub}" if sub else title
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="medium", color="#0F172A", linespacing=1.28, zorder=3)

    @staticmethod
    def _arrow(ax, x1: float, y1: float, x2: float, y2: float, col: str = "#475569", dashed: bool = False) -> None:
        """Draw a solid or dashed directional arrow between blocks."""
        ls = "--" if dashed else "-"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.5, color=col, ls=ls), zorder=1)

    @classmethod
    def render(cls, out_path: str) -> None:
        """Render stacked BASE vs IMPROVED architectural figure to 300-DPI PNG."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(14.4, 8.2), dpi=300)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0.0, 14.4)
        ax.set_ylim(0.0, 8.4)
        ax.axis("off")

        # Panel A: BASE 3DICE
        ax.add_patch(FancyBboxPatch((0.2, 5.25), 14.0, 2.95, boxstyle="round,pad=0.05,rounding_size=0.18", fc="#FAFAFA", ec="#CBD5E1", lw=1.2, zorder=0))
        ax.text(0.45, 7.90, "(A) Kiến trúc Gốc: 3DICE", fontsize=10.8, fontweight="bold", color="#334155")
        ax.text(3.35, 7.90, "12.87M params | 1,851 MB VRAM | 61.36 s/epoch (NVIDIA L4) | AUROC 0.8933", fontsize=9.2, color="#64748B")
        cls._box(ax, 0.45, 6.72, 2.35, 0.86, "ESM-IF1 (chỉ tọa độ 3D)", "Bỏ qua seq! | trim :-1 sai", "#FEF3C7", "#D97706")
        cls._box(ax, 0.45, 5.52, 2.35, 0.86, "Uni-Mol (Conformer 3D)", "44.5% nguyên tử H ở đuôi", "#FEF3C7", "#D97706")
        cls._box(ax, 3.35, 6.72, 2.25, 0.86, "ProteinCNN (1D k=3)", "Không chặn mask khi tích chập", "#FEE2E2", "#DC2626")
        cls._box(ax, 3.35, 5.52, 2.25, 0.86, "DrugCNN (1D k=3)", "Rò padding max |Δ| = 0.737", "#FEE2E2", "#DC2626")
        cls._box(ax, 6.15, 6.02, 2.45, 1.08, "Bidirectional Cross-Attn", "4 heads, Softmax thiếu null sink\n+ FFN(2048)", "#EDE9FE", "#7C3AED")
        cls._box(ax, 9.15, 6.02, 2.30, 1.08, "FusionNew (MeanPool)", "1.05M tham số dead (Linear 1024)\nkhông được gọi trong forward", "#F1F5F9", "#64748B")
        cls._box(ax, 12.00, 6.08, 1.95, 0.96, "MLP2 + CrossEntropy", "512 -> 1024 -> 512 -> 2", "#E2E8F0", "#475569")
        cls._arrow(ax, 2.84, 7.15, 3.31, 7.15, "#D97706"); cls._arrow(ax, 2.84, 5.95, 3.31, 5.95, "#D97706")
        cls._arrow(ax, 5.64, 7.15, 6.11, 6.78, "#DC2626"); cls._arrow(ax, 5.64, 5.95, 6.11, 6.32, "#DC2626")
        cls._arrow(ax, 8.64, 6.56, 9.11, 6.56, "#7C3AED"); cls._arrow(ax, 11.49, 6.56, 11.96, 6.56, "#475569")

        # Panel B: Omni3D-DTI
        ax.add_patch(FancyBboxPatch((0.2, 0.82), 14.0, 4.18, boxstyle="round,pad=0.05,rounding_size=0.18", fc="#F8FBFF", ec="#93C5FD", lw=1.4, zorder=0))
        ax.text(0.45, 4.70, "(B) Kiến trúc Cải tiến: Omni3D-DTI", fontsize=10.8, fontweight="bold", color="#0B3B8C")
        ax.text(4.35, 4.70, "5.63M params (-2.3x) | 127 MB (-14.5x) | 9.60 s/epoch (-6.4x) | AUROC 0.9242 (+3.09%)", fontsize=9.2, fontweight="bold", color="#1D4ED8")
        cls._box(ax, 0.45, 3.80, 2.35, 0.64, "ESM-2 650M (Seq+AA)", "Mean+Max (2560-D, không cắt)", "#DBEAFE", "#0284C7", 8.1)
        cls._box(ax, 0.45, 2.84, 2.35, 0.64, "ESM-IF1 (3D Backbone)", "Mean+Max (1024-D, đủ chuỗi)", "#FEF3C7", "#D97706", 8.1)
        cls._box(ax, 0.45, 1.88, 2.35, 0.64, "Chem2D (ECFP4/6+RDKit)", "Bất biến conformer (3263-D)", "#DBEAFE", "#0284C7", 8.1)
        cls._box(ax, 0.45, 0.92, 2.35, 0.64, "Uni-Mol (3D Conformer)", "Mean+Max (1024-D)", "#FEF3C7", "#D97706", 8.1)

        cls._box(ax, 3.35, 3.80, 2.15, 0.64, "Tower Prot-Seq", "LN + Noise + 2xGELU -> 384D", "#EFF6FF", "#2563EB", 8.1)
        cls._box(ax, 3.35, 2.84, 2.15, 0.64, "Tower Prot-3D", "LN + Noise + 2xGELU -> 384D", "#FFFBEB", "#D97706", 8.1)
        cls._box(ax, 3.35, 1.88, 2.15, 0.64, "Tower Drug-2D", "LN + Noise + 2xGELU -> 384D", "#EFF6FF", "#2563EB", 8.1)
        cls._box(ax, 3.35, 0.92, 2.15, 0.64, "Tower Drug-3D", "LN + Noise + 2xGELU -> 384D", "#FFFBEB", "#D97706", 8.1)

        cls._box(ax, 6.20, 3.20, 2.15, 0.88, "AdaptiveGate_P", "ModalityDrop (p=0.15)\ng_p=0.45 -> hp (384D)", "#D1FAE5", "#059669", 8.2)
        cls._box(ax, 6.20, 1.28, 2.15, 0.88, "AdaptiveGate_D", "ModalityDrop (p=0.15)\ng_d=0.54 -> hd (384D)", "#D1FAE5", "#059669", 8.2)
        cls._box(ax, 8.95, 2.02, 2.55, 1.32, "Multi-View Interaction", "2,304-D = [hp, hd, hp*hd,\n|hp-hd|, Seq*2D, 3D*3D]", "#EDE9FE", "#7C3AED", 8.3)
        cls._box(ax, 12.00, 2.14, 1.95, 1.08, "MLP + Dual Aux BCE", "2304 -> 512 -> 128 -> 1\n+ 0.2*(L_seq2d + L_3d)", "#DCFCE7", "#15803D", 8.2)

        for y_val, col in ((4.12, "#0284C7"), (3.16, "#D97706"), (2.20, "#0284C7"), (1.24, "#D97706")):
            cls._arrow(ax, 2.84, y_val, 3.31, y_val, col)
        cls._arrow(ax, 5.54, 4.12, 6.16, 3.78, "#0284C7"); cls._arrow(ax, 5.54, 3.16, 6.16, 3.50, "#D97706")
        cls._arrow(ax, 5.54, 2.20, 6.16, 1.86, "#0284C7"); cls._arrow(ax, 5.54, 1.24, 6.16, 1.58, "#D97706")
        ax.annotate("", xy=(4.42, 3.52), xytext=(4.42, 3.76), arrowprops=dict(arrowstyle="<->", lw=1.4, color="#059669", ls="--"), zorder=3)
        ax.annotate("", xy=(4.42, 1.60), xytext=(4.42, 1.84), arrowprops=dict(arrowstyle="<->", lw=1.4, color="#059669", ls="--"), zorder=3)
        ax.text(4.56, 3.64, "Cross-View InfoNCE", fontsize=7.6, fontweight="bold", color="#059669", ha="left", va="center")
        ax.text(4.56, 1.72, "Cross-View InfoNCE", fontsize=7.6, fontweight="bold", color="#059669", ha="left", va="center")
        cls._arrow(ax, 8.39, 3.64, 8.91, 2.90, "#059669"); cls._arrow(ax, 8.39, 1.72, 8.91, 2.44, "#059669"); cls._arrow(ax, 11.54, 2.68, 11.96, 2.68, "#7C3AED")

        # Legend
        ax.add_patch(FancyBboxPatch((0.2, 0.14), 14.0, 0.46, boxstyle="round,pad=0.03,rounding_size=0.10", fc="#F1F5F9", ec="#CBD5E1", lw=1.0))
        ax.text(0.42, 0.37, "VISUAL NOTATION:", fontsize=8.3, fontweight="bold", color="#0F172A", va="center")
        legend_items = [
            (2.25, "#DBEAFE", "Seq / Chem2D View"),
            (4.35, "#FEF3C7", "3D Structural View"),
            (6.45, "#FEE2E2", "Baseline Pad Leak"),
            (8.55, "#D1FAE5", "Gate + InfoNCE (- - >)"),
            (11.05, "#EDE9FE", "Multi-View Interaction"),
        ]
        for x_cur, col, lbl in legend_items:
            ax.add_patch(FancyBboxPatch((x_cur, 0.25), 0.22, 0.22, boxstyle="round,pad=0.01,rounding_size=0.04", fc=col, ec="#475569", lw=0.9))
            ax.text(x_cur + 0.30, 0.36, lbl, fontsize=8.1, color="#334155", va="center")

        plt.tight_layout(pad=0.15)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
