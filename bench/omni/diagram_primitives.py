"""Reusable Matplotlib primitive helpers for architectural flowcharts."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


class DiagramPrimitives:
    """Shared canvas, node, arrow, legend, and export utilities for diagrams."""

    @staticmethod
    def new_canvas(width: float, height: float, xlim: float, ylim: float):
        """Create a crisp 300-DPI white Matplotlib canvas without axes."""
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(width, height), dpi=300)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0.0, xlim)
        ax.set_ylim(0.0, ylim)
        ax.axis("off")
        return fig, ax

    @staticmethod
    def card(ax, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.3) -> None:
        """Draw a large background panel container."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18", fc=fc, ec=ec, lw=lw, zorder=0))

    @staticmethod
    def box(ax, x: float, y: float, w: float, h: float, title: str, sub: str, fc: str, ec: str, fs: float = 8.4) -> None:
        """Draw a rounded colored block with centered title and subtitle."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14", fc=fc, ec=ec, lw=1.6, zorder=2))
        txt = f"{title}\n{sub}" if sub else title
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="medium", color="#0F172A", linespacing=1.28, zorder=3)

    @staticmethod
    def arrow(ax, x1: float, y1: float, x2: float, y2: float, col: str = "#475569", style: str = "-|>", dashed: bool = False) -> None:
        """Draw a directional connector arrow between diagram components."""
        ls = "--" if dashed else "-"
        props = dict(arrowstyle=style, lw=1.5, color=col, ls=ls)
        if style == "<->":
            props["mutation_scale"] = 9.5
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=1)

    @staticmethod
    def legend_bar(ax, y: float, w: float, items: Iterable[Tuple[float, str, str]]) -> None:
        """Render a compact visual legend bar at the bottom of a figure."""
        ax.add_patch(FancyBboxPatch((0.18, y), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.10", fc="#F1F5F9", ec="#CBD5E1", lw=1.0))
        ax.text(0.40, y + 0.21, "CHÚ THÍCH:", fontsize=8.2, fontweight="bold", color="#0F172A", va="center")
        for x_cur, col, lbl in items:
            ax.add_patch(FancyBboxPatch((x_cur, y + 0.10), 0.22, 0.22, boxstyle="round,pad=0.01,rounding_size=0.04", fc=col, ec="#475569", lw=0.9))
            ax.text(x_cur + 0.30, y + 0.21, lbl, fontsize=8.0, color="#334155", va="center")

    @staticmethod
    def save(fig, out_path: str) -> None:
        """Save figure to 300-DPI PNG and close Matplotlib handles."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.tight_layout(pad=0.15)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
