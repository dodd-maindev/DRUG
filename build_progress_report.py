#!/usr/bin/env python3
"""CLI builder generating draw.io diagram, 300-DPI figure, and DOCX progress report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.omni.arch_diagram_plotter import ArchitectureDiagramPlotter
from bench.omni.docx_report_builder import MarkdownToDocxBuilder
from bench.omni.drawio_builder import DrawioArchitectureBuilder


class ProgressReportCli:
    """Generates draw.io architecture tabs, comparison figure, and Word report."""

    @staticmethod
    def main() -> None:
        """Parse paths and emit .drawio, architecture PNG, and .docx deliverable."""
        ap = argparse.ArgumentParser(description="Build 3DICE vs Omni3D-DTI diagrams and DOCX report.")
        ap.add_argument("--md", default="Bao_cao_Tien_do_3DICE.md")
        ap.add_argument("--docx", default="Bao_cao_Tien_do_3DICE.docx")
        ap.add_argument("--figures-dir", default="figures")
        args = ap.parse_args()

        os.makedirs(args.figures_dir, exist_ok=True)
        drawio_path = os.path.join(args.figures_dir, "3DICE_vs_Omni3D_DTI.drawio")
        arch_png = os.path.join(args.figures_dir, "architecture_comparison.png")

        DrawioArchitectureBuilder.write(drawio_path)
        ArchitectureDiagramPlotter.render(arch_png)
        if os.path.exists(args.md):
            MarkdownToDocxBuilder.build(args.md, args.docx, base_dir=str(ROOT))
        print(f"[built] {drawio_path}, {arch_png}, {args.docx}", flush=True)


if __name__ == "__main__":
    ProgressReportCli.main()
