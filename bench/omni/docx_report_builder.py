"""Markdown-to-DOCX academic report compiler embedding tables and 300-DPI figures."""

from __future__ import annotations

import os
import re

from docx import Document
from docx.shared import Inches, Pt

from bench.omni.docx_styles import DocxStyleHelper


class MarkdownToDocxBuilder:
    """Converts a structured Markdown progress report into a styled Word document."""

    @classmethod
    def build(cls, md_path: str, docx_path: str, base_dir: str | None = None) -> None:
        """Read Markdown report and emit a styled DOCX file with figures and tables."""
        doc = Document()
        sec = doc.sections[0]
        sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(0.75)
        doc.styles["Normal"].font.name, doc.styles["Normal"].font.size = "Calibri", Pt(10.5)

        root_dir = base_dir or os.path.dirname(os.path.abspath(md_path))
        lines = open(md_path, "r", encoding="utf-8").read().splitlines()
        i = 0
        while i < len(lines):
            ln = lines[i].rstrip()
            if not ln or ln == "---":
                i += 1
                continue
            if ln.startswith("|"):
                tbl_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tbl_lines.append(lines[i].strip())
                    i += 1
                DocxStyleHelper.add_table(doc, tbl_lines)
                continue
            m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", ln)
            if m_img:
                img_p = m_img.group(2) if os.path.isabs(m_img.group(2)) else os.path.join(root_dir, m_img.group(2))
                DocxStyleHelper.add_figure(doc, img_p, m_img.group(1))
            elif ln.startswith("# "):
                DocxStyleHelper.add_heading_line(doc, ln[2:].strip(), 1)
            elif ln.startswith("## "):
                DocxStyleHelper.add_heading_line(doc, ln[3:].strip(), 2)
            elif ln.startswith("### "):
                DocxStyleHelper.add_heading_line(doc, ln[4:].strip(), 3)
            elif ln.startswith("- "):
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(2)
                DocxStyleHelper.add_inline_runs(p, ln[2:].strip(), 10.0)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(4)
                DocxStyleHelper.add_inline_runs(p, ln.lstrip("> "), 10.2)
            i += 1
        os.makedirs(os.path.dirname(os.path.abspath(docx_path)), exist_ok=True)
        doc.save(docx_path)
