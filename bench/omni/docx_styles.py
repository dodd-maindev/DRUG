"""Formatting primitives (inline spans, headings, images, shaded tables) for DOCX."""

from __future__ import annotations

import os
import re
from typing import List

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor


class DocxStyleHelper:
    """Renders formatted paragraphs, tables, and embedded PNG figures into python-docx."""

    @staticmethod
    def add_inline_runs(paragraph, text: str, base_size: float = 10.5, base_color: RGBColor = RGBColor(30, 41, 59)) -> None:
        """Split **bold** and `code` spans onto a python-docx paragraph."""
        for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                r = paragraph.add_run(tok[2:-2])
                r.bold, r.font.size, r.font.color.rgb = True, Pt(base_size), base_color
            elif tok.startswith("`") and tok.endswith("`"):
                r = paragraph.add_run(tok[1:-1])
                r.font.name, r.font.size, r.font.color.rgb = "Consolas", Pt(max(base_size - 1.0, 8.0)), RGBColor(15, 65, 130)
            else:
                r = paragraph.add_run(tok)
                r.font.size, r.font.color.rgb = Pt(base_size), base_color

    @staticmethod
    def add_heading_line(doc: Document, text: str, level: int) -> None:
        """Append styled document title or section heading."""
        p = doc.add_paragraph()
        if level == 1:
            p.alignment, p.paragraph_format.space_after = 1, Pt(8)
            r = p.add_run(text)
            r.bold, r.font.size, r.font.color.rgb = True, Pt(18), RGBColor(15, 23, 42)
        elif level == 2:
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(12), Pt(4)
            r = p.add_run(text)
            r.bold, r.font.size, r.font.color.rgb = True, Pt(13.5), RGBColor(15, 23, 42)
        else:
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(3)
            r = p.add_run(text)
            r.bold, r.font.size, r.font.color.rgb = True, Pt(11.2), RGBColor(30, 58, 138)

    @staticmethod
    def add_figure(doc: Document, img_path: str, caption: str) -> None:
        """Embed a centered PNG image with italicized caption if present on disk."""
        if not os.path.exists(img_path):
            return
        p_img = doc.add_paragraph()
        p_img.alignment = 1
        p_img.add_run().add_picture(img_path, width=Inches(6.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = 1
        rc = p_cap.add_run(caption)
        rc.italic, rc.font.size, rc.font.color.rgb = True, Pt(9.0), RGBColor(71, 85, 105)

    @classmethod
    def add_table(cls, doc: Document, raw_lines: List[str]) -> None:
        """Render Markdown pipe table rows into a bordered shaded Word table."""
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in raw_lines]
        rows = [r for i, r in enumerate(rows) if not (i == 1 and all(set(c) <= set(":- ") for c in r))]
        if not rows:
            return
        table = doc.add_table(rows=len(rows), cols=len(rows[0]))
        table.style = "Table Grid"
        for r_idx, row in enumerate(rows):
            for c_idx, cell_txt in enumerate(row):
                cell = table.cell(r_idx, c_idx)
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(2)
                if r_idx == 0:
                    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1E3A8A"/>'))
                    cls.add_inline_runs(p, f"**{cell_txt}**", 8.5, RGBColor(255, 255, 255))
                else:
                    if r_idx % 2 == 0:
                        cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8FAFC"/>'))
                    cls.add_inline_runs(p, cell_txt, 8.5, RGBColor(30, 41, 59))
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
