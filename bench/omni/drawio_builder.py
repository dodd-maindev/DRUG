"""Draw.io XML generator comparing BASE_3DICE and Omni3D_DTI architectures."""

from __future__ import annotations

import html
import os
from typing import List


class DrawioArchitectureBuilder:
    """Generates a two-tab draw.io diagram matching academic architecture conventions."""

    FONT = "Comic Sans MS"

    @classmethod
    def _box(cls, cid: str, x: int, y: int, w: int, h: int, text: str, fill: str, stroke: str, fsize: int = 11, bold: bool = True) -> str:
        """Build one rounded draw.io box node."""
        val = html.escape(text, quote=True)
        b = "1" if bold else "0"
        st = f"rounded=1;arcSize=14;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.8;fontFamily={cls.FONT};fontSize={fsize};fontStyle={b};fontColor=#0F172A;"
        return f'<mxCell id="{cid}" value="{val}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'

    @classmethod
    def _edge(cls, eid: str, src: str, tgt: str, color: str = "#334155", dashed: bool = False, label: str = "") -> str:
        """Build a directed solid or dashed connector between two nodes."""
        d = "dashed=1;dashPattern=6 4;" if dashed else ""
        st = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;{d}strokeColor={color};strokeWidth=1.8;endArrow=block;endFill=1;fontFamily={cls.FONT};fontSize=9;fontColor={color};"
        return f'<mxCell id="{eid}" value="{html.escape(label, quote=True)}" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>'

    @classmethod
    def _wrap_tab(cls, tab_id: str, name: str, cells: List[str], w: int = 1280, h: int = 600) -> str:
        """Wrap cell XML strings into a valid draw.io diagram tab."""
        body = "".join(cells)
        return (f'<diagram id="{tab_id}" name="{html.escape(name, quote=True)}">'
                f'<mxGraphModel dx="1200" dy="700" grid="1" gridSize="10" page="1" pageWidth="{w}" pageHeight="{h}">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram>')

    @classmethod
    def _tab_base_3dice(cls) -> str:
        """Construct Tab 1 illustrating the original 3DICE architecture and limitations."""
        c = [
            cls._box("b_title", 40, 20, 1140, 36, "BASE: 3DICE Architecture (12.87M Trainable Params, 61.4s/ep)", "#F8FAFC", "#94A3B8", 13),
            cls._box("bp_in", 40, 95, 195, 76, "ESM-IF1 coords only\n(seq discarded!)\n[B, Lp-1, 512]", "#FEF3C7", "#D97706", 10),
            cls._box("bd_in", 40, 235, 195, 76, "Uni-Mol atomic coords\n(44.5% H tail)\n[B, Ld, 512]", "#FEF3C7", "#D97706", 10),
            cls._box("bp_cnn", 285, 95, 175, 76, "ProteinCNN (1D k=3)\nignores protein_mask\n512 -> 512", "#FEE2E2", "#DC2626", 10),
            cls._box("bd_cnn", 285, 235, 175, 76, "DrugCNN (1D k=3)\npad leak max=0.737\n512 -> 512", "#FEE2E2", "#DC2626", 10),
            cls._box("b_ca", 515, 145, 205, 115, "CrossAttention (4h)\nSoftmax has no null sink\n+ FFN(2048)", "#EDE9FE", "#7C3AED", 10),
            cls._box("b_pool", 775, 145, 195, 115, "Masked MeanPool\n+ FusionNew (1.05M\ndead unused params)\n-> 512-D", "#F1F5F9", "#475569", 10),
            cls._box("b_mlp", 1020, 160, 160, 86, "MLP2 Classifier\n512->1024->512->2\nCrossEntropy Loss", "#E2E8F0", "#334155", 10),
            cls._edge("be1", "bp_in", "bp_cnn", "#D97706"), cls._edge("be2", "bd_in", "bd_cnn", "#D97706"),
            cls._edge("be3", "bp_cnn", "b_ca", "#475569"), cls._edge("be4", "bd_cnn", "b_ca", "#475569"),
            cls._edge("be5", "b_ca", "b_pool", "#7C3AED"), cls._edge("be6", "b_pool", "b_mlp", "#334155"),
        ]
        return cls._wrap_tab("base_3dice", "BASE_3DICE", c, 1240, 380)

    @classmethod
    def _tab_omni3d(cls) -> str:
        """Construct Tab 2 illustrating Omni3D-DTI (4 views, gates, InfoNCE, 2x2 interaction)."""
        c = [
            cls._box("o_title", 40, 18, 1180, 36, "IMPROVED: Omni3D-DTI (5.63M Params, 127 MB VRAM, 9.6s/ep)", "#EFF6FF", "#2563EB", 13),
            cls._box("op_seq", 40, 80, 185, 58, "ESM-2 650M (Seq+AA)\nMean+Max (2560-D)", "#DBEAFE", "#0284C7", 10),
            cls._box("op_3d", 40, 155, 185, 58, "ESM-IF1 (3D Backbone)\nMean+Max (1024-D)", "#FEF3C7", "#D97706", 10),
            cls._box("od_2d", 40, 265, 185, 58, "Chem2D (ECFP4/6+RDKit)\nConformer-Invariant (3263-D)", "#DBEAFE", "#0284C7", 10),
            cls._box("od_3d", 40, 340, 185, 58, "Uni-Mol (3D Conformer)\nMean+Max (1024-D)", "#FEF3C7", "#D97706", 10),
            cls._box("tp_seq", 275, 80, 165, 58, "ModalityTower_P_Seq\nLN+Noise+2xMLP -> 384D", "#EFF6FF", "#2563EB", 10),
            cls._box("tp_3d", 275, 155, 165, 58, "ModalityTower_P_3D\nLN+Noise+2xMLP -> 384D", "#FFFBEB", "#D97706", 10),
            cls._box("td_2d", 275, 265, 165, 58, "ModalityTower_D_2D\nLN+Noise+2xMLP -> 384D", "#EFF6FF", "#2563EB", 10),
            cls._box("td_3d", 275, 340, 165, 58, "ModalityTower_D_3D\nLN+Noise+2xMLP -> 384D", "#FFFBEB", "#D97706", 10),
            cls._box("gp", 500, 112, 165, 72, "AdaptiveGate_P\n+ ModalityDrop (p=0.15)\ng_p=0.45 -> hp (384D)", "#D1FAE5", "#059669", 10),
            cls._box("gd", 500, 297, 165, 72, "AdaptiveGate_D\n+ ModalityDrop (p=0.15)\ng_d=0.54 -> hd (384D)", "#D1FAE5", "#059669", 10),
            cls._box("inter", 730, 175, 225, 130, "Multi-View Interaction (2304D)\n[hp, hd, hp*hd, |hp-hd|,\n hp_seq*hd_2d, hp_3d*hd_3d]", "#EDE9FE", "#7C3AED", 10),
            cls._box("clf", 1015, 195, 200, 90, "Classifier MLP\nLN -> 512 -> 128 -> 1\n+ Aux Seq2D/3D Loss", "#DCFCE7", "#15803D", 10),
            cls._edge("e1", "op_seq", "tp_seq", "#0284C7"), cls._edge("e2", "op_3d", "tp_3d", "#D97706"),
            cls._edge("e3", "od_2d", "td_2d", "#0284C7"), cls._edge("e4", "od_3d", "td_3d", "#D97706"),
            cls._edge("e5", "tp_seq", "gp", "#0284C7"), cls._edge("e6", "tp_3d", "gp", "#D97706"),
            cls._edge("e7", "td_2d", "gd", "#0284C7"), cls._edge("e8", "td_3d", "gd", "#D97706"),
            cls._edge("e9", "tp_seq", "tp_3d", "#059669", dashed=True, label="InfoNCE"),
            cls._edge("e10", "td_2d", "td_3d", "#059669", dashed=True, label="InfoNCE"),
            cls._edge("e11", "gp", "inter", "#059669"), cls._edge("e12", "gd", "inter", "#059669"),
            cls._edge("e13", "inter", "clf", "#7C3AED"),
        ]
        return cls._wrap_tab("omni3d_dti", "Omni3D_DTI", c, 1260, 440)

    @classmethod
    def write(cls, out_path: str) -> None:
        """Write two-tab XML diagram file compatible with draw.io."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        xml = f'<mxfile host="app.diagrams.net" modified="2026-11-22">{cls._tab_base_3dice()}{cls._tab_omni3d()}</mxfile>'
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(xml)
