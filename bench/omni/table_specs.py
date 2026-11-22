"""Literature baselines and structured section builders for Table 1 and Table 2."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

Stat = Tuple[float, float]


class PaperTableData:
    """Provides literature benchmarks and transforms omni3d_summary.json for rendering."""

    METRICS = ("accuracy", "ppv", "tpr", "f1", "mcc", "auroc", "auprc")
    HEADERS = ("Model", "ACC", "PPV", "TPR", "F1", "MCC", "AUROC", "AUPRC")
    DATASET_ORDER = [
        ("db_std", "DrugBank"),
        ("kiba", "KIBA"),
        ("drug_cs", "Drug cold-start"),
        ("prot_cs", "Protein cold-start"),
    ]
    BASELINES: Dict[str, List[Tuple[str, Tuple[Stat, ...]]]] = {
        "db_std": [
            ("MolTrans", ((0.7780, 0.0063), (0.7437, 0.0128), (0.8723, 0.0202), (0.8027, 0.0025), (0.5587, 0.0045), (0.8650, 0.0031), (0.8520, 0.0073))),
            ("DLM-DTI", ((0.7906, 0.0016), (0.7854, 0.0078), (0.8000, 0.0154), (0.7920, 0.0042), (0.5818, 0.0029), (0.8651, 0.0035), (0.8605, 0.0030))),
            ("MocFormer", ((0.7972, 0.0021), (0.7741, 0.0030), (0.8463, 0.0045), (0.8081, 0.0027), (0.5976, 0.0029), (0.8816, 0.0019), (0.8780, 0.0071))),
            ("3DICE", ((0.8198, 0.0040), (0.8086, 0.0118), (0.8385, 0.0100), (0.8231, 0.0018), (0.6403, 0.0074), (0.8933, 0.0021), (0.8852, 0.0037))),
        ],
        "kiba": [
            ("MolTrans", ((0.8813, 0.0022), (0.7189, 0.0160), (0.6505, 0.0231), (0.6826, 0.0083), (0.6113, 0.0081), (0.9047, 0.0041), (0.7513, 0.0148))),
            ("DLM-DTI", ((0.8679, 0.0035), (0.6970, 0.0412), (0.5754, 0.0726), (0.6262, 0.0284), (0.5535, 0.0188), (0.8868, 0.0048), (0.7053, 0.0118))),
            ("MocFormer", ((0.8904, 0.0009), (0.7352, 0.0062), (0.6792, 0.0152), (0.7058, 0.0060), (0.6391, 0.0049), (0.9222, 0.0004), (0.7913, 0.0005))),
            ("3DICE", ((0.8915, 0.0006), (0.7645, 0.0175), (0.6373, 0.0263), (0.6984, 0.0099), (0.6357, 0.0058), (0.9231, 0.0009), (0.7941, 0.0016))),
        ],
        "drug_cs": [
            ("MolTrans", ((0.6857, 0.0127), (0.6648, 0.0161), (0.6651, 0.0217), (0.6646, 0.0136), (0.3995, 0.0075), (0.7540, 0.0135), (0.7502, 0.0138))),
            ("DLM-DTI", ((0.7032, 0.0045), (0.7195, 0.0120), (0.6731, 0.0216), (0.6935, 0.0091), (0.4025, 0.0068), (0.7661, 0.0075), (0.7664, 0.0021))),
            ("MocFormer", ((0.6990, 0.0011), (0.6855, 0.0017), (0.7356, 0.0052), (0.7096, 0.0019), (0.3991, 0.0023), (0.7788, 0.0005), (0.7754, 0.0007))),
            ("3DICE", ((0.7474, 0.0058), (0.7471, 0.0207), (0.7434, 0.0351), (0.7445, 0.0101), (0.4913, 0.0130), (0.8195, 0.0044), (0.8182, 0.0052))),
        ],
        "prot_cs": [
            ("MolTrans", ((0.6987, 0.0010), (0.6775, 0.0110), (0.6816, 0.0141), (0.6795, 0.0022), (0.3964, 0.0058), (0.7689, 0.0051), (0.7767, 0.0036))),
            ("DLM-DTI", ((0.7139, 0.0056), (0.7398, 0.0045), (0.6564, 0.0238), (0.6954, 0.0121), (0.4288, 0.0109), (0.7752, 0.0097), (0.7714, 0.0069))),
            ("MocFormer", ((0.7407, 0.0011), (0.7625, 0.0018), (0.6991, 0.0007), (0.7294, 0.0009), (0.4830, 0.0023), (0.8137, 0.0010), (0.8211, 0.0009))),
            ("3DICE", ((0.7476, 0.0103), (0.7737, 0.0218), (0.7053, 0.0211), (0.7375, 0.0053), (0.5048, 0.0095), (0.8243, 0.0032), (0.8297, 0.0035))),
        ],
    }
    ABLATIONS = [
        ("Omni3D_DTI", "Omni3D-DTI (Full)"),
        ("Omni3D_wo_Gate", "w/o Adaptive Gate (Symm.)"),
        ("Omni3D_wo_Align", "w/o Cross-View InfoNCE"),
        ("Omni3D_wo_ModDrop", "w/o Modality Drop + Aux"),
        ("Omni3D_wo_3D", "w/o 3D (Seq + Chem2D)"),
        ("Omni3D_wo_Seq2D", "w/o Seq/2D (3D Only)"),
        ("Omni3D_wo_ESM2", "w/o ESM-2 (ESM-IF1 + Drug)"),
        ("Omni3D_wo_Chem2D", "w/o Chem2D (Prot + Uni-Mol)"),
    ]

    @classmethod
    def _from_result(cls, label: str, rec: Dict[str, Any] | None, accent: bool) -> Dict[str, Any]:
        """Convert one summary JSON record into display strings and sortable float values."""
        if rec is None:
            return {"label": label, "vals": [math.nan] * 7, "fmt": ["—"] * 7, "accent": accent}
        vals = [float(rec[k]) for k in cls.METRICS]
        fmt = [f"{rec[k]:.4f} ({rec[f'{k}_std']:.4f})" for k in cls.METRICS]
        return {"label": label, "vals": vals, "fmt": fmt, "accent": accent}

    @classmethod
    def build_table1_sections(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assemble Table 1 sections placing Omni3D-DTI (Ours) as Row 5 in every block."""
        index = {(r["split"], r["arm"]): r for r in results}
        sections: List[Dict[str, Any]] = []
        for split_key, title in cls.DATASET_ORDER:
            rows = [
                {"label": m, "vals": [s[0] for s in stats], "fmt": [f"{s[0]:.4f} ({s[1]:.4f})" for s in stats], "accent": False}
                for m, stats in cls.BASELINES[split_key]
            ]
            rows.append(cls._from_result("Omni3D-DTI (Ours)", index.get((split_key, "Omni3D_DTI")), accent=True))
            sections.append({"title": title, "rows": rows})
        return sections

    @classmethod
    def build_table2_sections(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assemble Table 2 ablation sections across DrugBank, Drug CS, and Protein CS."""
        index = {(r["split"], r["arm"]): r for r in results}
        sections: List[Dict[str, Any]] = []
        for split_key, title in cls.DATASET_ORDER:
            if not any((split_key, arm) in index for arm, _ in cls.ABLATIONS):
                continue
            rows = [cls._from_result(lbl, index.get((split_key, arm)), accent=(arm == "Omni3D_DTI")) for arm, lbl in cls.ABLATIONS]
            sections.append({"title": title, "rows": rows})
        return sections
