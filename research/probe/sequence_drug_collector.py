"""Collects DrugBank standard and cold-start splits with sequences and SMILES."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd


class SplitMetadataCollector:
    """Loads multi-split DTI CSVs and aggregates unique protein sequences and SMILES."""

    @staticmethod
    def _find_col(columns: List[str], candidates: List[str]) -> str:
        """Resolve the first matching column name case-insensitively."""
        lower_map = {str(c).lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        raise KeyError(f"None of {candidates} in {columns}")

    @classmethod
    def read_split(cls, csv_path: str) -> pd.DataFrame:
        """Read one CSV into standardized drug, protein, smiles, sequence, label columns."""
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        d_col = cls._find_col(cols, ["drug_id", "drug"])
        p_col = cls._find_col(cols, ["uniprot_id", "protein_id", "protein"])
        s_col = cls._find_col(cols, ["SMILES", "smiles"])
        q_col = cls._find_col(cols, ["Target_sequence", "sequence", "protein_sequence"])
        y_col = cls._find_col(cols, ["interaction", "label", "y"])
        return pd.DataFrame({
            "drug": df[d_col].astype(str),
            "protein": df[p_col].astype(str),
            "smiles": df[s_col].astype(str),
            "sequence": df[q_col].astype(str),
            "label": df[y_col].astype(int),
        })

    @classmethod
    def load_all_benchmarks(cls, list_dir: str) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, str], Dict[str, str]]:
        """Load db_std, drug_cs, and prot_cs splits plus unique sequence and SMILES maps."""
        specs = {
            "db_std": ("db_train.csv", "db_val.csv", "db_test.csv"),
            "drug_cs": ("db_drug_cs_train.csv", "db_drug_cs_val.csv", "db_drug_cs_test.csv"),
            "prot_cs": ("db_protein_cs_train.csv", "db_protein_cs_val.csv", "db_protein_cs_test.csv"),
        }
        benchmarks: Dict[str, Dict[str, pd.DataFrame]] = {}
        seqs: Dict[str, str] = {}
        smiles_map: Dict[str, str] = {}

        for name, (tr_f, va_f, te_f) in specs.items():
            tr_p, va_p, te_p = (os.path.join(list_dir, f) for f in (tr_f, va_f, te_f))
            if not (os.path.exists(tr_p) and os.path.exists(va_p) and os.path.exists(te_p)):
                continue
            parts = {"train": cls.read_split(tr_p), "val": cls.read_split(va_p), "test": cls.read_split(te_p)}
            benchmarks[name] = parts
            for df in parts.values():
                for pid, seq in zip(df["protein"], df["sequence"]):
                    if pid not in seqs and seq and seq != "nan":
                        seqs[pid] = seq
                for did, smi in zip(df["drug"], df["smiles"]):
                    if did not in smiles_map and smi and smi != "nan":
                        smiles_map[did] = smi
        return benchmarks, seqs, smiles_map
