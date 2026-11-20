"""CSV split loader retaining SMILES and computing RDKit Morgan fingerprints."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class SplitSmilesLoader:
    """Loads DTI splits with SMILES and builds dense ECFP4 fingerprint matrices."""

    @staticmethod
    def _detect_column(dataframe: pd.DataFrame, candidates: List[str]) -> str:
        """Pick the first matching column case-insensitively."""
        lower_map = {str(col).lower(): col for col in dataframe.columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        raise KeyError(f"None of {candidates} found in {list(dataframe.columns)}")

    @classmethod
    def load_split(cls, csv_path: str) -> pd.DataFrame:
        """Load split CSV with standardized drug, protein, smiles, and label columns."""
        dataframe = pd.read_csv(csv_path, sep=None, engine="python")
        drug_col = cls._detect_column(dataframe, ["drug_id", "drugbank_id", "drug"])
        prot_col = cls._detect_column(dataframe, ["uniprot_id", "protein_id", "target_id", "protein"])
        smiles_col = cls._detect_column(dataframe, ["SMILES", "smiles"])
        label_col = cls._detect_column(dataframe, ["interaction", "label", "y"])
        return pd.DataFrame({
            "drug": dataframe[drug_col].astype(str),
            "protein": dataframe[prot_col].astype(str),
            "smiles": dataframe[smiles_col].astype(str),
            "label": dataframe[label_col].astype(int),
        })

    @staticmethod
    def build_ecfp4_map(smiles_by_drug: Dict[str, str], radius: int = 2, bits: int = 2048) -> Dict[str, np.ndarray]:
        """Compute 2048-bit float32 Morgan fingerprints keyed by drug identifier."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        try:
            from rdkit.Chem import rdFingerprintGenerator
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
            encode = lambda mol: generator.GetFingerprintAsNumPy(mol).astype(np.float32)  # noqa: E731
        except Exception:
            encode = lambda mol: np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits), dtype=np.float32)  # noqa: E731

        fingerprints: Dict[str, np.ndarray] = {}
        for drug_id, smiles in smiles_by_drug.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                fingerprints[drug_id] = encode(mol)
        return fingerprints

    @staticmethod
    def build_drug_smiles_dict(dataframes: List[pd.DataFrame]) -> Dict[str, str]:
        """Aggregate unique drug_id -> SMILES mappings across multiple splits."""
        mapping: Dict[str, str] = {}
        for dataframe in dataframes:
            for drug_id, smiles in zip(dataframe["drug"], dataframe["smiles"]):
                if drug_id not in mapping and smiles and smiles != "nan":
                    mapping[drug_id] = smiles
        return mapping
