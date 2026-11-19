"""CSV dataset reader with automatic column detection for DTI pairs."""

import sys
from typing import Callable, Dict, Optional

import pandas as pd


class PairDatasetReader:
    """Reads and normalizes drug-target interaction CSV files."""

    def __init__(
        self,
        drug_column: Optional[str] = None,
        protein_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """Initialize optional explicit column overrides."""
        self.drug_column = drug_column
        self.protein_column = protein_column
        self.label_column = label_column

    @staticmethod
    def _select_column(
        dataframe: pd.DataFrame,
        explicit_name: Optional[str],
        predicate: Callable[[str], bool],
        role_name: str,
    ) -> str:
        """Resolve a required column name explicitly or by predicate match."""
        if explicit_name:
            if explicit_name not in dataframe.columns:
                sys.exit(f"[!] column '{explicit_name}' not in {list(dataframe.columns)}")
            return explicit_name
        for column in dataframe.columns:
            if predicate(str(column).lower()):
                return column
        sys.exit(f"[!] cannot detect {role_name} column in {list(dataframe.columns)}")

    def read_pairs(self, csv_path: str) -> pd.DataFrame:
        """Load a split CSV into standardized columns: drug, protein, label."""
        dataframe = pd.read_csv(csv_path, sep=None, engine="python")
        drug_key = self._select_column(
            dataframe,
            self.drug_column,
            lambda lower: "drug" in lower and "smiles" not in lower,
            "drug",
        )
        protein_key = self._select_column(
            dataframe,
            self.protein_column,
            lambda lower: any(token in lower for token in ("prot", "target", "uniprot"))
            and "seq" not in lower,
            "protein",
        )
        label_key = self._select_column(
            dataframe,
            self.label_column,
            lambda lower: lower in ("y", "label", "labels", "interaction")
            or "label" in lower
            or "interact" in lower,
            "label",
        )
        return pd.DataFrame(
            {
                "drug": dataframe[drug_key].astype(str).values,
                "protein": dataframe[protein_key].astype(str).values,
                "label": dataframe[label_key].astype(int).values,
            }
        )

    def load_splits(
        self,
        train_csv_path: Optional[str],
        validation_csv_path: Optional[str],
        test_csv_path: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """Load all provided dataset split CSVs into a dictionary."""
        splits: Dict[str, pd.DataFrame] = {}
        mapping = {
            "train": train_csv_path,
            "validation": validation_csv_path,
            "test": test_csv_path,
        }
        for split_name, csv_path in mapping.items():
            if csv_path:
                splits[split_name] = self.read_pairs(csv_path)
        return splits
