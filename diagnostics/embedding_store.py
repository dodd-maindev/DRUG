"""On-disk embedding loader matching 3DICE MyDataset path and trimming rules."""

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


class EmbeddingStore:
    """Resolves and loads precomputed protein and drug embedding tensors."""

    def __init__(self, protein_directory: Optional[str], drug_directory: Optional[str], trim: bool = True) -> None:
        """Initialize embedding directories and terminal-token trimming mode."""
        self.protein_directory = protein_directory
        self.drug_directory = drug_directory
        self.trim_terminal_token = trim

    @staticmethod
    def _extract_first_tensor(loaded_object: Any) -> torch.Tensor:
        """Extract the primary 2-D tensor from a loaded PyTorch artifact."""
        if torch.is_tensor(loaded_object):
            return loaded_object
        if isinstance(loaded_object, dict):
            for key in ("embedding", "emb", "representations", "atomic_reprs", "encoder_out"):
                if key in loaded_object and torch.is_tensor(loaded_object[key]):
                    return loaded_object[key]
            for value in loaded_object.values():
                if torch.is_tensor(value) and value.dim() == 2:
                    return value
        raise ValueError("Cannot find a 2-D tensor in .pt file.")

    def resolve_protein_path(self, protein_identifier: str) -> Optional[str]:
        """Find the protein .pt path using sorted files[-1] to match MyDataset."""
        if not self.protein_directory:
            return None
        folder_path = os.path.join(self.protein_directory, protein_identifier)
        if os.path.isdir(folder_path):
            files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
            return files[-1] if files else None
        direct_path = os.path.join(self.protein_directory, f"{protein_identifier}.pt")
        return direct_path if os.path.exists(direct_path) else None

    def resolve_drug_path(self, drug_identifier: str) -> Optional[str]:
        """Find the drug .pt path checking _unimol.pt first, then .pt."""
        if not self.drug_directory:
            return None
        for filename in (f"{drug_identifier}_unimol.pt", f"{drug_identifier}.pt"):
            candidate = os.path.join(self.drug_directory, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    def load_raw_protein(self, protein_identifier: str) -> torch.Tensor:
        """Load the untrimmed (L, 512) float32 protein embedding tensor."""
        loaded = torch.load(self.resolve_protein_path(protein_identifier), map_location="cpu", weights_only=False)
        return self._extract_first_tensor(loaded).float().reshape(-1, 512)

    def load_protein(self, protein_identifier: str) -> torch.Tensor:
        """Load the protein embedding tensor applying optional terminal trimming."""
        tensor = self.load_raw_protein(protein_identifier)
        return tensor[:-1] if (self.trim_terminal_token and tensor.shape[0] > 1) else tensor

    def load_drug(self, drug_identifier: str) -> torch.Tensor:
        """Load the drug embedding tensor applying optional terminal trimming."""
        loaded = torch.load(self.resolve_drug_path(drug_identifier), map_location="cpu", weights_only=False)
        raw_reprs = loaded["atomic_reprs"] if isinstance(loaded, dict) else loaded
        array = np.asarray(raw_reprs, dtype=np.float32).reshape(-1, 512)
        tensor = torch.from_numpy(array)
        return tensor[:-1] if (self.trim_terminal_token and tensor.shape[0] > 1) else tensor

    def compute_coverage(self, splits: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Summarize missing drug/protein embedding counts per split."""
        summary_rows: List[Dict[str, Any]] = []
        for split_name, dataframe in splits.items():
            missing_drug = np.array([self.resolve_drug_path(item) is None for item in dataframe["drug"]])
            missing_protein = np.array([self.resolve_protein_path(item) is None for item in dataframe["protein"]])
            dropped = missing_drug | missing_protein
            summary_rows.append({
                "split": split_name, "count": len(dataframe),
                "positive_rate": float(dataframe["label"].mean()),
                "missing_drug": int(missing_drug.sum()), "missing_protein": int(missing_protein.sum()),
                "dropped": int(dropped.sum()),
                "positive_rate_dropped": float(dataframe["label"][dropped].mean()) if dropped.any() else float("nan"),
            })
        return summary_rows
