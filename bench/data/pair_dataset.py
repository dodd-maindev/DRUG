"""PyTorch Dataset and padding collator for 3DICE and Delta-Interface pairs."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class BenchmarkPairDataset(Dataset):
    """Serves aligned protein/drug embeddings, 3D geometry, and optional coarse logits."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        entities: Dict[str, Dict[str, Any]],
        trim_protein_token: bool = False,
        coarse_logits: Optional[np.ndarray] = None,
    ) -> None:
        """Filter valid entity pairs and store split tensors and integer entity IDs."""
        proteins, drugs = entities["proteins"], entities["drugs"]
        keep = [p in proteins and d in drugs for p, d in zip(dataframe["protein"], dataframe["drug"])]
        self.df = dataframe[keep].reset_index(drop=True)
        self.proteins = proteins
        self.drugs = drugs
        self.trim_protein_token = trim_protein_token
        self.coarse_logits = (
            np.asarray(coarse_logits, dtype=np.float32)[keep]
            if coarse_logits is not None and len(coarse_logits) == len(dataframe)
            else np.zeros(len(self.df), dtype=np.float32)
        )
        p_vocab = {pid: idx for idx, pid in enumerate(sorted(set(self.df["protein"])))}
        d_vocab = {did: idx for idx, did in enumerate(sorted(set(self.df["drug"])))}
        self.protein_indices = np.array([p_vocab[p] for p in self.df["protein"]], dtype=np.int64)
        self.drug_indices = np.array([d_vocab[d] for d in self.df["drug"]], dtype=np.int64)
        self.protein_lengths = np.array([proteins[p]["emb"].shape[0] for p in self.df["protein"]], dtype=np.int64)

    def __len__(self) -> int:
        """Return total valid interaction pairs in the split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return unpadded tensors and metadata for one interaction pair."""
        pid, did = self.df.at[idx, "protein"], self.df.at[idx, "drug"]
        p_ent, d_ent = self.proteins[pid], self.drugs[did]
        p_emb = p_ent["emb"].float()
        ca, valid_ca, rho = p_ent["ca"], p_ent["valid_ca"], p_ent["rho"]
        if self.trim_protein_token and p_emb.shape[0] > 1:
            p_emb, ca, valid_ca, rho = p_emb[:-1], ca[:-1], valid_ca[:-1], rho[:-1]
        return {
            "protein_emb": p_emb, "ca_coords": ca, "valid_ca": valid_ca, "rho": rho,
            "drug_emb": d_ent["emb"].float(), "drug_coords": d_ent["coords"], "drug_is_h": d_ent["is_h"],
            "rg_feats": d_ent["rg_feats"], "rg": float(d_ent["rg"]),
            "label": int(self.df.at[idx, "label"]), "b_logit": float(self.coarse_logits[idx]),
            "protein_idx": int(self.protein_indices[idx]), "drug_idx": int(self.drug_indices[idx]),
            "uniprot_id": pid, "drug_id": did,
        }


class BenchmarkBatchCollator:
    """Collates variable-length pairs into padded GPU-ready tensors with True=PAD masks."""

    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad sequences and construct boolean padding masks and scalar batch fields."""
        p_embs = [b["protein_emb"] for b in batch]
        d_embs = [b["drug_emb"] for b in batch]
        p_lens = torch.tensor([x.size(0) for x in p_embs], dtype=torch.long)
        d_lens = torch.tensor([x.size(0) for x in d_embs], dtype=torch.long)
        lp_max, ld_max = int(p_lens.max()), int(d_lens.max())
        bsz = len(batch)

        return {
            "protein_emb": pad_sequence(p_embs, batch_first=True),
            "ca_coords": pad_sequence([b["ca_coords"] for b in batch], batch_first=True),
            "valid_ca": pad_sequence([b["valid_ca"] for b in batch], batch_first=True, padding_value=False),
            "rho": pad_sequence([b["rho"] for b in batch], batch_first=True),
            "protein_mask": torch.arange(lp_max).unsqueeze(0).expand(bsz, lp_max) >= p_lens.unsqueeze(1),
            "drug_emb": pad_sequence(d_embs, batch_first=True),
            "drug_coords": pad_sequence([b["drug_coords"] for b in batch], batch_first=True),
            "drug_is_h": pad_sequence([b["drug_is_h"] for b in batch], batch_first=True, padding_value=False),
            "drug_mask": torch.arange(ld_max).unsqueeze(0).expand(bsz, ld_max) >= d_lens.unsqueeze(1),
            "rg_feats": torch.stack([b["rg_feats"] for b in batch]),
            "rg": torch.tensor([b["rg"] for b in batch], dtype=torch.float32),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "b_logit": torch.tensor([b["b_logit"] for b in batch], dtype=torch.float32),
            "protein_idx": torch.tensor([b["protein_idx"] for b in batch], dtype=torch.long),
            "drug_idx": torch.tensor([b["drug_idx"] for b in batch], dtype=torch.long),
            "uniprot_id": [b["uniprot_id"] for b in batch],
            "drug_id": [b["drug_id"] for b in batch],
        }
