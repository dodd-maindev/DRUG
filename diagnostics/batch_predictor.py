"""Forward-signature adapter, padding collation, and batched logit inference."""

import inspect
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BatchPredictor:
    """Adapts variable forward() signatures and executes batched model inference."""

    @staticmethod
    def build_forward_caller(model: nn.Module) -> Callable:
        """Map forward() parameter names for protein/drug embeddings and masks."""
        mapping: Dict[str, str] = {}
        for name in inspect.signature(model.forward).parameters:
            lower = name.lower()
            if "mask" in lower and "prot" in lower:
                mapping["protein_mask"] = name
            elif "mask" in lower and "drug" in lower:
                mapping["drug_mask"] = name
            elif "prot" in lower:
                mapping["protein"] = name
            elif "drug" in lower:
                mapping["drug"] = name
        use_kwargs = len(mapping) == 4

        def call_forward(protein: torch.Tensor, drug: torch.Tensor, p_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
            if use_kwargs:
                output = model(**{
                    mapping["protein"]: protein, mapping["drug"]: drug,
                    mapping["protein_mask"]: p_mask, mapping["drug_mask"]: d_mask,
                })
            else:
                output = model(protein, drug, p_mask, d_mask)
            if isinstance(output, dict):
                output = output.get("logits", next(iter(output.values())))
            return output[0] if isinstance(output, (tuple, list)) else output

        return call_forward

    @staticmethod
    def collate_batch(
        items: List[Tuple[torch.Tensor, torch.Tensor]],
        extra_padding: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        """Pad variable-length sequences into a batch with True=PAD masks."""
        batch_size = len(items)
        max_protein = max(p.shape[0] for p, _ in items) + extra_padding
        max_drug = max(d.shape[0] for _, d in items) + extra_padding
        protein_batch = torch.zeros(batch_size, max_protein, 512)
        drug_batch = torch.zeros(batch_size, max_drug, 512)
        protein_mask = torch.ones(batch_size, max_protein, dtype=torch.bool)
        drug_mask = torch.ones(batch_size, max_drug, dtype=torch.bool)
        protein_pads, drug_pads = [], []
        for idx, (protein, drug) in enumerate(items):
            protein_batch[idx, : protein.shape[0]] = protein
            drug_batch[idx, : drug.shape[0]] = drug
            protein_mask[idx, : protein.shape[0]] = False
            drug_mask[idx, : drug.shape[0]] = False
            protein_pads.append(max_protein - protein.shape[0])
            drug_pads.append(max_drug - drug.shape[0])
        return protein_batch, drug_batch, protein_mask, drug_mask, protein_pads, drug_pads

    @classmethod
    @torch.no_grad()
    def predict_logits(
        cls,
        forward_caller: Callable,
        items: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        order: Optional[np.ndarray] = None,
        extra_padding: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batched inference and return logits alongside per-sample pad counts."""
        total = len(items)
        indices = list(range(total)) if order is None else list(order)
        logits = np.zeros((total, 2), dtype=np.float64)
        protein_pads = np.zeros(total, dtype=int)
        drug_pads = np.zeros(total, dtype=int)
        for start in range(0, total, batch_size):
            chunk = indices[start: start + batch_size]
            p_b, d_b, pm_b, dm_b, pp, dp = cls.collate_batch([items[i] for i in chunk], extra_padding)
            out = forward_caller(p_b.to(device), d_b.to(device), pm_b.to(device), dm_b.to(device))
            logits[chunk] = out.detach().float().cpu().numpy()
            protein_pads[chunk] = pp
            drug_pads[chunk] = dp
        return logits, protein_pads, drug_pads
