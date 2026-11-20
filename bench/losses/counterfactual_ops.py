"""Vectorized counterfactual perturbation and spatial compactness penalties."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


class CounterfactualRegularizer:
    """Computes L_nec, L_spec (via kNN re-run) and L_compact on positive pairs."""

    @staticmethod
    def _sample_masks(pi_p: torch.Tensor, prot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized construction of top-m and disjoint random-m residue masks."""
        bsz, lp = pi_p.shape
        device = pi_p.device
        valid = ~prot_mask
        n_valid = valid.sum(dim=1)
        m_per = torch.clamp((0.08 * n_valid.float()).to(torch.long), min=3, max=15)
        m_per = torch.minimum(m_per, n_valid // 3)
        max_k = min(15, lp)

        ranks_keep = torch.arange(max_k, device=device).unsqueeze(0) < m_per.unsqueeze(1)
        scores = pi_p.detach().masked_fill(prot_mask, -1e9)
        _, top_idx = torch.topk(scores, k=max_k, dim=1)
        top_mask = torch.zeros_like(prot_mask)
        top_mask.scatter_(1, top_idx, ranks_keep)

        rand_scores = torch.rand_like(scores).masked_fill(prot_mask | top_mask, -1.0)
        _, rand_idx = torch.topk(rand_scores, k=max_k, dim=1)
        rand_mask = torch.zeros_like(prot_mask)
        rem_valid = (~(prot_mask | top_mask)).gather(1, rand_idx)
        rand_mask.scatter_(1, rand_idx, ranks_keep & rem_valid)
        return top_mask, rand_mask

    @classmethod
    def counterfactual_losses(
        cls, model: Any, batch: Dict[str, Any], out: Dict[str, Any],
        max_pairs: int = 12, margin_nec: float = 0.5, delta_spec: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-run protein kNN under top-m and non-pocket deletion on y=1 pairs."""
        pos_idx = (batch["label"] == 1).nonzero(as_tuple=True)[0]
        device = out["r"].device
        if pos_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        if pos_idx.numel() > max_pairs:
            pos_idx = pos_idx[torch.randperm(int(pos_idx.numel()), device=device)[:max_pairs]]

        sub = {k: v[pos_idx] if torch.is_tensor(v) else [v[int(j)] for j in pos_idx] for k, v in batch.items()}
        r_pos = out["r"][pos_idx]
        top_mask, rand_mask = cls._sample_masks(out["prot_attn"][pos_idx], sub["protein_mask"])
        q_sub, mp_sub, md_sub = out["q_nodes"][pos_idx], out["mean_p"][pos_idx], out["mean_d"][pos_idx]

        r_drop_top = model.rerun_masked_protein(sub, q_sub, mp_sub, md_sub, top_mask)
        r_drop_rand = model.rerun_masked_protein(sub, q_sub, mp_sub, md_sub, rand_mask)
        l_nec = F.relu(margin_nec - (r_pos - r_drop_top)).mean()
        l_spec = F.relu((r_pos - r_drop_rand).abs() - delta_spec).mean()
        return l_nec, l_spec

    @staticmethod
    def compactness_loss(batch: Dict[str, Any], out: Dict[str, Any]) -> torch.Tensor:
        """Penalize 3D spatial attention radius exceeding R_d = r_g(d) + 6A on y=1."""
        pos_mask = (batch["label"] == 1) & (batch["valid_ca"].sum(dim=-1) >= 3)
        device = out["r"].device
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        weights = out["prot_attn"][pos_mask] * ((~batch["protein_mask"][pos_mask]) & batch["valid_ca"][pos_mask]).float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        ca = batch["ca_coords"][pos_mask]
        mu = torch.einsum("bp,bpc->bc", weights, ca)
        var = (weights * ((ca - mu.unsqueeze(1)) ** 2).sum(dim=-1)).sum(dim=-1)
        radius_limit = batch["rg"][pos_mask] + 6.0
        return F.relu(torch.sqrt(var + 1e-6) - radius_limit).pow(2).mean()
