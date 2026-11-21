"""Multi-scale Mean + Max + Bilinear Attention Network (BAN) fusion classifier."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean_pool(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Compute exact masked mean over non-padded (False) tokens."""
    valid = (~pad_mask).unsqueeze(-1).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def _masked_max_pool(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Compute masked max over non-padded tokens to preserve localized binding peaks."""
    return x.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(dim=1).values


class MultiScaleBilinearFusion(nn.Module):
    """Fuses masked Mean, Max, and multi-head Bilinear Attention interaction pooling."""

    def __init__(
        self,
        embed_dim: int = 512,
        ban_dim: int = 256,
        ban_heads: int = 4,
        dropout: float = 0.25,
        use_ban: bool = True,
    ) -> None:
        """Initialize low-rank bilinear attention heads and interaction classification MLP."""
        super().__init__()
        self.use_ban = use_ban
        self.ban_heads = ban_heads
        self.head_dim = ban_dim // ban_heads
        self.dropout = dropout
        if use_ban:
            self.ban_p = nn.Linear(embed_dim, ban_dim)
            self.ban_q = nn.Linear(embed_dim, ban_dim)
        in_dim = embed_dim * 4 + (ban_dim if use_ban else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _bilinear_pool(
        self,
        p_feat: torch.Tensor,
        q_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head low-rank bilinear attention pooling in fp32 for stability."""
        bsz, n_res, _ = p_feat.shape
        n_atom = q_feat.shape[1]
        u = F.dropout(F.gelu(self.ban_p(p_feat)), p=self.dropout, training=self.training)
        v = F.dropout(F.gelu(self.ban_q(q_feat)), p=self.dropout, training=self.training)
        with torch.amp.autocast("cuda", enabled=False):
            uh = u.float().view(bsz, n_res, self.ban_heads, self.head_dim).permute(0, 2, 1, 3)
            vh = v.float().view(bsz, n_atom, self.ban_heads, self.head_dim).permute(0, 2, 1, 3)
            logits = torch.matmul(uh, vh.transpose(-1, -2)) / math.sqrt(self.head_dim)
            pair_invalid = (prot_mask.unsqueeze(2) | drug_mask.unsqueeze(1)).unsqueeze(1)
            logits = logits.masked_fill(pair_invalid, -1e4)
            attn = F.softmax(logits.flatten(-2), dim=-1).view(bsz, self.ban_heads, n_res, n_atom)
            pooled = torch.einsum("bhnd,bhnm,bhmd->bhd", uh, attn, vh).reshape(bsz, -1)
        return pooled.to(p_feat.dtype)

    def forward(
        self,
        p_feat: torch.Tensor,
        q_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict DTI binding logit from global mean, peak max, and bilinear pools."""
        feats = [
            _masked_mean_pool(p_feat, prot_mask),
            _masked_mean_pool(q_feat, drug_mask),
            _masked_max_pool(p_feat, prot_mask),
            _masked_max_pool(q_feat, drug_mask),
        ]
        if self.use_ban:
            feats.append(self._bilinear_pool(p_feat, q_feat, prot_mask, drug_mask))
        return self.classifier(torch.cat(feats, dim=-1)).squeeze(-1)
