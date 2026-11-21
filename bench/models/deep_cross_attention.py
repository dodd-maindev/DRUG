"""Deep Pre-LN bidirectional cross-attention with LayerScale for DTI."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LayerScaledCrossAttentionBlock(nn.Module):
    """Symmetric Pre-LN protein-drug cross-attention block with learnable LayerScale."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
        mlp_ratio: int = 2,
        layer_scale_init: float = 0.1,
    ) -> None:
        """Initialize Pre-LN multi-head cross-attention, feed-forward blocks, and LayerScale."""
        super().__init__()
        self.norm_p1 = nn.LayerNorm(embed_dim)
        self.norm_q1 = nn.LayerNorm(embed_dim)
        self.attn_p2q = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_q2p = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gamma_p1 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))
        self.gamma_q1 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))

        hidden = embed_dim * mlp_ratio
        self.norm_p2 = nn.LayerNorm(embed_dim)
        self.norm_q2 = nn.LayerNorm(embed_dim)
        self.ffn_p = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_q = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout)
        )
        self.gamma_p2 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))
        self.gamma_q2 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))

    def forward(
        self,
        p_feat: torch.Tensor,
        q_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exchange bidirectional cross-modal context with True=PAD masks."""
        valid_p = (~prot_mask).unsqueeze(-1).float()
        valid_q = (~drug_mask).unsqueeze(-1).float()
        p_norm = self.norm_p1(p_feat)
        q_norm = self.norm_q1(q_feat)
        p_ctx, _ = self.attn_p2q(p_norm, q_norm, q_norm, key_padding_mask=drug_mask, need_weights=False)
        q_ctx, _ = self.attn_q2p(q_norm, p_norm, p_norm, key_padding_mask=prot_mask, need_weights=False)

        p_mid = p_feat + self.gamma_p1 * p_ctx
        q_mid = q_feat + self.gamma_q1 * q_ctx
        p_out = (p_mid + self.gamma_p2 * self.ffn_p(self.norm_p2(p_mid))) * valid_p
        q_out = (q_mid + self.gamma_q2 * self.ffn_q(self.norm_q2(q_mid))) * valid_q
        return p_out, q_out


class DeepCrossAttentionStack(nn.Module):
    """Multi-layer stack of Pre-LN LayerScaled bidirectional cross-attention blocks."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        """Build a sequence of symmetric cross-attention blocks."""
        super().__init__()
        self.layers = nn.ModuleList(
            [LayerScaledCrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        p_feat: torch.Tensor,
        q_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate protein and drug token states through all cross-attention layers."""
        for layer in self.layers:
            p_feat, q_feat = layer(p_feat, q_feat, prot_mask, drug_mask)
        return p_feat, q_feat
