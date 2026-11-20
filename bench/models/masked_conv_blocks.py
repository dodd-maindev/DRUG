"""Mask-aware Conv1d and cross-attention building blocks for clean 3DICE baselines."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareConv1dStack(nn.Module):
    """Residual 1D convolution stack with optional padding zero-masking."""

    def __init__(self, embed_dim: int = 512, num_layers: int = 2, kernel_size: int = 3, dropout_rate: float = 0.3, mask_aware: bool = True) -> None:
        """Initialize 1D convolutional layers preserving token dimensionality."""
        super().__init__()
        self.mask_aware = mask_aware
        self.layers = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply Conv1d stack with strict zeroing of padded positions when mask_aware."""
        valid = (~mask).unsqueeze(-1).float() if (self.mask_aware and mask is not None) else None
        if valid is not None:
            x = x * valid
        hidden = x.transpose(1, 2)
        for conv in self.layers:
            residual = hidden
            if valid is not None:
                hidden = hidden * valid.transpose(1, 2)
            hidden = self.dropout(F.relu(conv(hidden))) + residual
        out = hidden.transpose(1, 2)
        return out * valid if valid is not None else out


class BidirectionalCrossAttentionBlock(nn.Module):
    """Protein-drug bidirectional multi-head cross-attention matching 3DICE."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout_rate: float = 0.3) -> None:
        """Initialize bidirectional MultiheadAttention layers and feed-forward blocks."""
        super().__init__()
        self.CAp = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.CAd = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        ff_dim = embed_dim * 4
        self.ln_p1, self.ln_p2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ln_d1, self.ln_d2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ff_p = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.ff_d = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, protein_features: torch.Tensor, drug_features: torch.Tensor, protein_mask: torch.Tensor, drug_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend protein to drug and vice versa while zeroing padded token outputs."""
        att_p, attention_p = self.CAp(self.ln_p1(protein_features), drug_features, drug_features, key_padding_mask=drug_mask)
        protein_features = protein_features + self.dropout(att_p)
        protein_features = (protein_features + self.dropout(self.ff_p(self.ln_p2(protein_features)))).masked_fill(protein_mask.unsqueeze(-1), 0.0)

        att_d, _ = self.CAd(self.ln_d1(drug_features), protein_features, protein_features, key_padding_mask=protein_mask)
        drug_features = drug_features + self.dropout(att_d)
        drug_features = (drug_features + self.dropout(self.ff_d(self.ln_d2(drug_features)))).masked_fill(drug_mask.unsqueeze(-1), 0.0)
        return protein_features, drug_features, attention_p
