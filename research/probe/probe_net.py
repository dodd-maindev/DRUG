"""Dual-tower residual bilinear network for fast multi-modal signal probing."""

from __future__ import annotations

import torch
from torch import nn


class ModalityProbeNet(nn.Module):
    """Fuses protein tower, drug tower, Hadamard, L1 diff, and optional pair signals."""

    def __init__(self, p_dim: int, d_dim: int, pair_dim: int = 0, hidden: int = 512, dropout: float = 0.35) -> None:
        """Initialize modality projections, bilinear interaction head, and residual classifier."""
        super().__init__()
        self.p_proj = nn.Sequential(nn.LayerNorm(p_dim), nn.Linear(p_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.d_proj = nn.Sequential(nn.LayerNorm(d_dim), nn.Linear(d_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        in_dim = hidden * 4 + pair_dim
        self.pair_norm = nn.LayerNorm(pair_dim) if pair_dim > 0 else None
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, p_x: torch.Tensor, d_x: torch.Tensor, pair_x: torch.Tensor) -> torch.Tensor:
        """Compute binary binding logit from projected towers and optional pair signals."""
        hp = self.p_proj(p_x)
        hd = self.d_proj(d_x)
        parts = [hp, hd, hp * hd, (hp - hd).abs()]
        if self.pair_norm is not None:
            parts.append(self.pair_norm(pair_x))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)
