"""Modality encoder tower and adaptive stochastic gate blocks for Omni3D-DTI."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ModalityTower(nn.Module):
    """Projects one biological or chemical representation into a shared latent space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dropout: float = 0.20,
        dropout: float = 0.35,
        noise_std: float = 0.03,
    ) -> None:
        """Initialize Pre-LN input regularization and two-layer GELU residual projection."""
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.noise_std = noise_std
        self.drop_in = nn.Dropout(in_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input, add optional training noise, and project to latent space."""
        h = self.norm_in(x)
        if self.training and self.noise_std > 0.0:
            h = h + torch.randn_like(h) * self.noise_std
        return self.mlp(self.drop_in(h))


class AdaptiveModalityGate(nn.Module):
    """Fuses primary (Seq/2D) and 3D views via context gating and modality dropout."""

    def __init__(self, dim: int, use_gate: bool = True, mod_drop_p: float = 0.15) -> None:
        """Initialize context-aware gate network and stochastic modality-drop rate."""
        super().__init__()
        self.use_gate = use_gate
        self.mod_drop_p = mod_drop_p
        if use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim * 4, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )

    def forward(self, v_primary: torch.Tensor, v_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gated blend of two views and return fused tensor plus mean gate weight."""
        bsz = v_primary.size(0)
        if self.use_gate:
            ctx = torch.cat([v_primary, v_3d, v_primary * v_3d, (v_primary - v_3d).abs()], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(ctx))
        else:
            gate = torch.full_like(v_primary, 0.5)

        gate_eff = gate
        if self.training and self.mod_drop_p > 0.0:
            draw = torch.rand(bsz, 1, device=v_primary.device)
            only_prim = draw < self.mod_drop_p
            only_3d = (draw >= self.mod_drop_p) & (draw < 1.67 * self.mod_drop_p)
            gate_eff = torch.where(only_prim, torch.ones_like(gate), gate)
            gate_eff = torch.where(only_3d, torch.zeros_like(gate_eff), gate_eff)

        fused = gate_eff * v_primary + (1.0 - gate_eff) * v_3d
        return fused, gate.mean(dim=-1)
