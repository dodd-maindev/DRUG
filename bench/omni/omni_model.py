"""Factorized gated 4-view Omni3D-DTI architecture for protein-drug interaction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from bench.omni.modality_blocks import AdaptiveModalityGate, ModalityTower
from bench.omni.omni_config import OmniConfig


class Omni3dDtiModel(nn.Module):
    """Combines ESM-2, ESM-IF1, Chem2D, and Uni-Mol views via adaptive gates."""

    def __init__(self, cfg: OmniConfig, dims: Dict[str, int]) -> None:
        """Instantiate active modality towers, entity gates, main classifier, and aux heads."""
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.prot_seq = ModalityTower(dims["p_seq"], d, cfg.in_dropout, cfg.dropout, cfg.noise_std) if cfg.use_esm2 else None
        self.prot_3d = ModalityTower(dims["p_3d"], d, cfg.in_dropout, cfg.dropout, cfg.noise_std) if cfg.use_p3d else None
        self.drug_2d = ModalityTower(dims["d_2d"], d, cfg.in_dropout, cfg.dropout, cfg.noise_std) if cfg.use_chem2d else None
        self.drug_3d = ModalityTower(dims["d_3d"], d, cfg.in_dropout, cfg.dropout, cfg.noise_std) if cfg.use_d3d else None

        self.prot_gate = AdaptiveModalityGate(d, cfg.use_gate, cfg.mod_drop_p) if (cfg.use_esm2 and cfg.use_p3d) else None
        self.drug_gate = AdaptiveModalityGate(d, cfg.use_gate, cfg.mod_drop_p) if (cfg.use_chem2d and cfg.use_d3d) else None

        self.classifier = nn.Sequential(
            nn.LayerNorm(d * 6),
            nn.Linear(d * 6, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )
        self.head_seq2d = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)) if (cfg.use_esm2 and cfg.use_chem2d) else None
        self.head_3d = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)) if (cfg.use_p3d and cfg.use_d3d) else None

    @staticmethod
    def _fuse_side(
        v_prim: Optional[torch.Tensor],
        v_3d: Optional[torch.Tensor],
        gate_layer: Optional[AdaptiveModalityGate],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fused representation and primary-view gate weight in [0, 1]."""
        if v_prim is not None and v_3d is not None and gate_layer is not None:
            return gate_layer(v_prim, v_3d)
        if v_prim is not None:
            return v_prim, torch.ones(v_prim.size(0), device=v_prim.device, dtype=v_prim.dtype)
        assert v_3d is not None
        return v_3d, torch.zeros(v_3d.size(0), device=v_3d.device, dtype=v_3d.dtype)

    def forward(
        self,
        p_seq: torch.Tensor,
        p_3d: torch.Tensor,
        d_2d: torch.Tensor,
        d_3d: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Predict main logit, auxiliary view logits, view embeddings, and gate weights."""
        hp_seq = self.prot_seq(p_seq) if self.prot_seq is not None else None
        hp_3d = self.prot_3d(p_3d) if self.prot_3d is not None else None
        hd_2d = self.drug_2d(d_2d) if self.drug_2d is not None else None
        hd_3d = self.drug_3d(d_3d) if self.drug_3d is not None else None

        hp, gate_p = self._fuse_side(hp_seq, hp_3d, self.prot_gate)
        hd, gate_d = self._fuse_side(hd_2d, hd_3d, self.drug_gate)

        zeros = torch.zeros_like(hp)
        c_seq2d = (hp_seq * hd_2d) if (hp_seq is not None and hd_2d is not None) else zeros
        c_3d = (hp_3d * hd_3d) if (hp_3d is not None and hd_3d is not None) else zeros
        joint = torch.cat([hp, hd, hp * hd, (hp - hd).abs(), c_seq2d, c_3d], dim=-1)

        return {
            "logit": self.classifier(joint).squeeze(-1),
            "hp_seq": hp_seq,
            "hp_3d": hp_3d,
            "hd_2d": hd_2d,
            "hd_3d": hd_3d,
            "logit_seq2d": self.head_seq2d(c_seq2d).squeeze(-1) if self.head_seq2d is not None else None,
            "logit_3d": self.head_3d(c_3d).squeeze(-1) if self.head_3d is not None else None,
            "gate_p": gate_p,
            "gate_d": gate_d,
        }
