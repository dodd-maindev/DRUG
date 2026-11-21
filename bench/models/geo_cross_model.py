"""GeoCross3D model combining un-interfered 3D cross-attention with decoupled pocket head."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .deep_cross_attention import DeepCrossAttentionStack
from .drug_radius_encoder import DrugRadiusEncoder
from .geometric_encoders import ProteinKnnEncoder
from .hybrid_geo_encoders import HybridGeoDrugEncoder, HybridGeoProteinEncoder
from .interface_head import LowRankInterfaceHead
from .multiscale_fusion import MultiScaleBilinearFusion


class GeoCross3DModel(nn.Module):
    """Separates clean high-AUROC interaction backbone from decoupled 3D pocket localization."""

    def __init__(
        self,
        use_geo: bool = True,
        cross_layers: int = 2,
        use_ban: bool = True,
        noise_std: float = 0.05,
        dropout: float = 0.25,
    ) -> None:
        """Initialize main 3D cross-attention backbone and decoupled 128-to-256 pocket branch."""
        super().__init__()
        self.prot_enc = HybridGeoProteinEncoder(512, dropout=dropout, use_geo=use_geo, noise_std=noise_std)
        self.drug_enc = HybridGeoDrugEncoder(512, dropout=dropout, use_geo=use_geo, noise_std=noise_std)
        self.cross_stack = DeepCrossAttentionStack(512, num_layers=cross_layers, num_heads=8, dropout=0.2)
        self.fusion = MultiScaleBilinearFusion(512, ban_dim=256, ban_heads=4, dropout=dropout, use_ban=use_ban)

        self.pocket_prot_enc = ProteinKnnEncoder(512, 128, 256, k=12, dropout=dropout)
        self.pocket_drug_enc = DrugRadiusEncoder(512, 128, 256, radius=4.5, max_deg=16, dropout=dropout)
        self.pocket_head = LowRankInterfaceHead(256, attn_dim=64, use_s0=True, use_size_interaction=True)

    @staticmethod
    def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Compute valid-token mean over non-padded (False) positions."""
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        batch: Dict[str, Any],
        return_maps: bool = False,
        use_batch_oof_b: bool = False,
    ) -> Dict[str, Any]:
        """Run clean classification backbone and parallel detached 3D pocket localizer."""
        p_mask, d_mask = batch["protein_mask"], batch["drug_mask"]
        p_h = self.prot_enc(batch["protein_emb"], batch["ca_coords"], batch["valid_ca"], batch["rho"], p_mask)
        q_h = self.drug_enc(batch["drug_emb"], batch["drug_coords"], batch["drug_is_h"], d_mask)
        p_cross, q_cross = self.cross_stack(p_h, q_h, p_mask, d_mask)
        logit_main = self.fusion(p_cross, q_cross, p_mask, d_mask)

        p_raw, d_raw = batch["protein_emb"].detach(), batch["drug_emb"].detach()
        mean_p = self._masked_mean(p_raw, p_mask)
        mean_d = self._masked_mean(d_raw, d_mask)
        p_pkt = self.pocket_prot_enc(p_raw, batch["ca_coords"], p_mask, batch["valid_ca"])
        q_pkt = self.pocket_drug_enc(d_raw, batch["drug_coords"], batch["drug_is_h"], d_mask)
        attn = self.pocket_head.compute_attention(
            p_pkt, q_pkt, batch["rho"], batch["rg_feats"], mean_p, mean_d, p_mask, d_mask
        )
        r_iface = self.pocket_head.readout_interface(p_pkt, q_pkt, attn)

        return {
            "logit": logit_main,
            "r": r_iface,
            "b": logit_main.detach(),
            "prot_attn": attn["pi_p"],
            "q_nodes": q_pkt,
            "mean_p": mean_p,
            "mean_d": mean_d,
            "lse": attn["lme"],
            "s0": attn["s0"],
        }

    def rerun_masked_protein(
        self,
        sub_batch: Dict[str, Any],
        q_pkt: torch.Tensor,
        mean_p: torch.Tensor,
        mean_d: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Re-run decoupled 128-to-256 protein kNN and interface head under residue masking."""
        p_mask, d_mask = sub_batch["protein_mask"], sub_batch["drug_mask"]
        p_pkt = self.pocket_prot_enc(
            sub_batch["protein_emb"].detach(), sub_batch["ca_coords"], p_mask, sub_batch["valid_ca"], drop_mask=drop_mask
        )
        attn = self.pocket_head.compute_attention(
            p_pkt, q_pkt, sub_batch["rho"], sub_batch["rg_feats"], mean_p, mean_d, p_mask, d_mask, drop_mask=drop_mask
        )
        return self.pocket_head.readout_interface(p_pkt, q_pkt, attn)
