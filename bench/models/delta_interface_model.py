"""3D Cross-Modal Delta-Interface model combining global cross-attn with 3D pocket."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from bench.models.drug_radius_encoder import DrugRadiusEncoder
from bench.models.geometric_encoders import ProteinKnnEncoder
from bench.models.interface_head import LowRankInterfaceHead
from bench.models.masked_conv_blocks import BidirectionalCrossAttentionBlock


class DeltaInterfaceModel(nn.Module):
    """Un-starved 3D Cross-Modal network fusing global cross-attn with local pocket."""

    def __init__(self, tau_b: float = 1.0, stop_grad_b: bool = True, use_s0: bool = True, use_size_interaction: bool = True, coarse_state: Optional[dict] = None) -> None:
        """Initialize 3D encoders, cross-attention, pocket head, and global readout."""
        super().__init__()
        self.prot_enc = ProteinKnnEncoder(512, 256, 512, dropout=0.3)
        self.drug_enc = DrugRadiusEncoder(512, 256, 512, dropout=0.3)
        self.cross_attn = BidirectionalCrossAttentionBlock(512, num_heads=4, dropout_rate=0.3)
        self.head = LowRankInterfaceHead(512, 64, use_s0=use_s0, use_size_interaction=use_size_interaction)
        self.global_mlp = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 1),
        )
        self.stop_grad_b = stop_grad_b
        self.register_buffer("tau_b", torch.tensor(max(float(tau_b), 1e-6), dtype=torch.float32))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute valid-token mean embeddings."""
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, batch: Dict[str, Any], return_maps: bool = False, use_batch_oof_b: bool = False) -> Dict[str, Any]:
        """Compute 3D pocket residual r, cross-attended global logit, and combined logit."""
        p_mask, d_mask = batch["protein_mask"], batch["drug_mask"]
        mean_p = self._masked_mean(batch["protein_emb"], p_mask)
        mean_d = self._masked_mean(batch["drug_emb"], d_mask)

        p_geo = self.prot_enc(batch["protein_emb"], batch["ca_coords"], p_mask, batch["valid_ca"])
        q_geo = self.drug_enc(batch["drug_emb"], batch["drug_coords"], batch["drug_is_h"], d_mask)
        attn = self.head.compute_attention(p_geo, q_geo, batch["rho"], batch["rg_feats"], mean_p, mean_d, p_mask, d_mask)
        r_iface = self.head.readout_interface(p_geo, q_geo, attn)

        p_cross, q_cross, _ = self.cross_attn(p_geo, q_geo, p_mask, d_mask)
        g_p = self._masked_mean(p_cross, p_mask)
        g_d = self._masked_mean(q_cross, d_mask)
        z_pc = torch.bmm(attn["pi_p"].unsqueeze(1), p_cross.float()).squeeze(1)
        z_dc = torch.bmm(attn["pi_d"].unsqueeze(1), q_cross.float()).squeeze(1)
        b_cross = self.global_mlp(torch.cat([g_p.float(), g_d.float(), z_pc, z_dc], dim=-1)).squeeze(-1)

        logit = b_cross + 0.5 * r_iface
        return {
            "logit": logit, "r": r_iface, "b": b_cross, "prot_attn": attn["pi_p"],
            "q_nodes": q_geo, "mean_p": mean_p, "mean_d": mean_d, "lse": attn["lme"], "s0": attn["s0"],
        }

    def rerun_masked_protein(self, sub_batch: Dict[str, Any], q_geo: torch.Tensor, mean_p: torch.Tensor, mean_d: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
        """Re-run only 1-layer 3D protein kNN + interface head for fast counterfactual checks."""
        p_mask, d_mask = sub_batch["protein_mask"], sub_batch["drug_mask"]
        p_geo = self.prot_enc(sub_batch["protein_emb"], sub_batch["ca_coords"], p_mask, sub_batch["valid_ca"], drop_mask=drop_mask)
        attn = self.head.compute_attention(p_geo, q_geo, sub_batch["rho"], sub_batch["rg_feats"], mean_p, mean_d, p_mask, d_mask, drop_mask=drop_mask)
        return self.head.readout_interface(p_geo, q_geo, attn)
