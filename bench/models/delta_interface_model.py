"""Delta-Interface model combining frozen coarse b with 3D cross-modal residual r."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from bench.models.drug_radius_encoder import DrugRadiusEncoder
from bench.models.geometric_encoders import ProteinKnnEncoder
from bench.models.interface_head import LowRankInterfaceHead
from bench.models.masked_conv_blocks import BidirectionalCrossAttentionBlock
from research.coarse_model import CoarseInteractionNetwork


class DeltaInterfaceModel(nn.Module):
    """Full 3D Delta-Interface network predicting ell = b / tau_b + r."""

    def __init__(self, tau_b: float = 1.0, stop_grad_b: bool = True, use_s0: bool = True, use_size_interaction: bool = True, coarse_state: Optional[dict] = None) -> None:
        """Initialize 3D encoders, cross-attention, interface head, and coarse branch b."""
        super().__init__()
        self.prot_enc = ProteinKnnEncoder(512, 256, 512, dropout=0.3)
        self.drug_enc = DrugRadiusEncoder(512, 256, 512, dropout=0.3)
        self.cross_attn = BidirectionalCrossAttentionBlock(512, num_heads=4, dropout_rate=0.3)
        self.head = LowRankInterfaceHead(512, 64, use_s0=use_s0, use_size_interaction=use_size_interaction)
        self.stop_grad_b = stop_grad_b
        self.register_buffer("tau_b", torch.tensor(max(float(tau_b), 1e-6), dtype=torch.float32))
        self.b_net = CoarseInteractionNetwork()
        if coarse_state is not None:
            self.b_net.load_state_dict(coarse_state)
        if stop_grad_b:
            for param in self.b_net.parameters():
                param.requires_grad = False

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute valid-token mean embeddings for coarse and null inputs."""
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, batch: Dict[str, Any], return_maps: bool = False, use_batch_oof_b: bool = False) -> Dict[str, Any]:
        """Compute coarse logit b, 3D pocket attention pi_p, cross-attended r, and total."""
        p_mask, d_mask = batch["protein_mask"], batch["drug_mask"]
        mean_p = self._masked_mean(batch["protein_emb"], p_mask)
        mean_d = self._masked_mean(batch["drug_emb"], d_mask)
        b_raw = batch["b_logit"] if (use_batch_oof_b and self.stop_grad_b) else self.b_net(mean_d, mean_p)
        b_scaled = b_raw.float() / self.tau_b
        if self.stop_grad_b:
            b_scaled = b_scaled.detach()

        p_geo = self.prot_enc(batch["protein_emb"], batch["ca_coords"], p_mask, batch["valid_ca"])
        q_geo = self.drug_enc(batch["drug_emb"], batch["drug_coords"], batch["drug_is_h"], d_mask)
        attn = self.head.compute_attention(p_geo, q_geo, batch["rho"], batch["rg_feats"], mean_p, mean_d, p_mask, d_mask)
        p_cross, q_cross, _ = self.cross_attn(p_geo, q_geo, p_mask, d_mask)
        r = self.head.readout(p_cross, q_cross, attn)
        return {
            "logit": b_scaled + r, "r": r, "b": b_scaled, "prot_attn": attn["pi_p"],
            "q_nodes": q_geo, "mean_p": mean_p, "mean_d": mean_d, "lse": attn["lme"], "s0": attn["s0"],
        }

    def rerun_masked_protein(self, sub_batch: Dict[str, Any], q_geo: torch.Tensor, mean_p: torch.Tensor, mean_d: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
        """Re-run 3D kNN and cross-attention with counterfactual-dropped residues excluded."""
        p_mask, d_mask = sub_batch["protein_mask"], sub_batch["drug_mask"]
        eff_mask = p_mask | drop_mask
        p_geo = self.prot_enc(sub_batch["protein_emb"], sub_batch["ca_coords"], p_mask, sub_batch["valid_ca"], drop_mask=drop_mask)
        attn = self.head.compute_attention(p_geo, q_geo, sub_batch["rho"], sub_batch["rg_feats"], mean_p, mean_d, p_mask, d_mask, drop_mask=drop_mask)
        p_cross, q_cross, _ = self.cross_attn(p_geo, q_geo, eff_mask, d_mask)
        return self.head.readout(p_cross, q_cross, attn)
