"""C2 'Lego++' baseline using the same 3D encoders without Delta-Interface components."""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.models.drug_radius_encoder import DrugRadiusEncoder
from bench.models.geometric_encoders import ProteinKnnEncoder


class Lego3DModel(nn.Module):
    """Capacity-matched 3D GNN + MIL attention baseline trained purely with BCE."""

    def __init__(self, node_dim: int = 256, dropout: float = 0.25) -> None:
        """Initialize matched 3D encoders, bilinear MIL pooling, and MLP classifier."""
        super().__init__()
        self.prot_enc = ProteinKnnEncoder(hidden_dim=node_dim, out_dim=node_dim, dropout=dropout)
        self.drug_enc = DrugRadiusEncoder(hidden_dim=node_dim, out_dim=node_dim, dropout=dropout)
        self.pair_proj = nn.Linear(node_dim, 64, bias=False)
        self.q_proj = nn.Linear(node_dim, 64, bias=False)
        self.scale = 8.0
        self.classifier = nn.Sequential(
            nn.Linear(node_dim * 2 + 1024, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute valid-token mean embeddings."""
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, batch: Dict[str, Any], return_maps: bool = False) -> Dict[str, Any]:
        """Compute scalar interaction logit and MIL marginal protein attention."""
        p_mask, d_mask = batch["protein_mask"], batch["drug_mask"]
        p_nodes = self.prot_enc(batch["protein_emb"], batch["ca_coords"], p_mask, batch["valid_ca"])
        q_nodes = self.drug_enc(batch["drug_emb"], batch["drug_coords"], batch["drug_is_h"], d_mask)

        s_ia = torch.bmm(self.pair_proj(p_nodes), self.q_proj(q_nodes).transpose(1, 2)) / self.scale
        invalid = p_mask.unsqueeze(-1) | d_mask.unsqueeze(1)
        s_masked = s_ia.masked_fill(invalid, -1e4)
        bsz, lp, ld = s_masked.shape
        pi_ia = F.softmax(s_masked.view(bsz, lp * ld), dim=-1).view(bsz, lp, ld)
        pi_p = pi_ia.sum(dim=-1)
        pi_d = pi_ia.sum(dim=1)

        p_pool = torch.bmm(pi_p.unsqueeze(1), p_nodes).squeeze(1)
        d_pool = torch.bmm(pi_d.unsqueeze(1), q_nodes).squeeze(1)
        mean_p = self._masked_mean(batch["protein_emb"], p_mask)
        mean_d = self._masked_mean(batch["drug_emb"], d_mask)
        logit = self.classifier(torch.cat([p_pool, d_pool, mean_p, mean_d], dim=-1)).squeeze(-1)
        return {"logit": logit, "prot_attn": pi_p}
