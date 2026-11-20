"""One-layer 4.5A 3D radius graph encoder preserving ligand Hydrogens."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.models.geometric_encoders import GaussianRbfEncoder


class DrugRadiusEncoder(nn.Module):
    """One-layer 4.5A 3D radius graph encoder preserving Hydrogens with is_H flag."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128, out_dim: int = 64, radius: float = 4.5, max_deg: int = 16, dropout: float = 0.1) -> None:
        """Initialize hydrogen embedding, RBF distance encoder, and radius attention."""
        super().__init__()
        self.radius, self.max_deg = radius, max_deg
        self.h_type_emb = nn.Embedding(2, in_dim)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.rbf = GaussianRbfEncoder(16, max_dist=6.0)
        self.msg_net = nn.Sequential(nn.Linear(hidden_dim * 2 + 16, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.gate_net = nn.Linear(hidden_dim * 2 + 16, 1)
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, coords: torch.Tensor, is_h: torch.Tensor, drug_mask: torch.Tensor) -> torch.Tensor:
        """Encode 3D molecular atoms within 4.5 Angstrom neighborhoods."""
        active = ~drug_mask
        h = self.in_proj(x + self.h_type_emb(is_h.long())) * active.unsqueeze(-1).float()
        bsz, ld, _ = h.shape
        k_eff = min(self.max_deg, ld)

        with torch.no_grad():
            dist_mat = torch.cdist(coords, coords)
            in_rad = (dist_mat <= self.radius) & active.unsqueeze(1)
            eye = torch.eye(ld, device=x.device, dtype=torch.bool).unsqueeze(0)
            dist_masked = dist_mat.masked_fill(~(in_rad | eye), 1e6)
            _, nbr_idx = torch.topk(dist_masked, k=k_eff, dim=-1, largest=False)

        nbr_coords = torch.gather(coords.unsqueeze(1).expand(bsz, ld, ld, 3), 2, nbr_idx.unsqueeze(-1).expand(bsz, ld, k_eff, 3))
        edge_dist = torch.norm(coords.unsqueeze(2) - nbr_coords, dim=-1)
        nbr_h = torch.gather(h.unsqueeze(1).expand(bsz, ld, ld, h.size(-1)), 2, nbr_idx.unsqueeze(-1).expand(bsz, ld, k_eff, h.size(-1)))
        edge_in = torch.cat([h.unsqueeze(2).expand_as(nbr_h), nbr_h, self.rbf(edge_dist)], dim=-1)

        nbr_valid = (edge_dist <= self.radius) & torch.gather(active, 1, nbr_idx.view(bsz, -1)).view(bsz, ld, k_eff)
        attn = F.softmax(self.gate_net(edge_in).squeeze(-1).masked_fill(~nbr_valid, -1e4), dim=-1).unsqueeze(-1)
        agg = (attn * self.msg_net(edge_in)).sum(dim=2)
        return self.out_proj(h + agg) * active.unsqueeze(-1).float()
