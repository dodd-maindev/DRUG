"""Gaussian RBF distance basis and 1-layer 3D protein kNN encoder."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRbfEncoder(nn.Module):
    """Expands scalar 3D distances into Gaussian radial basis features."""

    def __init__(self, num_rbf: int = 16, max_dist: float = 20.0) -> None:
        """Register fixed centers and bandwidth over [0, max_dist]."""
        super().__init__()
        centers = torch.linspace(0.0, max_dist, num_rbf)
        gamma = (num_rbf / max(max_dist, 1.0)) ** 2
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute exp(-gamma * (d - mu)^2) along a trailing RBF dimension."""
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class ProteinKnnEncoder(nn.Module):
    """One-layer k=12 3D C-alpha kNN message-passing encoder with residual skip."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 512, k: int = 12, dropout: float = 0.3) -> None:
        """Initialize node projection, RBF distance basis, gated edge MLP, and output proj."""
        super().__init__()
        self.k = k
        self.use_skip = (in_dim == out_dim)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.rbf = GaussianRbfEncoder(16, max_dist=20.0)
        self.msg_net = nn.Sequential(nn.Linear(hidden_dim * 2 + 16, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.gate_net = nn.Linear(hidden_dim * 2 + 16, 1)
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim), nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, prot_mask: torch.Tensor, valid_ca: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode 3D protein residues via O(B*Lp*k) indexed gather without L^2 expansion."""
        active = (~prot_mask) if drop_mask is None else ((~prot_mask) & (~drop_mask))
        x_clean = x.masked_fill(drop_mask.unsqueeze(-1), 0.0) if drop_mask is not None else x
        h = self.in_proj(x_clean) * active.unsqueeze(-1).float()
        bsz, lp, hdim = h.shape
        k_eff = min(self.k, lp)

        with torch.no_grad():
            dist_mat = torch.cdist(coords.float(), coords.float())
            valid_key = (active & valid_ca).unsqueeze(1).expand(bsz, lp, lp)
            eye = torch.eye(lp, device=x.device, dtype=torch.bool).unsqueeze(0)
            _, knn_idx = torch.topk(dist_mat.masked_fill(~(valid_key | eye), 1e6), k=k_eff, dim=-1, largest=False)

        offset = (torch.arange(bsz, device=x.device) * lp).view(bsz, 1, 1)
        flat_idx = knn_idx + offset
        nbr_coords = coords.reshape(bsz * lp, 3)[flat_idx]
        nbr_h = h.reshape(bsz * lp, hdim)[flat_idx]
        nbr_active = active.reshape(bsz * lp)[flat_idx]
        edge_dist = torch.norm(coords.unsqueeze(2) - nbr_coords, dim=-1)
        edge_in = torch.cat([h.unsqueeze(2).expand_as(nbr_h), nbr_h, self.rbf(edge_dist)], dim=-1)

        attn = F.softmax(self.gate_net(edge_in).squeeze(-1).masked_fill(~nbr_active, -1e4), dim=-1).unsqueeze(-1)
        out = self.out_proj(h + (attn * self.msg_net(edge_in)).sum(dim=2))
        if self.use_skip:
            out = self.out_norm(x_clean + out)
        return out * active.unsqueeze(-1).float()
