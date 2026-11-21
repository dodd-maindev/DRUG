"""Hybrid 1D sequence + 3D spatial encoders for proteins and small molecules."""

from __future__ import annotations

import torch
from torch import nn

from .drug_radius_encoder import DrugRadiusEncoder
from .geometric_encoders import ProteinKnnEncoder
from .masked_conv_blocks import MaskAwareConv1dStack


def _masked_noise(tensor: torch.Tensor, pad_mask: torch.Tensor, std: float, training: bool) -> torch.Tensor:
    """Apply Gaussian feature noise strictly inside non-padded (False) tokens."""
    valid = (~pad_mask).unsqueeze(-1).float()
    clean = tensor * valid
    if training and std > 0.0:
        clean = (clean + torch.randn_like(clean) * std) * valid
    return clean


class HybridGeoProteinEncoder(nn.Module):
    """Encodes proteins with mask-aware 1D CNN plus optional 3D C-alpha k-NN."""

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.25,
        use_geo: bool = True,
        noise_std: float = 0.05,
        k_neighbors: int = 12,
    ) -> None:
        """Initialize sequence convolution, 3D C-alpha k-NN message passing, and rho projection."""
        super().__init__()
        self.use_geo = use_geo
        self.noise_std = noise_std
        self.seq_conv = MaskAwareConv1dStack(embed_dim, num_layers=2, kernel_size=3, dropout_rate=dropout, mask_aware=True)
        if use_geo:
            self.knn = ProteinKnnEncoder(embed_dim, embed_dim // 2, embed_dim, k=k_neighbors, dropout=dropout)
            self.rho_proj = nn.Linear(4, embed_dim)
            self.rho_scale = nn.Parameter(torch.full((embed_dim,), 0.1))

    def forward(
        self,
        p_esm: torch.Tensor,
        ca_coords: torch.Tensor,
        valid_ca: torch.Tensor,
        rho: torch.Tensor,
        prot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode residues (True=PAD in prot_mask) preserving tokens lacking 3D coords."""
        valid = (~prot_mask).unsqueeze(-1).float()
        p_clean = _masked_noise(p_esm, prot_mask, self.noise_std, self.training)
        h_seq = self.seq_conv(p_clean, mask=prot_mask)
        if not self.use_geo:
            return h_seq * valid
        h_geo = self.knn(h_seq, ca_coords, prot_mask, valid_ca)
        has_geo = ((~prot_mask) & valid_ca).unsqueeze(-1)
        h_merged = torch.where(has_geo, h_geo, h_seq.to(h_geo.dtype))
        rho_feat = self.rho_scale * self.rho_proj(rho.to(h_merged.dtype))
        return (h_merged + rho_feat) * valid


class HybridGeoDrugEncoder(nn.Module):
    """Encodes ligands with mask-aware 1D CNN plus optional 3D 4.5A radius graph."""

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.25,
        use_geo: bool = True,
        noise_std: float = 0.05,
        radius: float = 4.5,
    ) -> None:
        """Initialize atom sequence convolution and 3D all-atom radius graph encoder."""
        super().__init__()
        self.use_geo = use_geo
        self.noise_std = noise_std
        self.seq_conv = MaskAwareConv1dStack(embed_dim, num_layers=2, kernel_size=3, dropout_rate=dropout, mask_aware=True)
        if use_geo:
            self.radius_enc = DrugRadiusEncoder(embed_dim, embed_dim // 2, embed_dim, radius=radius, max_deg=16, dropout=dropout)

    def forward(
        self,
        d_unimol: torch.Tensor,
        drug_coords: torch.Tensor,
        is_h: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode drug atoms (True=PAD in drug_mask) with Hydrogens and 3D neighborhoods."""
        valid = (~drug_mask).unsqueeze(-1).float()
        d_clean = _masked_noise(d_unimol, drug_mask, self.noise_std, self.training)
        h_seq = self.seq_conv(d_clean, mask=drug_mask)
        if not self.use_geo:
            return h_seq * valid
        return self.radius_enc(h_seq, drug_coords, is_h, drug_mask) * valid
