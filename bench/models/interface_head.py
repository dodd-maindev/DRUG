"""Normalized low-rank interface head producing 3D pocket attention and residual r."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankInterfaceHead(nn.Module):
    """Computes local 3D pocket attention pi_ia, null diff, and pocket residual r."""

    def __init__(self, node_dim: int = 512, attn_dim: int = 64, use_s0: bool = True, use_size_interaction: bool = True) -> None:
        """Initialize attention keys, size affinity, null MLP, and pocket readout MLP."""
        super().__init__()
        self.use_s0 = use_s0
        self.use_size = use_size_interaction
        self.p_key = nn.Linear(node_dim, attn_dim, bias=False)
        self.q_key = nn.Linear(node_dim, attn_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(1.386, dtype=torch.float32))
        self.size_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.c_p = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 16))
        self.c_d = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 16))
        self.s0_net = nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        self.inter_mlp = nn.Sequential(
            nn.LayerNorm(node_dim * 3 + 1),
            nn.Linear(node_dim * 3 + 1, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.s0_net[-1].weight)
        nn.init.zeros_(self.s0_net[-1].bias)

    def compute_attention(
        self, p_geo: torch.Tensor, q_geo: torch.Tensor, rho: torch.Tensor, rg_feats: torch.Tensor,
        mean_p: torch.Tensor, mean_d: torch.Tensor, p_mask: torch.Tensor, d_mask: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute local 3D pocket distribution pi_ia, pi_p, pi_d, and scalar null diff."""
        with torch.amp.autocast("cuda", enabled=False):
            pf, qf = p_geo.float(), q_geo.float()
            p_norm = F.normalize(self.p_key(pf), dim=-1)
            q_norm = F.normalize(self.q_key(qf), dim=-1)
            s_ia = torch.bmm(p_norm, q_norm.transpose(1, 2)) * self.log_scale.exp().clamp(1.0, 16.0)
            if self.use_size:
                cp_n = F.normalize(self.c_p(rho.float()), dim=-1)
                cd_n = F.normalize(self.c_d(rg_feats.float()), dim=-1)
                s_ia = s_ia + self.size_scale.clamp(-2.0, 2.0) * torch.einsum("bpk,bk->bp", cp_n, cd_n).unsqueeze(-1)

            eff_p_mask = p_mask if drop_mask is None else (p_mask | drop_mask)
            invalid_pair = eff_p_mask.unsqueeze(-1) | d_mask.unsqueeze(1)
            s_masked = s_ia.masked_fill(invalid_pair, -1e4)
            bsz, lp, ld = s_masked.shape
            flat_s = s_masked.view(bsz, lp * ld)
            n_valid = (~invalid_pair).view(bsz, lp * ld).sum(dim=-1).float().clamp(min=1.0)
            lme = torch.logsumexp(flat_s, dim=-1) - torch.log(n_valid)
            pi_ia = F.softmax(flat_s, dim=-1).view(bsz, lp, ld)
            s0 = self.s0_net(torch.cat([mean_p.float(), mean_d.float()], dim=-1)).squeeze(-1) if self.use_s0 else torch.zeros_like(lme)
            return {"pi_ia": pi_ia, "pi_p": pi_ia.sum(dim=-1), "pi_d": pi_ia.sum(dim=1), "lme": lme, "s0": s0, "r_scalar": lme - s0}

    def readout_interface(self, p_geo: torch.Tensor, q_geo: torch.Tensor, attn: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute local 3D pocket residual score from 3D nodes and interface map."""
        with torch.amp.autocast("cuda", enabled=False):
            pf, qf = p_geo.float(), q_geo.float()
            z_p = torch.bmm(attn["pi_p"].unsqueeze(1), pf).squeeze(1)
            z_d = torch.bmm(attn["pi_d"].unsqueeze(1), qf).squeeze(1)
            z_contact = (pf * torch.bmm(attn["pi_ia"], qf)).sum(dim=1)
            r_vec = self.inter_mlp(torch.cat([z_p, z_d, z_contact, attn["r_scalar"].unsqueeze(-1)], dim=-1)).squeeze(-1)
            return attn["r_scalar"] + r_vec
