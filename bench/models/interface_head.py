"""Normalized low-rank interface head with 3D pocket vector readout and null s0."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankInterfaceHead(nn.Module):
    """Low-rank interface map producing both scalar LogMeanExp and pocket vector residual."""

    def __init__(self, node_dim: int = 256, attn_dim: int = 64, use_s0: bool = True, use_size_interaction: bool = True) -> None:
        """Initialize attention projections, size bias, null MLP, and readout MLP."""
        super().__init__()
        self.use_s0 = use_s0
        self.use_size = use_size_interaction
        self.p_key = nn.Linear(node_dim, attn_dim, bias=False)
        self.q_key = nn.Linear(node_dim, attn_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(1.386, dtype=torch.float32))
        self.size_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.c_p = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 16))
        self.c_d = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 16))
        self.s0_net = nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Dropout(0.25), nn.Linear(128, 1))
        self.inter_mlp = nn.Sequential(
            nn.LayerNorm(node_dim * 3 + 1),
            nn.Linear(node_dim * 3 + 1, 256), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.s0_net[-1].weight)
        nn.init.zeros_(self.s0_net[-1].bias)
        nn.init.zeros_(self.inter_mlp[-1].weight)
        nn.init.zeros_(self.inter_mlp[-1].bias)

    def forward(
        self, p_nodes: torch.Tensor, q_nodes: torch.Tensor, rho: torch.Tensor, rg_feats: torch.Tensor,
        mean_p: torch.Tensor, mean_d: torch.Tensor, p_mask: torch.Tensor, d_mask: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute bounded pairwise attention pi_ia, pocket readout [z_p, z_d], and r."""
        with torch.amp.autocast("cuda", enabled=False):
            pf, qf = p_nodes.float(), q_nodes.float()
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
            pi_p, pi_d = pi_ia.sum(dim=-1), pi_ia.sum(dim=1)

            s0 = self.s0_net(torch.cat([mean_p.float(), mean_d.float()], dim=-1)).squeeze(-1) if self.use_s0 else torch.zeros_like(lme)
            r_scalar = lme - s0
            z_p = torch.bmm(pi_p.unsqueeze(1), pf).squeeze(1)
            z_d = torch.bmm(pi_d.unsqueeze(1), qf).squeeze(1)
            r_vec = self.inter_mlp(torch.cat([z_p, z_d, z_p * z_d, r_scalar.unsqueeze(-1)], dim=-1)).squeeze(-1)
            return {"r": r_scalar + r_vec, "lse": lme, "s0": s0, "pi_p": pi_p, "s_ia": s_masked}
