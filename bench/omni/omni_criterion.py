"""Multi-view BCE and deduplicated Cross-View InfoNCE criterion for Omni3D-DTI."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from bench.omni.omni_config import OmniConfig


class OmniCriterion:
    """Combines primary BCE, auxiliary view supervision, and cross-view alignment."""

    def __init__(self, cfg: OmniConfig, align_tau: float = 0.15) -> None:
        """Store configuration weights and contrastive temperature."""
        self.cfg = cfg
        self.align_tau = align_tau

    def _cross_view_nce(
        self,
        view_a: Optional[torch.Tensor],
        view_b: Optional[torch.Tensor],
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE after deduplicating repeated entities in batch."""
        device = entity_ids.device
        if view_a is None or view_b is None:
            return torch.tensor(0.0, device=device)
        same = entity_ids.unsqueeze(1) == entity_ids.unsqueeze(0)
        keep = torch.tril(same).sum(dim=1) == 1
        if int(keep.sum()) < 2:
            return torch.tensor(0.0, device=device)

        za = F.normalize(view_a[keep].float(), dim=-1)
        zb = F.normalize(view_b[keep].float(), dim=-1)
        logits = torch.matmul(za, zb.transpose(0, 1)) / self.align_tau
        targets = torch.arange(za.size(0), device=device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))

    def compute(
        self,
        out: Dict[str, Optional[torch.Tensor]],
        labels: torch.Tensor,
        prot_ids: torch.Tensor,
        drug_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate total loss over main logit, auxiliary views, and cross-view NCE."""
        y = labels.float()
        main_logit = out["logit"]
        assert main_logit is not None
        total = F.binary_cross_entropy_with_logits(main_logit.float(), y)

        if self.cfg.w_aux > 0.0:
            if out["logit_seq2d"] is not None:
                total = total + (0.6 * self.cfg.w_aux) * F.binary_cross_entropy_with_logits(out["logit_seq2d"].float(), y)
            if out["logit_3d"] is not None:
                total = total + (0.4 * self.cfg.w_aux) * F.binary_cross_entropy_with_logits(out["logit_3d"].float(), y)

        if self.cfg.use_align and self.cfg.w_align > 0.0:
            l_p = self._cross_view_nce(out["hp_seq"], out["hp_3d"], prot_ids)
            l_d = self._cross_view_nce(out["hd_2d"], out["hd_3d"], drug_ids)
            total = total + self.cfg.w_align * (l_p + l_d)

        return total
