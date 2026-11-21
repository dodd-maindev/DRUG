"""Vectorized dual-supervision loss criterion (main CE, iface CE, InfoNCE, CF, compact)."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from bench.losses.counterfactual_ops import CounterfactualRegularizer


class DeltaInterfaceCriterion:
    """Supervises combined logit and un-starved 3D pocket residual jointly."""

    def __init__(self, w_cond: float = 0.20, w_cf: float = 0.08, w_compact: float = 0.02, label_smoothing: float = 0.0, tau_cond: float = 1.0) -> None:
        """Store loss weights, smoothing factor, and conditional temperature."""
        self.w_cond = w_cond
        self.w_cf = w_cf
        self.w_compact = w_compact
        self.label_smoothing = label_smoothing
        self.tau_cond = tau_cond

    def _within_entity_infonce(self, logits: torch.Tensor, entity_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Vectorized within-entity InfoNCE without Python loops or CPU-GPU sync."""
        bsz = logits.size(0)
        same = entity_ids.unsqueeze(1) == entity_ids.unsqueeze(0)
        neg_mask = same & (labels == 0).unsqueeze(0)
        has_group = (labels == 1) & neg_mask.any(dim=1)
        if not has_group.any():
            return torch.tensor(0.0, device=logits.device)

        eye = torch.eye(bsz, device=logits.device, dtype=torch.bool)
        scaled = (logits.float() / self.tau_cond).unsqueeze(0).expand(bsz, bsz)
        masked = scaled.masked_fill(~(neg_mask | eye), -1e4)
        losses = torch.logsumexp(masked, dim=1) - (logits.float() / self.tau_cond)
        return losses[has_group].mean()

    def compute(
        self,
        model: Any,
        batch: Dict[str, Any],
        out: Dict[str, Any],
        is_delta_model: bool,
        use_2c_ce: bool = False,
        is_geocross: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate total loss and detached scalar component diagnostics."""
        labels = batch["label"]
        targets = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        if use_2c_ce and "logits_2c" in out:
            l_ce = F.cross_entropy(out["logits_2c"].float(), labels, label_smoothing=self.label_smoothing)
        else:
            l_ce = F.binary_cross_entropy_with_logits(out["logit"].float(), targets)

        total = l_ce
        metrics = {"l_ce": float(l_ce.detach())}
        if not (is_delta_model or is_geocross):
            return total, metrics

        l_iface = F.binary_cross_entropy_with_logits(out["r"].float(), targets)
        total = total + (1.0 if is_geocross else 0.5) * l_iface
        metrics["l_if"] = float(l_iface.detach())

        if self.w_cond > 0:
            lp_r = self._within_entity_infonce(out["r"], batch["protein_idx"], labels)
            ld_r = self._within_entity_infonce(out["r"], batch["drug_idx"], labels)
            if is_geocross:
                l_cond = 0.5 * (lp_r + ld_r)
            else:
                lp_m = self._within_entity_infonce(out["logit"], batch["protein_idx"], labels)
                ld_m = self._within_entity_infonce(out["logit"], batch["drug_idx"], labels)
                l_cond = 0.5 * (lp_m + lp_r) + 0.5 * ld_m
            total = total + self.w_cond * l_cond
            metrics["l_cond"] = float(l_cond.detach())

        if self.w_cf > 0:
            l_nec, l_spec = CounterfactualRegularizer.counterfactual_losses(model, batch, out, max_pairs=4 if is_geocross else 12)
            total = total + self.w_cf * (l_nec + l_spec)
            metrics["l_nec"] = float(l_nec.detach())
            metrics["l_spec"] = float(l_spec.detach())

        if self.w_compact > 0:
            l_comp = CounterfactualRegularizer.compactness_loss(batch, out)
            total = total + self.w_compact * l_comp
            metrics["l_comp"] = float(l_comp.detach())

        return total, metrics
