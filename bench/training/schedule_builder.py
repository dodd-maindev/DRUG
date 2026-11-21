"""Learning-rate scheduler builder supporting linear warmup and cosine decay."""

from __future__ import annotations

import math
from typing import Optional

import torch

from bench.training.train_config import ArmConfig


class LrScheduleBuilder:
    """Constructs per-step cosine annealing schedulers with optional linear warmup."""

    @staticmethod
    def build(
        cfg: ArmConfig,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build step-level scheduler when cosine decay is enabled, else None."""
        if not cfg.use_cosine:
            return None
        total = max(1, cfg.epochs * steps_per_epoch)
        warmup = int(round(cfg.warmup_frac * total))
        if warmup <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=1e-6)
        min_ratio = min(1.0, 1e-6 / max(cfg.learning_rate, 1e-8))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = min(1.0, float(step - warmup) / float(max(1, total - warmup)))
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
