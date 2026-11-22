"""Step-level linear warmup and cosine decay scheduler for Omni3D-DTI."""

from __future__ import annotations

import math

import torch

from bench.omni.omni_config import OmniConfig


class OmniLrSchedule:
    """Constructs warmup + cosine annealing learning rate schedule."""

    @staticmethod
    def build(cfg: OmniConfig, opt: torch.optim.Optimizer, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Create step-level linear warmup followed by cosine decay down to 1e-5."""
        total = max(1, cfg.epochs * steps_per_epoch)
        warmup = int(round(cfg.warmup_frac * total))
        min_ratio = min(1.0, 1e-5 / max(cfg.learning_rate, 1e-8))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = min(1.0, float(step - warmup) / float(max(1, total - warmup)))
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
