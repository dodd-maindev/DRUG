"""Model factory, EMA shadow update, and batched inference helpers."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from bench.models.dice_baselines import Configurable3DiceModel
from bench.models.interface_head import DeltaInterfaceModel
from bench.models.lego_coarse_3d import Lego3DModel
from bench.training.train_config import ArmConfig


class TrainerHelpers:
    """Constructs models, moves collated batches, updates EMA, and runs inference."""

    @staticmethod
    def build_model(cfg: ArmConfig, coarse_payload: Optional[Dict[str, Any]], device: torch.device) -> torch.nn.Module:
        """Instantiate the neural network corresponding to ArmConfig.family."""
        if cfg.family == "3dice":
            model = Configurable3DiceModel(cfg.mask_aware_conv, cfg.prune_dead_fusion)
        elif cfg.family == "lego3d":
            model = Lego3DModel()
        else:
            tau_b = coarse_payload["tau_b"] if coarse_payload else 1.0
            state = coarse_payload["state_dict"] if coarse_payload else None
            model = DeltaInterfaceModel(tau_b, cfg.stop_grad_b, cfg.use_s0, cfg.use_size_interaction, state)
        return model.to(device)

    @staticmethod
    def update_ema(ema_model: torch.nn.Module, live_model: torch.nn.Module, decay: float = 0.999) -> None:
        """Update exponential moving average shadow weights."""
        with torch.no_grad():
            for ema_p, live_p in zip(ema_model.parameters(), live_model.parameters()):
                ema_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)

    @staticmethod
    def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Transfer tensor values in a collated batch to the target device."""
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    @classmethod
    def predict_loader(cls, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
        """Run evaluation over a DataLoader returning logits, probs, and residual r."""
        model.eval()
        logits_all, r_all, b_all, labels_all = [], [], [], []
        with torch.no_grad():
            for raw_batch in loader:
                batch = cls.move_batch(raw_batch, device)
                out = model(batch, return_maps=False)
                logits_all.append(out["logit"].cpu().numpy())
                r_all.append(out.get("r", out["logit"]).cpu().numpy())
                b_all.append(out.get("b", torch.zeros_like(out["logit"])).cpu().numpy())
                labels_all.append(raw_batch["label"].numpy())
        logits = np.concatenate(logits_all)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return {
            "logits": logits, "probs": probs, "labels": np.concatenate(labels_all),
            "r": np.concatenate(r_all), "b": np.concatenate(b_all),
        }
