"""Model factory, EMA shadow update, checkpoint persistence, and batched inference."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from bench.models.delta_interface_model import DeltaInterfaceModel
from bench.models.dice_baselines import Configurable3DiceModel
from bench.models.lego_coarse_3d import Lego3DModel
from bench.training.train_config import ArmConfig
from diagnostics.metrics import MetricEvaluator


class TrainerHelpers:
    """Constructs models, manages AMP inference, and persists per-seed results."""

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
    def update_ema(ema_model: torch.nn.Module, live_model: torch.nn.Module, decay: float = 0.99) -> None:
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
        """Run evaluation over a DataLoader using bfloat16 AMP when available."""
        model.eval()
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        logits_all, r_all, b_all, labels_all = [], [], [], []
        with torch.no_grad():
            for raw_batch in loader:
                batch = cls.move_batch(raw_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(batch, return_maps=False)
                logits_all.append(out["logit"].float().cpu().numpy())
                r_all.append(out.get("r", out["logit"]).float().cpu().numpy())
                b_all.append(out.get("b", torch.zeros_like(out["logit"])).float().cpu().numpy())
                labels_all.append(raw_batch["label"].numpy())
        logits = np.concatenate(logits_all)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return {"logits": logits, "probs": probs, "labels": np.concatenate(labels_all), "r": np.concatenate(r_all), "b": np.concatenate(b_all)}

    @staticmethod
    def r_only_auroc(labels: np.ndarray, residual: np.ndarray) -> float:
        """Compute binary AUROC from residual logits r alone."""
        r_prob = 1.0 / (1.0 + np.exp(-np.clip(residual, -60.0, 60.0)))
        return float(roc_auc_score(labels, r_prob))

    @classmethod
    def load_resumed_seed(cls, pred_path: str, is_delta: bool) -> Tuple[Dict[str, float], Optional[float]]:
        """Reload classification metrics and optional residual AUROC from saved .npz."""
        saved = np.load(pred_path, allow_pickle=True)
        metrics = MetricEvaluator.compute_classification_metrics(saved["label"], saved["prob"], float(saved["threshold"]))
        r_auc = cls.r_only_auroc(saved["label"], saved["r"]) if is_delta else None
        return metrics, r_auc

    @classmethod
    def save_seed_artifacts(cls, ckpt_path: str, pred_path: str, state: dict, test_df: pd.DataFrame, result: Dict[str, Any], is_delta: bool) -> Optional[float]:
        """Save model checkpoint and test prediction arrays, returning optional r AUROC."""
        torch.save(state, ckpt_path)
        te = result["test"]
        np.savez(
            pred_path, protein=test_df["protein"].values, drug=test_df["drug"].values,
            label=te["labels"], prob=te["probs"], logit=te["logits"],
            r=te["r"], b=te["b"], threshold=result["threshold"],
        )
        return cls.r_only_auroc(te["labels"], te["r"]) if is_delta else None

    @classmethod
    def log_epoch(cls, name: str, seed: int, epoch: int, epochs: int, metric: str, va_score: float, va_out: Dict[str, np.ndarray], loss_sum: Dict[str, float], steps: int, is_delta: bool) -> None:
        """Print concise per-epoch validation metric and averaged loss diagnostics."""
        avg_l = " ".join(f"{k}={v / max(steps, 1):.3f}" for k, v in loss_sum.items())
        r_msg = f" val_r_auc={cls.r_only_auroc(va_out['labels'], va_out['r']):.4f}" if is_delta else ""
        print(f"  [{name} s={seed}] ep={epoch:02d}/{epochs} val_{metric}={va_score:.4f}{r_msg} ({avg_l})", flush=True)
