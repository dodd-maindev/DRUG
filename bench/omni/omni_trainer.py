"""Single-seed trainer with linear warmup, cosine decay, and ramped EMA for Omni3D-DTI."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bench.omni.omni_config import OmniConfig
from bench.omni.omni_criterion import OmniCriterion
from bench.omni.omni_model import Omni3dDtiModel
from bench.omni.omni_schedule import OmniLrSchedule
from diagnostics.metrics import MetricEvaluator


class OmniTrainer:
    """Fits one seed of Omni3D-DTI and evaluates all 7 Paper Table 1 metrics."""

    @staticmethod
    def _update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float, step: int) -> None:
        """Update exponential moving average shadow parameters with early warmup ramp."""
        eff = min(decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for ep, lp in zip(ema.parameters(), model.parameters()):
                ep.data.mul_(eff).add_(lp.data, alpha=1.0 - eff)

    @staticmethod
    def _predict_split(
        model: torch.nn.Module, split: Dict[str, Any], device: torch.device, batch_size: int = 1024
    ) -> Tuple[np.ndarray, float, float]:
        """Run batched evaluation returning probabilities and mean gate weights."""
        model.eval()
        probs, gp, gd = [], [], []
        n = split["y"].shape[0]
        with torch.no_grad():
            for st in range(0, n, batch_size):
                ed = min(n, st + batch_size)
                out = model(
                    split["p_seq"][st:ed].to(device), split["p_3d"][st:ed].to(device),
                    split["d_2d"][st:ed].to(device), split["d_3d"][st:ed].to(device),
                )
                probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
                gp.append(out["gate_p"].cpu().numpy())
                gd.append(out["gate_d"].cpu().numpy())
        return np.concatenate(probs), float(np.mean(np.concatenate(gp))), float(np.mean(np.concatenate(gd)))

    @classmethod
    def train_seed(
        cls, cfg: OmniConfig, packed: Dict[str, Dict[str, Any]], dims: Dict[str, int], seed: int
    ) -> Tuple[Dict[str, float], Dict[str, Any], dict]:
        """Train one random seed selecting best EMA checkpoint by val AUROC."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Omni3dDtiModel(cfg, dims).to(device)
        ema = copy.deepcopy(model).eval()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        tr = packed["train"]
        tr_ds = TensorDataset(tr["p_seq"], tr["p_3d"], tr["d_2d"], tr["d_3d"], tr["p_idx"], tr["d_idx"], tr["y"])
        loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        sched = OmniLrSchedule.build(cfg, opt, len(loader))
        criterion = OmniCriterion(cfg)

        va_y, te_y = packed["val"]["y"].numpy(), packed["test"]["y"].numpy()
        best_auc, best_state, g_step = -1.0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0

        for _ in range(cfg.epochs):
            model.train()
            for ps, p3, d2, d3, pid, did, y in loader:
                opt.zero_grad()
                out = model(ps.to(device), p3.to(device), d2.to(device), d3.to(device))
                criterion.compute(out, y.to(device), pid.to(device), did.to(device)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                cls._update_ema(ema, model, cfg.ema_decay, g_step)
                g_step += 1

            va_prob, _, _ = cls._predict_split(ema, packed["val"], device)
            va_auc = float(MetricEvaluator.compute_classification_metrics(va_y, va_prob, 0.5)["auroc"])
            if va_auc > best_auc:
                best_auc, best_state = va_auc, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        va_prob, _, _ = cls._predict_split(ema, packed["val"], device)
        te_prob, gp, gd = cls._predict_split(ema, packed["test"], device)
        thr = MetricEvaluator.find_best_mcc_threshold(va_y, va_prob)
        metrics = {**MetricEvaluator.compute_classification_metrics(te_y, te_prob, thr), "gate_p": gp, "gate_d": gd}
        preds = {"y": te_y, "prob": te_prob, "threshold": np.array(thr, dtype=np.float32), "gate_p": np.array(gp), "gate_d": np.array(gd)}
        return metrics, preds, best_state
