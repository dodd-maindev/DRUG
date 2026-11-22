"""Multi-seed trainer and Table 1 evaluator for multi-modal signal probes."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diagnostics.metrics import MetricEvaluator
from research.probe.probe_net import ModalityProbeNet


class ProbeEvaluator:
    """Trains and evaluates multi-seed signal probes returning all 7 Table 1 metrics."""

    @staticmethod
    def _standardize(tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> List[torch.Tensor]:
        """Standardize train/val/test matrices strictly using train statistics with soft std floor."""
        if tr.shape[1] == 0:
            return [torch.zeros((len(x), 0), dtype=torch.float32) for x in (tr, va, te)]
        mu = tr.mean(axis=0, keepdims=True)
        sd = np.maximum(tr.std(axis=0, keepdims=True), 0.25)
        return [torch.from_numpy(np.clip((x - mu) / sd, -10.0, 10.0).astype(np.float32)) for x in (tr, va, te)]

    @classmethod
    def _run_seed(
        cls, p_t: List[torch.Tensor], d_t: List[torch.Tensor], x_t: List[torch.Tensor],
        y_t: List[np.ndarray], seed: int, epochs: int = 35, batch_size: int = 256,
    ) -> Dict[str, float]:
        """Train one probe seed on GPU/CPU and return Paper Table 1 test metrics."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ModalityProbeNet(p_t[0].shape[1], d_t[0].shape[1], x_t[0].shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        tr_ds = TensorDataset(p_t[0], d_t[0], x_t[0], torch.from_numpy(y_t[0].astype(np.float32)))
        loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)), eta_min=1e-5)
        bce = nn.BCEWithLogitsLoss()

        va_p, va_d, va_x = p_t[1].to(device), d_t[1].to(device), x_t[1].to(device)
        te_p, te_d, te_x = p_t[2].to(device), d_t[2].to(device), x_t[2].to(device)
        best_auc, best_state = -1.0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for _ in range(epochs):
            model.train()
            for bp, bd, bx, by in loader:
                bp, bd, bx, by = bp.to(device), bd.to(device), bx.to(device), by.to(device)
                opt.zero_grad()
                bce(model(bp, bd, bx), by).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                va_prob = torch.sigmoid(model(va_p, va_d, va_x)).cpu().numpy()
            va_auc = float(MetricEvaluator.compute_classification_metrics(y_t[1], va_prob, 0.5)["auroc"])
            if va_auc > best_auc:
                best_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(va_p, va_d, va_x)).cpu().numpy()
            te_prob = torch.sigmoid(model(te_p, te_d, te_x)).cpu().numpy()
        thr = MetricEvaluator.find_best_mcc_threshold(y_t[1], va_prob)
        return MetricEvaluator.compute_classification_metrics(y_t[2], te_prob, thr)

    @classmethod
    def evaluate_arm(
        cls, p_arrs: List[np.ndarray], d_arrs: List[np.ndarray], pair_arrs: List[np.ndarray],
        y_arrs: List[np.ndarray], seeds: List[int],
    ) -> Dict[str, float]:
        """Run multiple seeds and average all 7 Paper Table 1 classification metrics."""
        p_t = cls._standardize(*p_arrs)
        d_t = cls._standardize(*d_arrs)
        x_t = cls._standardize(*pair_arrs)
        runs = [cls._run_seed(p_t, d_t, x_t, y_arrs, s) for s in seeds]
        return {k: float(np.mean([r[k] for r in runs])) for k in ("accuracy", "ppv", "tpr", "f1", "mcc", "auroc", "auprc")}
