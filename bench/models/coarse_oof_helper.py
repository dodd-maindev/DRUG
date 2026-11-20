"""Out-of-fold C1 coarse logit generator for unbiased Delta-Interface training."""

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.coarse_model import CoarseInteractionNetwork, TemperatureCalibrator


class CoarseOofGenerator:
    """Fits 5-fold out-of-fold C1 logits on train and full C1 logits on val/test."""

    @staticmethod
    def _mean_tensors(df: pd.DataFrame, entities: Dict[str, Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Extract mean-pooled protein and drug vectors for a split dataframe."""
        p_vecs = torch.stack([entities["proteins"][p]["emb"].float().mean(dim=0) for p in df["protein"]]).to(device)
        d_vecs = torch.stack([entities["drugs"][d]["emb"].float().mean(dim=0) for d in df["drug"]]).to(device)
        labels = df["label"].values.astype(np.float32)
        return p_vecs, d_vecs, labels

    @staticmethod
    def _fit_c1(d_tr: torch.Tensor, p_tr: torch.Tensor, y_tr: np.ndarray, d_ev: torch.Tensor, p_ev: torch.Tensor, seed: int, epochs: int = 50) -> Tuple[np.ndarray, CoarseInteractionNetwork]:
        """Train one CoarseInteractionNetwork and return predictions on eval tensors."""
        torch.manual_seed(seed)
        device = d_tr.device
        net = CoarseInteractionNetwork().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(d_tr), device=device)
            for start in range(0, len(perm), 256):
                idx = perm[start: start + 256]
                loss = loss_fn(net(d_tr[idx], p_tr[idx]), y_t[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            ev_logits = net(d_ev, p_ev).cpu().numpy()
        return ev_logits, net

    @classmethod
    def compute_or_load(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        entities: Dict[str, Dict[str, Any]],
        seed: int,
        cache_path: str,
    ) -> Dict[str, Any]:
        """Return OOF b_train, full b_val/b_test, calibrated tau_b, and full C1 state."""
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        from sklearn.model_selection import StratifiedKFold

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p_tr, d_tr, y_tr = cls._mean_tensors(train_df, entities, device)
        p_va, d_va, y_va = cls._mean_tensors(val_df, entities, device)
        p_te, d_te, _ = cls._mean_tensors(test_df, entities, device)

        b_train_oof = np.zeros(len(train_df), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, oof_idx) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr)):
            tr_i = torch.tensor(tr_idx, device=device)
            oof_i = torch.tensor(oof_idx, device=device)
            fold_logits, _ = cls._fit_c1(d_tr[tr_i], p_tr[tr_i], y_tr[tr_idx], d_tr[oof_i], p_tr[oof_i], seed * 10 + fold, epochs=45)
            b_train_oof[oof_idx] = fold_logits

        b_val, full_net = cls._fit_c1(d_tr, p_tr, y_tr, d_va, p_va, seed, epochs=60)
        with torch.no_grad():
            b_test = full_net(d_te, p_te).cpu().numpy()
        tau_b = TemperatureCalibrator.fit_temperature(torch.tensor(b_val), torch.tensor(y_va))

        payload = {
            "b_train": b_train_oof, "b_val": b_val.astype(np.float32), "b_test": b_test.astype(np.float32),
            "tau_b": float(tau_b), "state_dict": {k: v.cpu() for k, v in full_net.state_dict().items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(payload, cache_path)
        print(f"[coarse-oof] seed={seed} tau_b={tau_b:.3f} saved -> {cache_path}", flush=True)
        return payload
