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
    def _filter_tensors(df: pd.DataFrame, entities: Dict[str, Dict[str, Any]], device: torch.device) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
        """Filter pairs with cached entities and extract mean-pooled split tensors."""
        proteins, drugs = entities["proteins"], entities["drugs"]
        keep = np.array([p in proteins and d in drugs for p, d in zip(df["protein"], df["drug"])], dtype=bool)
        sub = df[keep].reset_index(drop=True)
        p_vecs = torch.stack([proteins[p]["emb"].float().mean(dim=0) for p in sub["protein"]]).to(device)
        d_vecs = torch.stack([drugs[d]["emb"].float().mean(dim=0) for d in sub["drug"]]).to(device)
        return keep, p_vecs, d_vecs, sub["label"].values.astype(np.float32)

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
        cls, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
        entities: Dict[str, Dict[str, Any]], seed: int, cache_path: str,
    ) -> Dict[str, Any]:
        """Return aligned OOF b_train, full b_val/b_test, tau_b, and full C1 state."""
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        from sklearn.model_selection import StratifiedKFold

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k_tr, p_tr, d_tr, y_tr = cls._filter_tensors(train_df, entities, device)
        k_va, p_va, d_va, y_va = cls._filter_tensors(val_df, entities, device)
        k_te, p_te, d_te, _ = cls._filter_tensors(test_df, entities, device)

        oof_sub = np.zeros(len(y_tr), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, oof_idx) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr)):
            tr_i, oof_i = torch.tensor(tr_idx, device=device), torch.tensor(oof_idx, device=device)
            oof_sub[oof_idx], _ = cls._fit_c1(d_tr[tr_i], p_tr[tr_i], y_tr[tr_idx], d_tr[oof_i], p_tr[oof_i], seed * 10 + fold, epochs=45)

        b_va_sub, full_net = cls._fit_c1(d_tr, p_tr, y_tr, d_va, p_va, seed, epochs=60)
        with torch.no_grad():
            b_te_sub = full_net(d_te, p_te).cpu().numpy()
        tau_b = TemperatureCalibrator.fit_temperature(torch.tensor(b_va_sub), torch.tensor(y_va))

        b_tr = np.zeros(len(train_df), dtype=np.float32)
        b_va = np.zeros(len(val_df), dtype=np.float32)
        b_te = np.zeros(len(test_df), dtype=np.float32)
        b_tr[k_tr], b_va[k_va], b_te[k_te] = oof_sub, b_va_sub, b_te_sub

        payload = {
            "b_train": b_tr, "b_val": b_va, "b_test": b_te,
            "tau_b": float(tau_b), "state_dict": {k: v.cpu() for k, v in full_net.state_dict().items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(payload, cache_path)
        print(f"[coarse-oof] seed={seed} tau_b={tau_b:.3f} saved -> {cache_path}", flush=True)
        return payload
