"""Single-seed training loop with optional EMA, cosine schedule, and checkpointing."""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from bench.data.entity_batch_sampler import EntityGroupedBatchSampler
from bench.data.pair_dataset import BenchmarkBatchCollator, BenchmarkPairDataset
from bench.losses.delta_criterion import DeltaInterfaceCriterion
from bench.training.model_factory import TrainerHelpers
from bench.training.train_config import ArmConfig
from diagnostics.metrics import MetricEvaluator


class ArmTrainer:
    """Trains one seed of an experimental arm and returns calibrated predictions."""

    @staticmethod
    def train_seed(
        cfg: ArmConfig, train_ds: BenchmarkPairDataset, val_ds: BenchmarkPairDataset,
        test_ds: BenchmarkPairDataset, coarse_payload: Optional[Dict[str, Any]], seed: int,
    ) -> Tuple[Dict[str, Any], dict]:
        """Fit one random seed and evaluate best validation checkpoint on test."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collate = BenchmarkBatchCollator.collate

        if cfg.use_entity_sampler:
            sampler = EntityGroupedBatchSampler(train_ds.protein_indices, train_ds.drug_indices, train_ds.df["label"].values, cfg.batch_size, seed)
            tr_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, pin_memory=(device.type == "cuda"))
        else:
            tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=collate, pin_memory=(device.type == "cuda"))
        va_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

        model = TrainerHelpers.build_model(cfg, coarse_payload, device)
        ema_model = copy.deepcopy(model).eval() if cfg.use_ema else None
        params = [p for p in model.parameters() if p.requires_grad]
        opt_cls = torch.optim.AdamW if cfg.optimizer_type == "adamw" else torch.optim.Adam
        opt = opt_cls(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = max(1, cfg.epochs * len(tr_loader))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6) if cfg.use_cosine else None
        criterion = DeltaInterfaceCriterion(cfg.w_cond, cfg.w_cf, cfg.w_compact, cfg.label_smoothing)
        is_delta = cfg.family == "delta"

        best_score, best_state = -1.0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for raw_batch in tr_loader:
                batch = TrainerHelpers.move_batch(raw_batch, device)
                out = model(batch, return_maps=is_delta, use_batch_oof_b=is_delta) if is_delta else model(batch)
                loss, _ = criterion.compute(model, batch, out, is_delta, use_2c_ce=(cfg.family == "3dice"))
                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                if ema_model is not None:
                    TrainerHelpers.update_ema(ema_model, model)

            eval_net = ema_model if ema_model is not None else model
            va_out = TrainerHelpers.predict_loader(eval_net, va_loader, device)
            va_score = MetricEvaluator.compute_classification_metrics(va_out["labels"], va_out["probs"], 0.5)[cfg.select_metric]
            if va_score > best_score:
                best_score = va_score
                best_state = {k: v.detach().cpu().clone() for k, v in eval_net.state_dict().items()}
            if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                print(f"  [{cfg.name} s={seed}] ep={epoch:02d}/{cfg.epochs} val_{cfg.select_metric}={va_score:.4f}", flush=True)

        eval_net = ema_model if ema_model is not None else model
        eval_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        va_out = TrainerHelpers.predict_loader(eval_net, va_loader, device)
        te_out = TrainerHelpers.predict_loader(eval_net, te_loader, device)
        thr = 0.5 if cfg.family == "3dice" and cfg.select_metric == "accuracy" else MetricEvaluator.find_best_mcc_threshold(va_out["labels"], va_out["probs"])
        metrics = MetricEvaluator.compute_classification_metrics(te_out["labels"], te_out["probs"], thr)
        return {"metrics": metrics, "threshold": float(thr), "val": va_out, "test": te_out}, best_state
