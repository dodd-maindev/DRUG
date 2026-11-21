"""Single-seed training loop with length bucketing, bfloat16 AMP, and EMA."""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from bench.data.entity_batch_sampler import EntityGroupedBatchSampler
from bench.data.pair_dataset import BenchmarkBatchCollator, BenchmarkPairDataset
from bench.losses.delta_criterion import DeltaInterfaceCriterion
from bench.training.model_factory import TrainerHelpers
from bench.training.schedule_builder import LrScheduleBuilder
from bench.training.train_config import ArmConfig
from diagnostics.metrics import MetricEvaluator


class ArmTrainer:
    """Trains one seed of an experimental arm and returns calibrated predictions."""

    @staticmethod
    def _build_train_loader(cfg: ArmConfig, train_ds: BenchmarkPairDataset, seed: int, device: torch.device) -> DataLoader:
        """Construct entity-partitioned or standard i.i.d. shuffled training DataLoader."""
        pin = device.type == "cuda"
        if cfg.use_entity_sampler:
            sampler = EntityGroupedBatchSampler(train_ds.protein_indices, train_ds.drug_indices, train_ds.df["label"].values, cfg.batch_size, seed, train_ds.protein_lengths)
            return DataLoader(train_ds, batch_sampler=sampler, collate_fn=BenchmarkBatchCollator.collate, pin_memory=pin)
        return DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=BenchmarkBatchCollator.collate, pin_memory=pin)

    @classmethod
    def train_seed(
        cls, cfg: ArmConfig, train_ds: BenchmarkPairDataset, val_ds: BenchmarkPairDataset,
        test_ds: BenchmarkPairDataset, coarse_payload: Optional[Dict[str, Any]], seed: int,
    ) -> Tuple[Dict[str, Any], dict]:
        """Fit one random seed using bfloat16 AMP on CUDA and select best checkpoint."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        tr_loader = cls._build_train_loader(cfg, train_ds, seed, device)
        va_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=BenchmarkBatchCollator.collate)
        te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=BenchmarkBatchCollator.collate)

        model = TrainerHelpers.build_model(cfg, coarse_payload, device)
        ema_model = copy.deepcopy(model).eval() if cfg.use_ema else None
        params = [p for p in model.parameters() if p.requires_grad]
        opt_cls = torch.optim.AdamW if cfg.optimizer_type == "adamw" else torch.optim.Adam
        opt = opt_cls(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        sched = LrScheduleBuilder.build(cfg, opt, len(tr_loader))
        criterion = DeltaInterfaceCriterion(cfg.w_cond, cfg.w_cf, cfg.w_compact, cfg.label_smoothing)
        is_delta, is_geo = cfg.family == "delta", cfg.family == "geocross3d"
        has_residual = is_delta or is_geo

        best_score, best_state = -1.0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        global_step = 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            loss_sum: Dict[str, float] = {}
            steps = 0
            for raw_batch in tr_loader:
                batch = TrainerHelpers.move_batch(raw_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(batch, return_maps=has_residual, use_batch_oof_b=is_delta) if has_residual else model(batch)
                    loss, lm = criterion.compute(model, batch, out, is_delta, use_2c_ce=(cfg.family == "3dice"), is_geocross=is_geo)
                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                if ema_model is not None:
                    TrainerHelpers.update_ema(ema_model, model, step=global_step)
                global_step += 1
                for k, v in lm.items():
                    loss_sum[k] = loss_sum.get(k, 0.0) + v
                steps += 1

            eval_net = ema_model if ema_model is not None else model
            va_out = TrainerHelpers.predict_loader(eval_net, va_loader, device)
            va_score = MetricEvaluator.compute_classification_metrics(va_out["labels"], va_out["probs"], 0.5)[cfg.select_metric]
            if va_score > best_score:
                best_score = va_score
                best_state = {k: v.detach().cpu().clone() for k, v in eval_net.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
                TrainerHelpers.log_epoch(cfg.name, seed, epoch, cfg.epochs, cfg.select_metric, va_score, va_out, loss_sum, steps, has_residual)

        eval_net = ema_model if ema_model is not None else model
        eval_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        va_out = TrainerHelpers.predict_loader(eval_net, va_loader, device)
        te_out = TrainerHelpers.predict_loader(eval_net, te_loader, device)
        thr = 0.5 if cfg.family == "3dice" and cfg.select_metric == "accuracy" else MetricEvaluator.find_best_mcc_threshold(va_out["labels"], va_out["probs"])
        metrics = MetricEvaluator.compute_classification_metrics(te_out["labels"], te_out["probs"], thr)
        return {"metrics": metrics, "threshold": float(thr), "val": va_out, "test": te_out}, best_state
