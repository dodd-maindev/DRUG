"""GPU computational complexity profiler for Omni3D-DTI vs 3DICE at batch size 32."""

from __future__ import annotations

import time
from typing import Any, Dict


class ComplexityProfiler:
    """Measures trainable parameters, peak memory, forward latency, and epoch time."""

    @staticmethod
    def exact_param_count_m(dims: Dict[str, int], use_gate: bool = True, latent_dim: int = 384) -> float:
        """Compute exact trainable parameter count (millions) for 4-view Omni3dDtiModel."""
        d = latent_dim
        towers = sum((2 * dims[k]) + (dims[k] * d + d) + (2 * d) + (d * d + d) for k in ("p_seq", "p_3d", "d_2d", "d_3d"))
        gates = 2 * ((4 * d * (d // 2) + (d // 2)) + ((d // 2) * d + d)) if use_gate else 0
        clf = (2 * 6 * d) + (6 * d * 512 + 512) + (512 * 128 + 128) + 129
        return float(towers + gates + clf + 2 * ((2 * d) + (d + 1))) / 1e6

    @classmethod
    def _profile_loop(cls, name: str, model, opt, loader, move_fn, fwd_fn, loss_fn, device, batch_size: int) -> Dict[str, Any]:
        """Measure params, peak train VRAM (MB), per-sample forward (ms), and 1-epoch time (s)."""
        import torch
        sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None  # noqa: E731
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        sample = move_fn(next(iter(loader)), device)
        model.eval()
        with torch.no_grad():
            for i in range(60):
                if i == 10:
                    sync()
                    t0 = time.perf_counter()
                fwd_fn(model, sample)
            sync()
        fwd_ms = (time.perf_counter() - t0) * 1000.0 / float(50 * batch_size)

        model.train()
        for step_idx, raw in enumerate(loader):
            b = move_fn(raw, device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model, fwd_fn(model, b), b).backward()
            opt.step()
            if step_idx >= 2:
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        t_ep = time.perf_counter()
        for raw in loader:
            b = move_fn(raw, device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model, fwd_fn(model, b), b).backward()
            opt.step()
        sync()
        epoch_s = time.perf_counter() - t_ep
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        return {"model": name, "params_m": params_m, "peak_mem_mb": peak_mb, "forward_ms": fwd_ms, "epoch_s": epoch_s, "gpu": gpu}

    @classmethod
    def profile_omni(cls, cfg, packed: Dict[str, Any], dims: Dict[str, int], device, batch_size: int = 32) -> Dict[str, Any]:
        """Profile an Omni3D-DTI variant on GPU using standardized train tensors."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from bench.omni.omni_criterion import OmniCriterion
        from bench.omni.omni_model import Omni3dDtiModel

        tr = packed["train"]
        ds = TensorDataset(tr["p_seq"], tr["p_3d"], tr["d_2d"], tr["d_3d"], tr["p_idx"], tr["d_idx"], tr["y"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
        model = Omni3dDtiModel(cfg, dims).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        crit = OmniCriterion(cfg)
        return cls._profile_loop(
            cfg.name, model, opt, loader, lambda r, d: tuple(t.to(d, non_blocking=True) for t in r),
            lambda m, b: m(b[0], b[1], b[2], b[3]), lambda m, o, b: crit.compute(o, b[6], b[4], b[5]), device, batch_size,
        )

    @classmethod
    def profile_3dice(cls, train_ds, device, batch_size: int = 32) -> Dict[str, Any]:
        """Profile baseline 3DICE on the same GPU and batch size for controlled comparison."""
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from bench.data.pair_dataset import BenchmarkBatchCollator
        from bench.models.dice_baselines import Configurable3DiceModel

        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=BenchmarkBatchCollator.collate, pin_memory=(device.type == "cuda"))
        model = Configurable3DiceModel(mask_aware_conv=False, prune_dead_fusion=False).to(device)
        return cls._profile_loop(
            "3DICE", model, torch.optim.Adam(model.parameters(), lr=5e-5), loader,
            lambda raw, d: {k: v.to(d, non_blocking=True) if torch.is_tensor(v) else v for k, v in raw.items()},
            lambda m, b: m(b, return_maps=False), lambda m, o, b: F.cross_entropy(o["logits_2c"], b["label"]), device, batch_size,
        )
