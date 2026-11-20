"""H-A2 evaluation: positive-vs-negative JS divergence, spread, and padding leak."""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from bench.data.pair_dataset import BenchmarkBatchCollator, BenchmarkPairDataset
from bench.training.model_factory import TrainerHelpers
from bench.training.train_config import ArmConfig


class AttentionJsEvaluator:
    """Measures residue-attention divergence between positives/negatives and pad leak."""

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Compute base-2 Jensen-Shannon divergence in [0, 1] between two distributions."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = p / max(p.sum(), eps)
        q = q / max(q.sum(), eps)
        m = 0.5 * (p + q)
        kl_pm = np.sum(np.where(p > eps, p * np.log2((p + eps) / (m + eps)), 0.0))
        kl_qm = np.sum(np.where(q > eps, q * np.log2((q + eps) / (m + eps)), 0.0))
        return float(0.5 * (kl_pm + kl_qm))

    @classmethod
    def evaluate(
        cls, cfg: ArmConfig, state_dict: dict, dataset: BenchmarkPairDataset, coarse_state: dict | None = None
    ) -> Dict[str, float]:
        """Compute mean pos/neg JS divergence, normalized entropy, spatial spread, and leak."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tau_b = coarse_state["tau_b"] if coarse_state else 1.0
        b_sd = coarse_state["state_dict"] if coarse_state else None
        model = TrainerHelpers.build_model(cfg, {"tau_b": tau_b, "state_dict": b_sd} if coarse_state else None, device)
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()

        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=BenchmarkBatchCollator.collate)
        pos_maps: Dict[str, List[np.ndarray]] = defaultdict(list)
        neg_maps: Dict[str, List[np.ndarray]] = defaultdict(list)
        entropies, spreads = [], []

        with torch.no_grad():
            for raw in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
                out = model(batch, return_maps=True)
                attn = out["prot_attn"].cpu().numpy()
                p_mask = raw["protein_mask"].numpy()
                ca = raw["ca_coords"].numpy()
                valid_ca = raw["valid_ca"].numpy()
                for i in range(len(raw["label"])):
                    valid = ~p_mask[i]
                    vec = attn[i, valid]
                    vec = vec / max(vec.sum(), 1e-10)
                    pid, y = raw["uniprot_id"][i], int(raw["label"][i])
                    (pos_maps if y == 1 else neg_maps)[pid].append(vec)
                    if y == 1:
                        ent = -np.sum(np.where(vec > 1e-10, vec * np.log(vec + 1e-10), 0.0))
                        entropies.append(float(ent / max(np.log(max(len(vec), 2)), 1e-6)))
                        ca_v = valid & valid_ca[i]
                        if ca_v.sum() >= 3:
                            w = attn[i, ca_v]
                            w = w / max(w.sum(), 1e-10)
                            pts = ca[i, ca_v]
                            mu = (w[:, None] * pts).sum(axis=0, keepdims=True)
                            spreads.append(float(np.sqrt((w * ((pts - mu) ** 2).sum(axis=-1)).sum())))

        js_vals = [cls._js_divergence(np.mean(pos_maps[p], axis=0), np.mean(neg_maps[p], axis=0))
                   for p in sorted(set(pos_maps) & set(neg_maps))]
        leak_max = cls._measure_pad_leak(model, dataset, device)
        return {
            "js_pos_neg_mean": float(np.mean(js_vals)) if js_vals else 0.0,
            "n_js_proteins": len(js_vals),
            "pos_norm_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "pos_spatial_spread_A": float(np.mean(spreads)) if spreads else 0.0,
            "pad_leak_max_abs": leak_max,
        }

    @staticmethod
    def _measure_pad_leak(model: torch.nn.Module, dataset: BenchmarkPairDataset, device: torch.device, n: int = 64) -> float:
        """Compare B=1 vs B=32 logits to verify zero padding leakage."""
        sub = [dataset[i] for i in range(min(n, len(dataset)))]
        b1_logits = []
        with torch.no_grad():
            for item in sub:
                b1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in BenchmarkBatchCollator.collate([item]).items()}
                b1_logits.append(float(model(b1)["logit"].item()))
            b32_logits = []
            for start in range(0, len(sub), 32):
                b32 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in BenchmarkBatchCollator.collate(sub[start: start + 32]).items()}
                b32_logits.extend(model(b32)["logit"].cpu().tolist())
        return float(np.max(np.abs(np.array(b1_logits) - np.array(b32_logits)))) if b1_logits else 0.0
