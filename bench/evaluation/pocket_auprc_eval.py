"""H-A3 evaluation: zero-shot binding pocket localization AUPRC and Hit@10."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from bench.data.pair_dataset import BenchmarkBatchCollator, BenchmarkPairDataset
from bench.training.model_factory import TrainerHelpers
from bench.training.train_config import ArmConfig


class PocketLocalizationEvaluator:
    """Evaluates residue attention maps against 4.5A co-crystal pocket masks."""

    @staticmethod
    def evaluate(
        cfg: ArmConfig, state_dict: dict, entities: Dict[str, Dict[str, Any]],
        pocket_masks: Dict[Tuple[str, str], Dict[str, np.ndarray]], coarse_state: dict | None = None,
    ) -> Dict[str, float]:
        """Compute mean pocket AUPRC, AUROC, AUPRC lift, Top-10 hit rate, and distance."""
        rows = [{"drug": d, "protein": p, "label": 1} for (d, p) in sorted(pocket_masks)]
        if not rows:
            return {"n_complexes": 0, "pocket_auprc": 0.0, "pocket_auroc": 0.0, "auprc_lift": 0.0, "top10_hit": 0.0, "attn_lig_dist_A": 0.0}

        ds = BenchmarkPairDataset(pd.DataFrame(rows), entities, cfg.trim_protein_token)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=BenchmarkBatchCollator.collate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tau_b = coarse_state["tau_b"] if coarse_state else 1.0
        b_sd = coarse_state["state_dict"] if coarse_state else None
        model = TrainerHelpers.build_model(cfg, {"tau_b": tau_b, "state_dict": b_sd} if coarse_state else None, device)
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()

        auprcs, aurocs, lifts, hit10s, dists = [], [], [], [], []
        with torch.no_grad():
            for raw in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
                attn = model(batch, return_maps=True)["prot_attn"].cpu().numpy()
                p_mask = raw["protein_mask"].numpy()
                ca = raw["ca_coords"].numpy()
                valid_ca = raw["valid_ca"].numpy()
                for i in range(len(raw["label"])):
                    key = (raw["drug_id"][i], raw["uniprot_id"][i])
                    gt_info = pocket_masks[key]
                    gt = gt_info["pocket_mask"]
                    n_res = min(int((~p_mask[i]).sum()), len(gt))
                    gt_trim = gt[:n_res]
                    if gt_trim.sum() == 0 or gt_trim.all():
                        continue
                    scores = attn[i, :n_res]
                    ap = float(average_precision_score(gt_trim, scores))
                    auprcs.append(ap)
                    lifts.append(ap / float(gt_trim.mean()))
                    aurocs.append(float(roc_auc_score(gt_trim, scores)))
                    top10 = np.argsort(scores)[-min(10, n_res):]
                    hit10s.append(float(gt_trim[top10].any()))
                    ca_v = (~p_mask[i]) & valid_ca[i]
                    if ca_v.any():
                        w = attn[i, ca_v]
                        w = w / max(w.sum(), 1e-10)
                        center = (w[:, None] * ca[i, ca_v]).sum(axis=0)
                        dists.append(float(np.linalg.norm(center - gt_info["lig_center"])))

        return {
            "n_complexes": len(auprcs),
            "pocket_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
            "pocket_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
            "auprc_lift": float(np.mean(lifts)) if lifts else 0.0,
            "top10_hit": float(np.mean(hit10s)) if hit10s else 0.0,
            "attn_lig_dist_A": float(np.mean(dists)) if dists else 0.0,
        }
