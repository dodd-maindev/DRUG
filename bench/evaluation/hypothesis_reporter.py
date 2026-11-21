"""Cross-arm hypothesis reporting (H-A1 through H-A5 and CEF summary table)."""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import torch

from bench.data.entity_bundle_cache import EntityBundleCache
from bench.data.pair_dataset import BenchmarkPairDataset
from bench.data.track_b_builder import TrackBBuilder
from bench.evaluation.attention_js_eval import AttentionJsEvaluator
from bench.evaluation.pocket_auprc_eval import PocketLocalizationEvaluator
from bench.evaluation.pocket_mask_extractor import PocketMaskExtractor
from bench.evaluation.report_formatter import ReportFormatter
from bench.evaluation.within_protein_eval import WithinProteinEvaluator
from bench.training.train_config import ArmRegistry
from diagnostics.dataset_reader import PairDatasetReader


class HypothesisReporter:
    """Evaluates saved arm checkpoints and builds H-A1..H-A5 verification tables."""

    @staticmethod
    def _load_oof(oof_dir: Optional[str], track: str, seed: int) -> Optional[dict]:
        """Load cached OOF coarse state for Delta-Interface arms if present."""
        if not oof_dir:
            return None
        path = os.path.join(oof_dir, f"oof_c1_track{track.upper()}_seed{seed}.pt")
        return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None

    @classmethod
    def evaluate_all(
        cls, arms: List[str], runs_root: str, train_csv: str, val_csv: str, test_csv: str,
        protein_dir: str, drug_dir: str, ca_dir: str, bundle_path: str, oof_dir: str,
        overlap_csv: Optional[str], track: str, out_dir: str,
    ) -> Dict[str, Any]:
        """Run global, H-A1, H-A2, and H-A3 evaluations across all trained arms."""
        splits = PairDatasetReader().load_splits(train_csv, val_csv, test_csv)
        if track.upper() == "B":
            splits = TrackBBuilder.convert_splits(splits, seed=42)
        entities = EntityBundleCache.load_or_build(
            [splits["train"], splits["validation"], splits["test"]], protein_dir, drug_dir, ca_dir, bundle_path
        )
        pocket_masks = PocketMaskExtractor.build_pocket_masks(overlap_csv, entities) if (overlap_csv and os.path.exists(overlap_csv)) else {}
        test_pairs = set(zip(splits["test"]["drug"], splits["test"]["protein"]))
        test_pockets = {k: v for k, v in pocket_masks.items() if k in test_pairs}

        rows: List[Dict[str, Any]] = []
        within_maps: Dict[str, Dict[str, float]] = {}
        for arm in arms:
            arm_dir = os.path.join(runs_root, arm)
            sum_path = os.path.join(arm_dir, "summary.json")
            if not os.path.exists(sum_path):
                continue
            with open(sum_path, "r", encoding="utf-8") as handle:
                base = json.load(handle)

            npz_paths = sorted(glob.glob(os.path.join(arm_dir, "preds_seed*.npz")))
            ha1 = WithinProteinEvaluator.evaluate_preds(npz_paths)
            within_maps[arm] = ha1.pop("per_protein_auroc")

            cfg = ArmRegistry.get(arm)
            ckpt0 = torch.load(os.path.join(arm_dir, "ckpt_seed0.pt"), map_location="cpu", weights_only=False)
            coarse0 = cls._load_oof(oof_dir, track, 0) if cfg.family == "delta" else None
            te_ds = BenchmarkPairDataset(splits["test"], entities, cfg.trim_protein_token)
            ha2 = AttentionJsEvaluator.evaluate(cfg, ckpt0, te_ds, coarse0)
            ha3 = PocketLocalizationEvaluator.evaluate(cfg, ckpt0, entities, pocket_masks, coarse0) if pocket_masks else {}
            ha3_te = PocketLocalizationEvaluator.evaluate(cfg, ckpt0, entities, test_pockets, coarse0) if test_pockets else {}
            rows.append(ReportFormatter.build_row(arm, base, ha1, ha2, ha3, ha3_te))

        boot = {}
        for cand in ("G1", "M1"):
            if cand in within_maps:
                for ref in ("A0", "A2", "C2", "M1"):
                    if ref != cand and ref in within_maps:
                        boot[f"{cand}_vs_{ref}"] = WithinProteinEvaluator.paired_bootstrap(within_maps[cand], within_maps[ref])
        return ReportFormatter.emit_reports(rows, boot, out_dir)
