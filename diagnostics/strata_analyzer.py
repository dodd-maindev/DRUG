"""Stratified evaluation by 3D protein structure source (PDB vs AlphaFold)."""

import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from diagnostics.batch_predictor import BatchPredictor
from diagnostics.checkpoint_loader import CheckpointLoader
from diagnostics.common import DiagnosticUtilities
from diagnostics.dataset_reader import PairDatasetReader
from diagnostics.embedding_store import EmbeddingStore
from diagnostics.metrics import MetricEvaluator
from diagnostics.structure_locator import ProteinStructureLocator


class StrataAnalyzer:
    """Evaluates checkpoint performance stratified by PDB vs AlphaFold structure source."""

    def __init__(self, arguments: Any) -> None:
        """Store CLI arguments for the stratified source evaluation command."""
        self.arguments = arguments

    def _obtain_predictions(self, reader: PairDatasetReader) -> pd.DataFrame:
        """Load precomputed predictions or run checkpoint inference on test CSV."""
        if self.arguments.pred_csv:
            pred_df = pd.read_csv(self.arguments.pred_csv)
            prot_col = "protein" if "protein" in pred_df.columns else "prot"
            label_col = "label" if "label" in pred_df.columns else "y"
            return pd.DataFrame({
                "drug": pred_df["drug"].astype(str), "protein": pred_df[prot_col].astype(str),
                "label": pred_df[label_col].astype(int), "prob": pred_df["prob"].astype(float),
            })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = EmbeddingStore(self.arguments.protein_dir, self.arguments.drug_dir, self.arguments.trim)
        test_df = reader.read_pairs(self.arguments.test_csv)
        keep = [i for i, (d, p) in enumerate(zip(test_df["drug"], test_df["protein"])) if store.resolve_drug_path(d) and store.resolve_protein_path(p)]
        print(f"[strata] test pairs with embeddings: {len(keep)}/{len(test_df)}", flush=True)
        test_df = test_df.iloc[keep].reset_index(drop=True)
        items = [(store.load_protein(p), store.load_drug(d)) for d, p in zip(test_df["drug"], test_df["protein"])]
        model = CheckpointLoader.load_checkpoint_model(self.arguments.repo, self.arguments.cfg_spec, self.arguments.model_spec, self.arguments.ckpt, device)
        caller = BatchPredictor.build_forward_caller(model)
        order = np.random.default_rng(self.arguments.seed).permutation(len(items))
        logits, _, _ = BatchPredictor.predict_logits(caller, items, self.arguments.batch, device, order=order)
        test_df["prob"] = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        os.makedirs(self.arguments.out, exist_ok=True)
        test_df.to_csv(os.path.join(self.arguments.out, "test_predictions.csv"), index=False)
        return test_df

    def run(self) -> None:
        """Compute stratified PDB vs AlphaFold metrics and source-only prior AUROC."""
        DiagnosticUtilities.configure_determinism()
        reader = PairDatasetReader(self.arguments.col_drug, self.arguments.col_prot, self.arguments.col_label)
        test_df = self._obtain_predictions(reader)
        index = ProteinStructureLocator.index_structures(self.arguments.struct_dir)
        overrides: Dict[str, str] = {}
        if self.arguments.source_map:
            map_df = pd.read_csv(self.arguments.source_map, sep=None, engine="python")
            overrides = dict(zip(map_df.iloc[:, 0].astype(str), map_df.iloc[:, 1].astype(str)))
        src_map = {p: ProteinStructureLocator.infer_structure_source(ProteinStructureLocator.find_structure(index, p), p, overrides) for p in sorted(set(test_df["protein"]))}
        test_df["source"] = test_df["protein"].map(src_map)
        print("\nprotein source counts (unique in test):", dict(Counter(src_map.values())), flush=True)

        rows: List[Dict[str, Any]] = []
        for name, group in [("ALL", test_df)] + [(s, g) for s, g in test_df.groupby("source")]:
            scores = MetricEvaluator.compute_classification_metrics(group["label"].values, group["prob"].values)
            lo, hi = MetricEvaluator.bootstrap_auroc_interval(group["label"].values, group["prob"].values)
            rows.append({"stratum": name, "n": scores["count"], "n_prot": group["protein"].nunique(), "pos_rate": scores["positive_rate"], "auroc": scores["auroc"], "ci_lo": lo, "ci_hi": hi, "auprc": scores["auprc"], "acc": scores["accuracy"], "mcc": scores["mcc"]})
        result: Dict[str, Any] = {"strata": rows}
        print("\n=== checkpoint metrics by protein structure source (thr=0.5) ===", flush=True)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        if self.arguments.train_csv:
            from sklearn.metrics import roc_auc_score
            train_df = reader.read_pairs(self.arguments.train_csv)
            train_df["source"] = train_df["protein"].map(lambda p: src_map.get(p) or ProteinStructureLocator.infer_structure_source(ProteinStructureLocator.find_structure(index, p), p, overrides))
            rates = train_df.groupby("source")["label"].agg(["mean", "count"])
            print("\ntrain pos-rate by source:\n" + rates.to_string(), flush=True)
            prior = train_df["label"].mean()
            smoothed = {s: (r["mean"] * r["count"] + prior * 20) / (r["count"] + 20) for s, r in rates.iterrows()}
            if len(set(test_df["label"].tolist())) > 1:
                src_auc = float(roc_auc_score(test_df["label"], test_df["source"].map(lambda s: smoothed.get(s, prior))))
                print(f"source-only predictor AUROC on test: {src_auc:.4f}", flush=True)
                result["source_only_auroc"] = src_auc
        DiagnosticUtilities.save_json(result, self.arguments.out, "strata.json")
