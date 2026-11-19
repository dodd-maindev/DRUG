"""Mask-leakage diagnostic comparing B=1 unpadded logits vs padded batches."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from diagnostics.batch_predictor import BatchPredictor
from diagnostics.checkpoint_loader import CheckpointLoader
from diagnostics.common import DiagnosticUtilities
from diagnostics.dataset_reader import PairDatasetReader
from diagnostics.embedding_store import EmbeddingStore


class MaskLeakageAnalyzer:
    """Tests whether model logits depend on padding length or batch grouping."""

    def __init__(self, arguments: Any) -> None:
        """Store CLI arguments for the mask-leakage command."""
        self.arguments = arguments

    @staticmethod
    def _logit_margin(logits: np.ndarray) -> np.ndarray:
        """Compute positive-minus-negative logit margin per sample."""
        return logits[:, 1] - logits[:, 0]

    def run(self) -> None:
        """Compare B=1 reference logits against padded batches and extra PAD tokens."""
        DiagnosticUtilities.configure_determinism()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = np.random.default_rng(self.arguments.seed)
        store = EmbeddingStore(self.arguments.protein_dir, self.arguments.drug_dir, self.arguments.trim)
        reader = PairDatasetReader(self.arguments.col_drug, self.arguments.col_prot, self.arguments.col_label)
        dataframe = reader.read_pairs(self.arguments.test_csv)
        dataframe = dataframe.iloc[generator.permutation(len(dataframe))]

        items = []
        for drug_id, protein_id in zip(dataframe["drug"], dataframe["protein"]):
            if store.resolve_drug_path(drug_id) is None or store.resolve_protein_path(protein_id) is None:
                continue
            items.append((store.load_protein(protein_id), store.load_drug(drug_id)))
            if len(items) >= self.arguments.n_pairs:
                break

        print(f"[leak] {len(items)} pairs, trim={self.arguments.trim}, device={device}", flush=True)
        model = CheckpointLoader.load_checkpoint_model(
            self.arguments.repo, self.arguments.cfg_spec, self.arguments.model_spec, self.arguments.ckpt, device
        )
        caller = BatchPredictor.build_forward_caller(model)
        ref_logits, _, _ = BatchPredictor.predict_logits(caller, items, 1, device)
        ref_margin = self._logit_margin(ref_logits)

        arms: Dict[str, Any] = {}
        order = generator.permutation(len(items))
        arms[f"B={self.arguments.batch} random batches"] = BatchPredictor.predict_logits(
            caller, items, self.arguments.batch, device, order=order
        )
        for extra in (8, 32, 128):
            arms[f"B=1 + {extra} extra PAD"] = BatchPredictor.predict_logits(
                caller, items, 1, device, extra_padding=extra
            )

        rows: List[Dict[str, Any]] = []
        for arm_name, (logits, _, _) in arms.items():
            diffs = np.abs(self._logit_margin(logits) - ref_margin)
            rows.append({
                "arm": arm_name, "max_abs_margin_diff": float(diffs.max()),
                "mean_abs": float(diffs.mean()), "p99_abs": float(np.quantile(diffs, 0.99)),
                "flip_rate": float((logits.argmax(1) != ref_logits.argmax(1)).mean()),
            })
        print("\n=== logit-margin (l1-l0) difference vs B=1 reference ===", flush=True)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3e}"), flush=True)

        worst = max(row["max_abs_margin_diff"] for row in rows)
        if worst > 1e-3:
            verdict = "PAD-SENSITIVE: logits depend on padding/batch composition (mask leakage in Conv1d)."
        elif worst > 1e-4:
            verdict = "BORDERLINE: small padding dependence; verify on more samples."
        else:
            verdict = "OK: no measurable padding dependence at this precision."
        print("VERDICT (leak):", verdict, flush=True)
        DiagnosticUtilities.save_json({"rows": rows, "verdict": verdict}, self.arguments.out, "leak.json")
