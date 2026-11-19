"""Facade coordinating coverage, degree, and pooled-embedding shortcut diagnostics."""

import sys
from typing import Any, Dict

import pandas as pd

from diagnostics.common import DiagnosticUtilities
from diagnostics.dataset_reader import PairDatasetReader
from diagnostics.degree_baseline_evaluator import DegreeBaselineEvaluator
from diagnostics.embedding_store import EmbeddingStore
from diagnostics.pooled_baseline_evaluator import PooledBaselineEvaluator


class ShortcutAnalyzer:
    """Runs embedding coverage, degree baselines, and pooled-modality baselines."""

    def __init__(self, arguments: Any) -> None:
        """Store CLI arguments for the shortcut diagnostic command."""
        self.arguments = arguments

    def run(self) -> None:
        """Execute all shortcut baselines and save shortcut.json."""
        DiagnosticUtilities.configure_determinism()
        reader = PairDatasetReader(
            self.arguments.col_drug,
            self.arguments.col_prot,
            self.arguments.col_label,
        )
        splits = reader.load_splits(
            self.arguments.train_csv,
            self.arguments.val_csv,
            self.arguments.test_csv,
        )
        if "train" not in splits or "test" not in splits:
            sys.exit("[!] need at least --train-csv and --test-csv")
        store = EmbeddingStore(
            self.arguments.protein_dir,
            self.arguments.drug_dir,
            self.arguments.trim,
        )
        results: Dict[str, Any] = {}
        if self.arguments.protein_dir and self.arguments.drug_dir:
            results["coverage"] = store.compute_coverage(splits)
            print("\n=== embedding coverage per split ===", flush=True)
            print(pd.DataFrame(results["coverage"]).to_string(index=False), flush=True)
        results["degree"] = DegreeBaselineEvaluator.evaluate(splits)
        if self.arguments.protein_dir and self.arguments.drug_dir:
            results["pooled"] = PooledBaselineEvaluator.evaluate(
                splits,
                store,
                self.arguments.out,
                self.arguments.trim,
                self.arguments.seeds,
            )
        combined = results["degree"] + results.get("pooled", [])
        print("\n=== TEST metrics of shortcut baselines (threshold = max-MCC on val) ===", flush=True)
        print(pd.DataFrame(combined).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        DiagnosticUtilities.save_json(results, self.arguments.out, "shortcut.json")
