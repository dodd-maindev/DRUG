"""Orchestrator executing Step 1 audit commands (Pair-kNN, C1, PDB overlap)."""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from diagnostics.common import DiagnosticUtilities
from diagnostics.dataset_reader import PairDatasetReader
from diagnostics.embedding_store import EmbeddingStore
from diagnostics.pooled_feature_cache import PooledFeatureCache
from research.cif_ligand_matcher import CifLigandMatcher
from research.coarse_trainer import CoarseBaselineTrainer
from research.pair_knn_auditor import PairKnnAuditor


class StepOneAuditRunner:
    """Coordinates Step 1 Neighborhood Lookup, C1 CEF, and PDB ligand audits."""

    @staticmethod
    def _load_pooled_entities(
        splits: Dict[str, pd.DataFrame],
        protein_dir: str,
        drug_dir: str,
        out_dir: str,
        protein_npz: str | None,
        drug_npz: str | None,
        trim_protein: bool,
    ) -> tuple:
        """Load cached .npz pooled vectors or compute missing ones cleanly."""
        proteins = sorted(set(pd.concat([s["protein"] for s in splits.values()])))
        drugs = sorted(set(pd.concat([s["drug"] for s in splits.values()])))
        prot_store = EmbeddingStore(protein_dir, drug_dir, trim=trim_protein)
        drug_store = EmbeddingStore(protein_dir, drug_dir, trim=True)

        p_cache = protein_npz or os.path.join(out_dir, f"pooled_protein_{'trim' if trim_protein else 'notrim'}.npz")
        d_cache = drug_npz or os.path.join(out_dir, "pooled_drug_trim.npz")
        pv = PooledFeatureCache.load_or_compute(prot_store, "protein", proteins, p_cache)
        dv = PooledFeatureCache.load_or_compute(drug_store, "drug", drugs, d_cache)
        return pv, dv

    @classmethod
    def run_pair_knn(cls, args: Any) -> List[Dict[str, Any]]:
        """Run Pair-kNN lookup baseline on the target train/val/test CSVs."""
        splits = PairDatasetReader().load_splits(args.train_csv, args.val_csv, args.test_csv)
        pv, _ = cls._load_pooled_entities(splits, args.protein_dir, args.drug_dir, args.out, args.protein_npz, args.drug_npz, args.trim_protein)
        rows = PairKnnAuditor.evaluate(args.train_csv, args.val_csv, args.test_csv, pv, [None, 50, 10])
        print("\n=== Pair-kNN Audit ===", flush=True)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        DiagnosticUtilities.save_json({"pair_knn": rows}, args.out, "pair_knn.json")
        return rows

    @classmethod
    def run_c1_coarse(cls, args: Any) -> Dict[str, Any]:
        """Train C1 coarse baseline across seeds and compute CEF."""
        splits = PairDatasetReader().load_splits(args.train_csv, args.val_csv, args.test_csv)
        pv, dv = cls._load_pooled_entities(splits, args.protein_dir, args.drug_dir, args.out, args.protein_npz, args.drug_npz, args.trim_protein)
        summary = CoarseBaselineTrainer.evaluate(splits, dv, pv, args.ref_auroc, args.out, seeds=args.seeds)
        DiagnosticUtilities.save_json(summary, args.out, "c1_coarse.json")
        return summary

    @classmethod
    def run_pdb_overlap(cls, args: Any) -> Dict[str, Any]:
        """Count positive DrugBank pairs matching co-crystallized PDB ligands."""
        csvs = [p for p in (args.train_csv, args.val_csv, args.test_csv) if p]
        summary = CifLigandMatcher.evaluate(csvs, args.struct_dir, args.out, fetch_rcsb=not args.no_rcsb, tanimoto_threshold=args.tanimoto)
        DiagnosticUtilities.save_json(summary, args.out, "pdb_overlap.json")
        return summary
