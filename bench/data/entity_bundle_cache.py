"""Persistent disk bundle caching for fast zero-Drive-I/O training runs."""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from bench.data.entity_precompute import EntityPrecomputer


class EntityBundleCache:
    """Loads or builds a unified float16 entity cache across split CSVs."""

    @staticmethod
    def load_or_build(
        dataframes: List[pd.DataFrame],
        protein_dir: str,
        drug_dir: str,
        ca_dir: Optional[str],
        cache_path: Optional[str],
        workers: int = 4,
    ) -> Dict[str, Dict[str, Any]]:
        """Load cached bundle if complete, else build missing entities and persist."""
        need_proteins = sorted(set(pd.concat([df["protein"] for df in dataframes])))
        need_drugs = sorted(set(pd.concat([df["drug"] for df in dataframes])))

        if cache_path and os.path.exists(cache_path):
            print(f"[cache] loading entity bundle from {cache_path}", flush=True)
            bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            missing_p = [p for p in need_proteins if p not in bundle.get("proteins", {})]
            missing_d = [d for d in need_drugs if d not in bundle.get("drugs", {})]
            if not missing_p and not missing_d:
                print(f"[cache] ready: {len(bundle['proteins'])} proteins, {len(bundle['drugs'])} drugs", flush=True)
                return bundle
            extra = EntityPrecomputer.build_entities(missing_p, missing_d, protein_dir, drug_dir, ca_dir, workers)
            bundle["proteins"].update(extra["proteins"])
            bundle["drugs"].update(extra["drugs"])
        else:
            bundle = EntityPrecomputer.build_entities(need_proteins, need_drugs, protein_dir, drug_dir, ca_dir, workers)

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(bundle, cache_path)
            print(f"[cache] saved entity bundle -> {cache_path}", flush=True)
        return bundle
