"""Fast mean+max loader for precomputed 3DICE ESM-IF1 and Uni-Mol embeddings."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional, Tuple

import torch

from diagnostics.embedding_store import EmbeddingStore


class Pooled3DLoader:
    """Extracts and caches 1024-D (mean+max) vectors from entity_bundle or raw .pt dirs."""

    @staticmethod
    def _pool_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate mean and max across token axis as fp16 [1024]."""
        t = tensor.float()
        return torch.cat([t.mean(dim=0), t.max(dim=0).values], dim=0).half()

    @classmethod
    def _from_bundle(cls, bundle_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Pool all proteins and drugs directly from entity_bundle.pt in seconds."""
        print(f"[pooled3d] pooling mean+max from {bundle_path}", flush=True)
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        prots = {pid: cls._pool_tensor(rec["emb"]) for pid, rec in bundle["proteins"].items()}
        drugs = {did: cls._pool_tensor(rec["emb"]) for did, rec in bundle["drugs"].items()}
        return {"proteins": prots, "drugs": drugs}

    @classmethod
    def _load_one(cls, store: EmbeddingStore, kind: str, ident: str) -> Tuple[str, Optional[torch.Tensor]]:
        """Load one raw entity embedding from Drive and return fp16 [1024]."""
        try:
            if (store.resolve_drug_path(ident) if kind == "drug" else store.resolve_protein_path(ident)) is None:
                return ident, None
            t = store.load_drug(ident) if kind == "drug" else store.load_protein(ident)
            return ident, cls._pool_tensor(t)
        except Exception:
            return ident, None

    @classmethod
    def load_or_build(
        cls,
        protein_ids: Iterable[str],
        drug_ids: Iterable[str],
        protein_dir: str,
        drug_dir: str,
        bundle_path: Optional[str],
        cache_path: str,
        workers: int = 8,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load cached pooled 3D features or build from bundle/raw directories."""
        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[pooled3d] loaded {len(data['proteins'])} proteins, {len(data['drugs'])} drugs", flush=True)
            return data
        if bundle_path and os.path.exists(bundle_path):
            data = cls._from_bundle(bundle_path)
        else:
            store = EmbeddingStore(protein_dir, drug_dir, trim=False)
            store_d = EmbeddingStore(protein_dir, drug_dir, trim=True)
            p_list, d_list = sorted(set(protein_ids)), sorted(set(drug_ids))
            print(f"[pooled3d] scanning {len(p_list)} proteins & {len(d_list)} drugs from disk", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                p_res = list(ex.map(lambda x: cls._load_one(store, "protein", x), p_list))
                d_res = list(ex.map(lambda x: cls._load_one(store_d, "drug", x), d_list))
            data = {
                "proteins": {k: v for k, v in p_res if v is not None},
                "drugs": {k: v for k, v in d_res if v is not None},
            }
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(data, cache_path)
        print(f"[pooled3d] saved -> {cache_path}", flush=True)
        return data
