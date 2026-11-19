"""Disk-cached mean-pooled 512-D embedding extractor for shortcut baselines."""

import os
from typing import Dict, List

import numpy as np

from diagnostics.embedding_store import EmbeddingStore


class PooledFeatureCache:
    """Computes and caches mean-pooled 512-D entity vectors in .npz archives."""

    @staticmethod
    def load_or_compute(
        store: EmbeddingStore,
        kind: str,
        identifiers: List[str],
        cache_path: str,
    ) -> Dict[str, np.ndarray]:
        """Load cached pooled embeddings or compute missing ones from disk."""
        cache: Dict[str, np.ndarray] = {}
        if os.path.exists(cache_path):
            loaded = np.load(cache_path, allow_pickle=True)
            cache = dict(zip(loaded["ids"].tolist(), loaded["vecs"]))
        missing = [item for item in identifiers if item not in cache]
        print(f"[pooled {kind}] cached={len(cache)}, computing={len(missing)}", flush=True)
        for index, identifier in enumerate(missing, 1):
            try:
                path = store.resolve_drug_path(identifier) if kind == "drug" else store.resolve_protein_path(identifier)
                if path is None:
                    continue
                tensor = store.load_drug(identifier) if kind == "drug" else store.load_protein(identifier)
                cache[identifier] = tensor.mean(0).numpy().astype(np.float32)
            except Exception as error:
                print(f"  skip {identifier}: {error}", flush=True)
            if index % 500 == 0:
                print(f"  {index}/{len(missing)}", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            ids=np.array(list(cache.keys()), dtype=object),
            vecs=np.stack(list(cache.values())) if cache else np.zeros((0, 512)),
        )
        return cache
