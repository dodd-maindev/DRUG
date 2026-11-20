"""Disk-cached mean-pooled 512-D embedding extractor for shortcut baselines."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from diagnostics.embedding_store import EmbeddingStore


class PooledFeatureCache:
    """Computes and caches mean-pooled 512-D entity vectors in .npz archives."""

    @staticmethod
    def _load_one(store: EmbeddingStore, kind: str, identifier: str) -> Tuple[str, Optional[np.ndarray]]:
        """Load one entity embedding tensor and compute its 512-D mean vector."""
        try:
            path = store.resolve_drug_path(identifier) if kind == "drug" else store.resolve_protein_path(identifier)
            if path is None:
                return identifier, None
            tensor = store.load_drug(identifier) if kind == "drug" else store.load_protein(identifier)
            return identifier, tensor.mean(0).numpy().astype(np.float32)
        except Exception as error:
            print(f"  skip {identifier}: {error}", flush=True)
            return identifier, None

    @classmethod
    def load_or_compute(
        cls,
        store: EmbeddingStore,
        kind: str,
        identifiers: List[str],
        cache_path: str,
        workers: int = 16,
    ) -> Dict[str, np.ndarray]:
        """Load cached pooled embeddings or compute missing ones concurrently."""
        cache: Dict[str, np.ndarray] = {}
        if os.path.exists(cache_path):
            loaded = np.load(cache_path, allow_pickle=True)
            cache = dict(zip(loaded["ids"].tolist(), loaded["vecs"]))
        missing = [item for item in identifiers if item not in cache]
        print(f"[pooled {kind}] cached={len(cache)}, computing={len(missing)} (workers={workers})", flush=True)

        if missing:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(lambda item: cls._load_one(store, kind, item), missing)
                for index, (identifier, vector) in enumerate(iterator, 1):
                    if vector is not None:
                        cache[identifier] = vector
                    if index % 500 == 0 or index == len(missing):
                        print(f"  [pooled {kind}] {index}/{len(missing)}", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(
            cache_path,
            ids=np.array(list(cache.keys()), dtype=object),
            vecs=np.stack(list(cache.values())) if cache else np.zeros((0, 512)),
        )
        return cache
