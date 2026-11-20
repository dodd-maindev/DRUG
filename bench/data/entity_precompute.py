"""Concurrent multi-entity precomputer aggregating proteins and drugs in RAM."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from bench.data.entity_record_loader import EntityRecordLoader
from diagnostics.embedding_store import EmbeddingStore


class EntityPrecomputer:
    """Concurrently loads unique proteins and drugs into an in-memory dictionary."""

    @staticmethod
    def build_entities(
        protein_ids: List[str], drug_ids: List[str], protein_dir: str, drug_dir: str, ca_dir: Optional[str], workers: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """Load protein and drug entries concurrently using Drive-safe threads."""
        store = EmbeddingStore(protein_dir, drug_dir, trim=False)
        proteins, drugs = {}, {}
        print(f"[precompute] loading {len(protein_ids)} proteins & {len(drug_ids)} drugs (workers={workers})", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            p_futs = [ex.submit(EntityRecordLoader.load_protein, store, ca_dir, p) for p in protein_ids]
            for idx, fut in enumerate(as_completed(p_futs), 1):
                pid, item = fut.result()
                if item is not None:
                    proteins[pid] = item
                if idx % 500 == 0 or idx == len(protein_ids):
                    print(f"  [precompute] proteins {idx}/{len(protein_ids)} ok={len(proteins)}", flush=True)

            d_futs = [ex.submit(EntityRecordLoader.load_drug, store, d) for d in drug_ids]
            for idx, fut in enumerate(as_completed(d_futs), 1):
                did, item = fut.result()
                if item is not None:
                    drugs[did] = item
                if idx % 1000 == 0 or idx == len(drug_ids):
                    print(f"  [precompute] drugs {idx}/{len(drug_ids)} ok={len(drugs)}", flush=True)
        return {"proteins": proteins, "drugs": drugs}
