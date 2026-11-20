"""Entity-grouped batch sampler yielding within-protein and within-drug pairs."""

from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Sampler


class EntityGroupedBatchSampler(Sampler[List[int]]):
    """Samples balanced batches with paired positive/negative examples per entity."""

    def __init__(
        self,
        protein_indices: np.ndarray,
        drug_indices: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        """Index positive and negative sample lists by protein and drug IDs."""
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.n_samples = len(labels)

        self.prot_pos: Dict[int, List[int]] = defaultdict(list)
        self.prot_neg: Dict[int, List[int]] = defaultdict(list)
        self.drug_pos: Dict[int, List[int]] = defaultdict(list)
        self.drug_neg: Dict[int, List[int]] = defaultdict(list)
        for idx, (pid, did, y) in enumerate(zip(protein_indices, drug_indices, labels)):
            (self.prot_pos if y == 1 else self.prot_neg)[int(pid)].append(idx)
            (self.drug_pos if y == 1 else self.drug_neg)[int(did)].append(idx)

        self.paired_proteins = [p for p in self.prot_pos if p in self.prot_neg]
        self.paired_drugs = [d for d in self.drug_pos if d in self.drug_neg]
        self._num_batches = max(1, (self.n_samples + batch_size - 1) // batch_size)

    def _build_group(self, entity_id: int, pos_map: Dict[int, List[int]], neg_map: Dict[int, List[int]], rng: np.random.Generator) -> List[int]:
        """Build one (1 pos + up to 2 neg) entity group sharing the same anchor."""
        pos_choice = int(rng.choice(pos_map[entity_id]))
        neg_pool = neg_map[entity_id]
        k_neg = min(2, len(neg_pool))
        neg_choices = rng.choice(neg_pool, size=k_neg, replace=False).tolist()
        return [pos_choice] + [int(x) for x in neg_choices]

    def __iter__(self) -> Iterator[List[int]]:
        """Yield shuffled batches composed predominantly of entity-matched groups."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        ordered: List[int] = []

        p_order = rng.permutation(self.paired_proteins) if self.paired_proteins else []
        d_order = rng.permutation(self.paired_drugs) if self.paired_drugs else []
        p_idx, d_idx = 0, 0
        while len(ordered) < self.n_samples and (p_idx < len(p_order) or d_idx < len(d_order)):
            if p_idx < len(p_order):
                ordered.extend(self._build_group(int(p_order[p_idx]), self.prot_pos, self.prot_neg, rng))
                p_idx += 1
            if d_idx < len(d_order) and len(ordered) < self.n_samples:
                ordered.extend(self._build_group(int(d_order[d_idx]), self.drug_pos, self.drug_neg, rng))
                d_idx += 1

        if len(ordered) < self.n_samples:
            extra = rng.permutation(self.n_samples)[: self.n_samples - len(ordered)].tolist()
            ordered.extend(extra)

        ordered = ordered[: self.n_samples]
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start: start + self.batch_size]
            if len(batch) >= 2:
                yield batch

    def __len__(self) -> int:
        """Return expected number of batches per epoch."""
        return self._num_batches
