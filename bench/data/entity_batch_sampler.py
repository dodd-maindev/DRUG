"""Entity-grouped length-bucketed batch sampler for conditional within-entity loss."""

from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class EntityGroupedBatchSampler(Sampler[List[int]]):
    """Samples paired pos/neg entity groups packed with dynamic length token budget."""

    def __init__(
        self, protein_indices: np.ndarray, drug_indices: np.ndarray, labels: np.ndarray,
        batch_size: int = 64, seed: int = 0, protein_lengths: Optional[np.ndarray] = None,
        max_tokens: int = 24000,
    ) -> None:
        """Index positive and negative sample lists by protein and drug IDs."""
        super().__init__()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.epoch = 0
        self.n_samples = len(labels)
        self.lengths = protein_lengths if protein_lengths is not None else np.ones(self.n_samples, dtype=np.int64)

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

    def _build_group(self, eid: int, pos_map: Dict[int, List[int]], neg_map: Dict[int, List[int]], rng: np.random.Generator) -> List[int]:
        """Build one (1 pos + up to 2 neg) entity group sharing the same anchor."""
        pos_choice = int(rng.choice(pos_map[eid]))
        neg_pool = neg_map[eid]
        neg_choices = rng.choice(neg_pool, size=min(2, len(neg_pool)), replace=False).tolist()
        return [pos_choice] + [int(x) for x in neg_choices]

    def __iter__(self) -> Iterator[List[int]]:
        """Yield shuffled batches respecting both batch_size and max padded token budget."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups: List[List[int]] = []
        total = 0

        p_order = rng.permutation(self.paired_proteins) if self.paired_proteins else []
        d_order = rng.permutation(self.paired_drugs) if self.paired_drugs else []
        p_idx, d_idx = 0, 0
        while total < self.n_samples and (p_idx < len(p_order) or d_idx < len(d_order)):
            if p_idx < len(p_order):
                g = self._build_group(int(p_order[p_idx]), self.prot_pos, self.prot_neg, rng)
                groups.append(g)
                total += len(g)
                p_idx += 1
            if d_idx < len(d_order) and total < self.n_samples:
                g = self._build_group(int(d_order[d_idx]), self.drug_pos, self.drug_neg, rng)
                groups.append(g)
                total += len(g)
                d_idx += 1

        if total < self.n_samples:
            extra = rng.permutation(self.n_samples)[: self.n_samples - total].tolist()
            groups.extend([[i] for i in extra])

        groups.sort(key=lambda g: max(self.lengths[i] for i in g) * rng.uniform(0.85, 1.15))
        batches, cur, cur_max_len = [], [], 0
        for g in groups:
            g_max = max(int(self.lengths[i]) for i in g)
            next_max = max(cur_max_len, g_max)
            if len(cur) >= 4 and (len(cur) + len(g) > self.batch_size or next_max * (len(cur) + len(g)) > self.max_tokens):
                batches.append(cur)
                cur, cur_max_len = [], 0
            cur.extend(g)
            cur_max_len = max(cur_max_len, g_max)
        if len(cur) >= 2:
            batches.append(cur)
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return expected number of batches per epoch."""
        return self._num_batches
