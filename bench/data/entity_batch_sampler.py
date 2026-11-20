"""Lossless entity-grouped batch sampler visiting every training pair once per epoch."""

from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class EntityGroupedBatchSampler(Sampler[List[int]]):
    """Partitions all indices into paired same-protein chunks without duplication."""

    def __init__(
        self, protein_indices: np.ndarray, drug_indices: np.ndarray, labels: np.ndarray,
        batch_size: int = 32, seed: int = 0, protein_lengths: Optional[np.ndarray] = None,
        max_tokens: int = 24000,
    ) -> None:
        """Index positive and negative examples per protein for lossless chunking."""
        super().__init__()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.epoch = 0
        self.n_samples = len(labels)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.protein_indices = np.asarray(protein_indices, dtype=np.int64)
        self.lengths = protein_lengths if protein_lengths is not None else np.ones(self.n_samples, dtype=np.int64)
        self._num_batches = max(1, (self.n_samples + batch_size - 1) // batch_size)

    def _build_epoch_chunks(self, rng: np.random.Generator) -> List[List[int]]:
        """Partition all N indices into shuffled same-protein pos/neg chunks + singles."""
        prot_pos: Dict[int, List[int]] = defaultdict(list)
        prot_neg: Dict[int, List[int]] = defaultdict(list)
        for idx in rng.permutation(self.n_samples).tolist():
            pid = int(self.protein_indices[idx])
            (prot_pos if self.labels[idx] == 1 else prot_neg)[pid].append(idx)

        chunks: List[List[int]] = []
        singles: List[int] = []
        for pid in set(prot_pos) | set(prot_neg):
            pos_list = prot_pos.get(pid, [])
            neg_list = prot_neg.get(pid, [])
            n_pairs = min(len(pos_list), len(neg_list))
            for k in range(n_pairs):
                chunk = [pos_list[k], neg_list[k]]
                chunks.append(chunk)
            singles.extend(pos_list[n_pairs:])
            singles.extend(neg_list[n_pairs:])

        for i in range(0, len(singles), 2):
            chunks.append(singles[i: i + 2])
        rng.shuffle(chunks)
        return chunks

    def __iter__(self) -> Iterator[List[int]]:
        """Yield shuffled batches covering every training pair exactly once."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        chunks = self._build_epoch_chunks(rng)

        batches, cur, cur_max_len = [], [], 0
        for chunk in chunks:
            c_max = max(int(self.lengths[i]) for i in chunk)
            next_max = max(cur_max_len, c_max)
            if len(cur) >= 4 and (len(cur) + len(chunk) > self.batch_size or next_max * (len(cur) + len(chunk)) > self.max_tokens):
                batches.append(cur)
                cur, cur_max_len = [], 0
            cur.extend(chunk)
            cur_max_len = max(cur_max_len, c_max)
        if cur:
            batches.append(cur)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return expected number of batches per epoch."""
        return self._num_batches
