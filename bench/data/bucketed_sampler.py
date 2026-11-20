"""Length-bucketed random batch sampler minimizing wasted padding overhead."""

from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class BucketedBatchSampler(Sampler[List[int]]):
    """Packs stochastic mega-batches sorted by noisy sequence length."""

    def __init__(self, lengths: np.ndarray, batch_size: int = 64, seed: int = 0) -> None:
        """Store sequence lengths, target batch size, and base random seed."""
        super().__init__()
        self.lengths = np.asarray(lengths, dtype=np.float32)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.n_samples = len(lengths)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield shuffled batches built from noisy-sorted sequence lengths."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        noisy = self.lengths * rng.uniform(0.85, 1.15, size=self.n_samples)
        order = np.argsort(noisy)
        batches = [order[i: i + self.batch_size].tolist() for i in range(0, self.n_samples, self.batch_size)]
        rng.shuffle(batches)
        for batch in batches:
            if batch:
                yield batch

    def __len__(self) -> int:
        """Return total number of batches per epoch."""
        return max(1, (self.n_samples + self.batch_size - 1) // self.batch_size)
