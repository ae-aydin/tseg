from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class LimitedSampler(Sampler):
    def __init__(
        self, dataset, max_tiles_per_slide: int, shuffle: bool = True, seed: int = 42
    ):
        self.dataset = dataset
        self.max_tiles_per_slide = max_tiles_per_slide
        self.shuffle = shuffle
        self.seed = seed
        self.slide_to_indices = defaultdict(list)
        for idx, slide_name in enumerate(self.dataset.df["slide_name"]):
            self.slide_to_indices[slide_name].append(idx)
        self.indices = self._generate_indices()

    def _generate_indices(self):
        rng = np.random.default_rng(self.seed)
        indices = []
        for slide, idxs in self.slide_to_indices.items():
            if self.shuffle:
                idxs = rng.permutation(idxs)
            else:
                idxs = np.array(idxs)
            limited = idxs[: self.max_tiles_per_slide]
            indices.extend(limited)
        if self.shuffle:
            indices = rng.permutation(indices)
        return list(indices)

    def __iter__(self):
        if self.shuffle:
            self.indices = self._generate_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
