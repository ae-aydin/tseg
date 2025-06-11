import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class SlideBalancedSampler(Sampler):
    def __init__(self, dataset, samples_per_epoch: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch

        self.slide_to_indices = defaultdict(list)
        for idx, slide_name in enumerate(self.dataset.df["slide_name"]):
            self.slide_to_indices[slide_name].append(idx)

        self.slide_ids = list(self.slide_to_indices.keys())

    def __iter__(self):
        slide_deck = np.random.choice(
            self.slide_ids, self.samples_per_epoch, replace=True
        )
        epoch_indices = []
        for slide_id in slide_deck:
            possible_indices = self.slide_to_indices[slide_id]
            chosen_index = random.choice(possible_indices)
            epoch_indices.append(chosen_index)
        return iter(epoch_indices)

    def __len__(self):
        return self.samples_per_epoch
