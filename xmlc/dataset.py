import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from itertools import chain
from typing import List, Tuple, Set

class XMLDataset(Dataset):
    """ (Extreme) Multi-Labeling Dataset """

    def __init__(self,
        input_dataset:Dataset,
        labels:List[Set[int]],
        num_candidates:int,
        label_pool:List[int] =None
    ) -> None:
        # save the input dataset and labels
        self.input_dataset = input_dataset
        self.labels = [set(l) for l in labels]
        # number of candidates per example
        self.num_candidates = num_candidates
        # create a fill list of all labels in the dataset if not provided
        if label_pool is None:
            label_pool = list(chain(*self.labels))
            label_pool = np.unique(label_pool)
        self.label_pool = label_pool
        # some quick tests
        n, m = len(self.labels), len(self.input_dataset)
        assert n == m, "Targets (%i) and Input Dataset (%i) do not align!" % (n, m)
        assert all(l in self.label_pool for ls in self.labels for l in ls), "Some Labels not contained in Label Pool!"
        assert n >= num_candidates, "Too many candidates (%i) for labels (%i)!" % (num_candidates, n)

    def __len__(self) -> int:
        return len(self.input_dataset)

    def __getitem__(self, index:int) -> Tuple[Tensor]:
        # get inputs and labels from index
        inputs = self.input_dataset[index]
        labels = self.labels[index]
        # at most half of the labels are positives
        n_positives = min(len(labels), self.num_candidates//2)
        positives = np.random.choice(tuple(labels), size=n_positives, replace=False)
        # the rest are negative samples
        n_negatives = self.num_candidates - n_positives
        negative_pool = tuple(filter(lambda l: l not in labels, self.label_pool))
        negatives = np.random.choice(negative_pool, size=n_negatives, replace=False)
        # concatenate candidates
        candidates = np.concatenate((positives, negatives))
        candidates = torch.LongTensor(candidates)
        labels = torch.FloatTensor([1] * n_positives + [0] * n_negatives)
        # return all features
        return inputs + (candidates, labels)
