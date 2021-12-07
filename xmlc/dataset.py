import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from .utils import build_group_membership
from random import sample
from itertools import chain
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

class NamedTensorDataset(TensorDataset):
    """ Named Tensor Dataset """
    
    def __init__(self, **named_tensors) -> None:
        # get tupel of tensor names
        self.names = tuple(named_tensors.keys())
        tensors = (named_tensors[n] for n in self.names)
        # initialize tensor dataset
        super(NamedTensorDataset, self).__init__(*tensors)

    def __getitem__(self, index) -> Dict[str, Tensor]:
        # get tensors and add names
        tensors = tuple(tensor[index] for tensor in self.tensors)
        return dict(zip(self.names, tensors))

class MultiLabelDataset(Dataset):
    """ Simple Multi-label Classification Dataset """
    
    def __init__(self,
        input_dataset:Dataset,
        labels:List[Set[int]],
        label_pool:Set[int],
        num_candidates:int
    ) -> None:
        # save arguments
        self.input_dataset = input_dataset
        self.num_candidates = min(num_candidates, len(label_pool))
        self.labels = [set(l) for l in labels]
        self.label_pool = label_pool
        # check if candidate sampling is needed
        self.sampling_disabled = (num_candidates == len(self.label_pool))
        # some quick tests
        n, m = len(self.labels), len(self.input_dataset)
        assert n == m, "Labels (%i) and Input Dataset (%i) do not align!" % (n, m)
        assert all(l in label_pool for ls in labels for l in ls), "Not all labels are present in the label pool!"
        assert self.num_candidates <= len(self.label_pool)
        
    def sample_candidates(self, index:int, positives:Set[int]) -> Set[int]:
        # choose candidates completely random but make sure positives are contained
        positives = sample(positives, k=min(len(positives), self.num_candidates//2))
        negatives = sample(
            self.label_pool - set(positives), 
            k=self.num_candidates - len(positives)
        )
        return set(chain(positives, negatives))
        
    def __len__(self) -> int:
        return len(self.input_dataset)

    def __getitem__(self, index:int) -> Tuple[Tensor]:
        # gather inputs and labels
        inputs = self.input_dataset[index]
        labels = self.labels[index]
        # generate candidates and build targets
        candidates = tuple(self.label_pool if self.sampling_disabled else \
            self.sample_candidates(index, labels))
        targets = [int(c in labels) for c in candidates]
        assert len(candidates) == self.num_candidates
        # convert to tensors
        candidates = torch.LongTensor(candidates)
        targets = torch.FloatTensor(targets)
        # return all features
        return {
            'candidates': candidates, 
            'labels': targets,
            **inputs
        }

@dataclass
class GroupWeights(object):
    """ Helper class managing the weights of label-groups """
    weights:torch.FloatTensor =None
    layout:torch.LongTensor =None
    mask:torch.BoolTensor =None
    
    def get_weights(self, 
        index:int, 
        group_ids:torch.LongTensor
    ) -> torch.FloatTensor:
        # find groups in layout
        mask = (self.layout[index, :].unsqueeze(0) == group_ids.unsqueeze(1))
        idx = torch.where(mask)[1]
        # build weight vector for given groups
        w = torch.zeros_like(group_ids, dtype=torch.float)
        w[mask.any(dim=-1)] = self.weights[index, idx]
        # return weight vector
        return w
        
class GroupWeightedMultiLabelDataset(MultiLabelDataset):
    """ Multi-label dataset where labels are organized into groups and
        groups are wheighted indicating how probable their members are
        for the corresponding inputs

        The candidate selection first chooses members from the groups containing
        positive labels and afterwards selects candidates from negative groups but
        prefering the probable groups
    """

    def __init__(self,
        input_dataset:Dataset,
        labels:List[Set[int]],
        label_pool:Set[int],
        num_candidates:int,
        groups:Dict[int, Set[int]],
        group_weights:GroupWeights
    ) -> None:
        # initialize multi-label dataset
        super(GroupWeightedMultiLabelDataset, self).__init__(
            input_dataset=input_dataset,
            labels=labels,
            label_pool=label_pool,
            num_candidates=num_candidates
        )
        # save group information
        self.groups = groups
        self.weights = group_weights
        self.group_membership = build_group_membership(groups)
        
    def sample_candidates(self, index:int, positives:Set[int]) -> Set[int]:
        # get the positive groups and sort them by their weights
        positive_groups = self.group_membership[list(positives)].unique()
        weights = self.weights.get_weights(index, positive_groups)
        positive_groups = positive_groups[weights.argsort(descending=True)]
        # build negative candidates from positive groups
        negatives = chain(*(
            self.groups[group_id.item()] - positives 
            for group_id in positive_groups)
        )
        # build candidates and check if num_candidates is already fulfilled
        candidates = tuple(chain(positives, negatives))
        candidates = candidates[:self.num_candidates]
        if len(candidates) == self.num_candidates:
            return candidates
        
        # get the negative groups and sort them by their weights
        negative_groups = torch.LongTensor([g for g in self.groups if g not in positive_groups])
        weights = self.weights.get_weights(index, negative_groups)
        negative_groups = negative_groups[weights.argsort(descending=True)]
        # build negative candidates from negative groups
        negatives = chain(*(
            self.groups[group_id.item()]
            for group_id in negative_groups
        ))
        # add negatives to candidates
        candidates = tuple(chain(candidates, negatives))
        candidates = candidates[:self.num_candidates]
        # at this point the num_candidates condition must be fulfilled
        # as the each possible label was considered as either part of a
        # positive group or a negative group
        assert len(candidates) == self.num_candidates
        # return the candidates
        return candidates
