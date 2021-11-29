import torch
from torch import Tensor
from treelib import Tree
from random import sample
from torch.utils.data import Dataset
from itertools import chain, groupby
from typing import List, Tuple, Set
from .tree_utils import propagate_labels_upwards, convert_labels_to_ids

class __LabelTreeDataset(Dataset):
    """ Abstract Label Tree Dataset 
        Child Classes have to overwrite the `sample_candidates` function.
    """

    def __init__(self,
        input_dataset:Dataset,
        tree:Tree,
        labels:List[Set[str]],
        num_candidates:int
    ) -> None:
        # save arguments
        self.input_dataset = input_dataset
        self.num_candidates = num_candidates
        self.labels = [set(l) for l in labels]
        self.tree = tree
        # get the depth of the tree once here
        # this is incredibly computational heavy
        self.path_len = tree.depth() - 1 # ignore root in paths
        # some quick tests
        n, m = len(self.labels), len(self.input_dataset)
        assert n == m, "Labels (%i) and Input Dataset (%i) do not align!" % (n, m)
        
    def sample_candidates(self, positives:Set[str]) -> Set[str]:
        raise NotImplementedError()
        
    def __len__(self) -> int:
        return len(self.input_dataset)
    
    def __getitem__(self, index:int) -> Tuple[Tensor]:
        
        # gather inputs and labels
        inputs = self.input_dataset[index]
        labels = self.labels[index]

        # generate candidates
        candidates = self.sample_candidates(labels)
        assert len(candidates) == self.num_candidates
        
        # build target label tensor
        targets = [int(c in labels) for c in candidates]
        labels = torch.FloatTensor(targets)
        
        # for each candidate build the path to the root node
        # the lowest level in each path corresponds to the candidate
        paths = torch.zeros((self.path_len + 1, self.num_candidates), dtype=torch.long)
        paths[-1, :] = torch.LongTensor(convert_labels_to_ids(self.tree, candidates))
        # build paths to root
        for i in range(self.path_len-1, -1, -1):
            candidates = propagate_labels_upwards(self.tree, candidates)
            paths[i, :] = torch.LongTensor(convert_labels_to_ids(self.tree, candidates))
        # return all features
        return inputs + (paths, labels)

    
class LabelTreePureRandomDataset(__LabelTreeDataset):
    """ Label Tree Pure Random Dataset
        Candidates are sampled arbitrary of their group and relation to positive
        candidates, e.g. draw n candidates from a label pool containing all negative labels
    """
    
    def __init__(self,
        input_dataset:Dataset,
        tree:Tree,
        labels:List[Set[str]],
        num_candidates:int,
    ) -> None:
        # initialize base dataset
        super(LabelTreePureRandomDataset, self).__init__(
            input_dataset=input_dataset,
            tree=tree,
            labels=labels,
            num_candidates=num_candidates
        )
        # get a flat list of all candidates
        self.label_pool = set((n.identifier for n in tree.leaves()))
    
    def sample_candidates(self, positives:Set[str]) -> Set[str]:
        
        # at most half of the labels are positives
        n_positives = min(len(positives), self.num_candidates//2)
        positives = sample(tuple(positives), k=n_positives)
        # the rest are negative samples
        n_negatives = self.num_candidates - n_positives
        negative_pool = tuple(filter(lambda l: l not in positives, self.label_pool))
        negatives = sample(negative_pool, k=n_negatives)
        # concatenate candidates and build labels
        return list(chain(positives, negatives))


class LabelTreeGroupBasedDataset(__LabelTreeDataset):
    """ Label Tree Group-based Dataset
        Sample almost all candidates from the groups containing the positives
        Add some random candidates to increase model robustness
    """
    def __init__(self,
        input_dataset:Dataset,
        tree:Tree,
        labels:List[Set[str]],
        num_candidates:int,
    ) -> None:
        # initialize base dataset
        super(LabelTreeGroupBasedDataset, self).__init__(
            input_dataset=input_dataset,
            tree=tree,
            labels=labels,
            num_candidates=num_candidates
        )
        # get a flat list of all candidates
        self.label_pool = set((n.identifier for n in tree.leaves()))
        # build all groups for easier access later
        self.label_groups = {
            group.identifier: set((n.identifier for n in self.tree.children(group.identifier)))
            for group in self.tree.filter_nodes(lambda n: not n.is_leaf(self.tree.identifier))
        }
    
    def sample_candidates(self, positives:Set[str]) -> Set[str]:
        
        # flat all positives
        grouped_labels = {
            group: set(members)
            for group, members in groupby(
                iterable=positives, 
                key=lambda i: self.tree.parent(i).identifier
            )
        }
        
        # some candidates are sampled completely arbitrary
        num_random_candidates = max(1, int(0.05 * self.num_candidates))
        num_group_candidates = self.num_candidates - num_random_candidates
        # compute number of candidates per group
        num_groups = len(grouped_labels)
        num_candidates_per_group = [num_group_candidates // num_groups] * num_groups
        num_candidates_per_group[0] = (num_group_candidates - sum(num_candidates_per_group[1:]))
        assert sum(num_candidates_per_group) == num_group_candidates
        
        # cut the positive samples to guarantee that 
        # they take up at most half the candidates per group
        grouped_positive_candidates = {
            g: sample(
                tuple(labels), 
                k=min(len(labels), n // 2)
            ) for n, (g, labels) in zip(
                num_candidates_per_group,
                grouped_labels.items()
            )
        }
        # generate negative samples per group
        # note that we filter out all positives to make
        # sure we end up with negative samples
        grouped_negative_candidates = {
            g: sample(
                tuple(self.label_groups[g] - positives),
                k=n - len(labels)
            ) for n, (g, labels) in zip(
                num_candidates_per_group,
                grouped_positive_candidates.items()
            )
        }
        
        # concatenate labels and samples to build full candidate list
        candidates = set(chain(
            *grouped_positive_candidates.values(),
            *grouped_negative_candidates.values()
        ))
        
        # add some random samples
        random_negative_candidates = sample(
            tuple(self.label_pool - positives - candidates),
            k=num_random_candidates
        )
        # return all candidates
        return list(chain(candidates, random_negative_candidates))
