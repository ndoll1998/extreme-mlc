import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from xmlc.utils import build_group_membership
from xmlc.tree_utils import yield_tree_levels
from treelib import Tree
from dataclasses import dataclass
from typing import Callable

@dataclass
class PLTOutput(object):
    probs:torch.FloatTensor =None
    candidates:torch.LongTensor =None
    mask:torch.BoolTensor =None

    def cpu(self) -> "PLTOutput":
        # move all to cpu
        self.probs = self.probs.cpu()
        self.candidates = self.candidates.cpu()
        self.mask = self.mask.cpu()
        return self

    def topk(self, k:int) -> "PLTOutput":
        b = self.probs.size(0)
        batch_idx = torch.arange(b).unsqueeze(1)
        # get the topk predicted candidates
        k_ = min(k, self.probs.size(-1))
        topk_probs, topk_arg = torch.topk(self.probs, k=k_, dim=-1)
        topk_candidates = self.candidates[batch_idx, topk_arg]
        topk_mask = self.mask[batch_idx, topk_arg]
        # mark invalids
        topk_candidates[~topk_mask] = -1
        # pad all to match k
        pad_shape = (b, k - k_)
        topk_probs = torch.cat((topk_probs, torch.zeros(pad_shape, device=topk_probs.device)), dim=1)
        topk_candidates = torch.cat((topk_candidates, -torch.ones(pad_shape, dtype=torch.long, device=topk_candidates.device)), dim=1)
        topk_mask = torch.cat((topk_mask, torch.zeros(pad_shape, dtype=torch.bool, device=topk_mask.device)), dim=1)
        # return topk output
        return PLTOutput(
            probs=topk_probs,
            candidates=topk_candidates,
            mask=topk_mask
        )

class ProbabilisticLabelTree(nn.Module):
    """ Probabilistic Label Tree """

    def __init__(
        self,
        tree:Tree,
        cls_factory:Callable[[int], nn.Module]
    ) -> None:
        # initialize module
        super(ProbabilisticLabelTree, self).__init__()
        # create a classifier per hierarchy
        self.classifiers = nn.ModuleList([
            cls_factory(num_labels)
            for i, num_labels in enumerate(
                map(len, yield_tree_levels(tree)
            )) if i > 0 # ignore root
        ])

        # build group-membership per level
        self.group_membership_per_level = nn.ParameterList([
            nn.Parameter(
                build_group_membership({
                    group.data.level_index: set(
                        member.data.level_index
                        for member in tree.children(group.identifier)
                    ) for group in level
                }), requires_grad=False
            )
            for i, level in enumerate(yield_tree_levels(tree, max_depth=tree.depth()))
            if i > 0 # ignore first layer as it has only one group
        ])
        # build level members
        self.level_members = nn.ParameterList([
            nn.Parameter(torch.LongTensor(
                [m.data.level_index for m in level]
            ), requires_grad=False)
            for i, level in enumerate(yield_tree_levels(tree))
            if i > 0 # ignore first layer as it has only one member (the root)
        ])

    def get_classifier(self, level:int) -> nn.Module:
        return self.classifiers[level]

    def num_labels(self, level:int=-1) -> int:
        return self.level_members[level].size(0)

    @property
    def num_levels(self) -> int:
        return len(self.level_members) + 1

    def forward(
        self, 
        *args, 
        topk:int=-1, 
        candidates:torch.LongTensor =None,
        candidates_mask:torch.BoolTensor =None,
        restrict_depth:int =None, 
        **kwargs
    ) -> PLTOutput:

        # validate restrict depth
        restrict_depth = self.num_levels if restrict_depth is None else restrict_depth
        assert 0 < restrict_depth < (self.num_levels + 1)

        # check prediction kind
        if candidates is not None:
            # create candidate mask if not provided
            candidates_mask = torch.ones_like(candidates, dtype=torch.bool) \
                    if candidates_mask is None else candidates_mask
            # candidate-based prediction
            return self.forward_candidates(
                *args, 
                candidates=candidates,
                candidates_mask=candidates_mask,
                restrict_depth=restrict_depth, 
                **kwargs
            )
            
        else:
            # topk-based prediction
            return self.forward_topk(
                *args, 
                topk=topk if topk > 0 else float('inf'), 
                restrict_depth=restrict_depth, 
                **kwargs
            )

    def forward_candidates(
        self,
        *args,
        candidates:torch.LongTensor,
        candidates_mask:torch.BoolTensor,
        restrict_depth:int,
        **kwargs
    ) -> PLTOutput:
        
        # build the paths for all candidates
        # by propagating them up the tree
        shape = (self.num_levels-1, *candidates.size())
        paths = torch.empty(shape, dtype=torch.long, device=candidates.device)
        paths[-1, ...] = candidates
        for i in range(self.num_levels-3, -1, -1):
            membership = self.group_membership_per_level[i]
            paths[i, ...] = membership[paths[i+1, ...]]

        probs = 1
        # predict
        for i, cls in enumerate(self.classifiers):
            # apply classifier to get logits for the candidates in the current level
            logits = cls(*args, **kwargs, candidates=paths[i, ...])
            logits = logits.masked_fill(~candidates_mask, -1e5)
            # update probabilities
            probs *= torch.sigmoid(logits)
            
        # return output
        return PLTOutput(probs=probs, candidates=candidates, mask=candidates_mask)
            
            
    def forward_topk(
        self,
        *args, 
        topk:int, 
        restrict_depth:int, 
        **kwargs
    ) -> PLTOutput:
        # predict very first layer
        logits = self.classifiers[0](*args, **kwargs)
        probs = torch.sigmoid(logits)
        # get the batch size
        b = logits.size(0)
        batch_idx = torch.arange(b).unsqueeze(1)
        # initial group pool
        group_pool = self.level_members[0].unsqueeze(0).repeat(b, 1)
        padding_mask = torch.zeros_like(group_pool).bool()

        # predict upcoming layers
        for members, group_membership, cls in zip(
            self.level_members[1:restrict_depth],
            self.group_membership_per_level[:restrict_depth-1],
            self.classifiers[1:restrict_depth]
        ):

            # select the top-k groups
            # TODO: do we want the accumulated probs here or the non-accumulated logits
            _, selected_group_idx = torch.topk(probs, k=min(topk, logits.size(-1)), dim=-1)
            selected_groups = group_pool[batch_idx, selected_group_idx]

            # build the candidate mask by choosing all group members from the selected groups
            candidate_mask = (group_membership[None, None, :] == selected_groups[:, :, None])
            candidate_mask = candidate_mask.any(dim=1)
            # map each candidate to the group it comes from
            candidate_group_membership = [group_membership[mask] for mask in candidate_mask]
            candidate_group_membership = pad_sequence(candidate_group_membership, batch_first=True, padding_value=-1)
            # gather candidates and pad them
            candidates = [members[mask] for mask in candidate_mask]
            candidates = pad_sequence(candidates, batch_first=True, padding_value=-1)
            # create mask for valid candidates
            padding_mask = (candidates == -1)
            candidates[padding_mask] = 0 # label-embedding cannot handle -1

            # apply classifier for current level and apply padding-mask
            logits = cls(*args, **kwargs, candidates=candidates)
            logits = logits.masked_fill(padding_mask, -1e5)

            # compute label probabilities for candidates
            probs = probs[batch_idx, candidate_group_membership]
            probs *= torch.sigmoid(logits)

            # the candidates are the new group-pools
            # note that since we apply the padding-mask to the logits
            # the probability for choosing invalid groups (due to padding)
            # in upcoming iterations is zero
            group_pool = candidates

        # return last level output
        return PLTOutput(probs=probs, candidates=group_pool, mask=~padding_mask)

