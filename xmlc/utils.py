import torch
from typing import Dict, Tuple, Set

def build_sparse_tensor(
    args:torch.LongTensor,
    mask:torch.BoolTensor,
    size:Tuple[int]
) -> torch.Tensor:
    # make sure sizes align
    assert size[0] == args.size(0)
    assert len(size) == args.dim() == 2
    # build sparse tensor
    idx = torch.arange(size[0]).unsqueeze(1).repeat(1, args.size(1))
    return torch.sparse_coo_tensor(
        indices=torch.stack((idx[mask], args[mask]), dim=0),
        values=[1] * mask.sum(),
        size=size
    )

def build_group_membership(groups:Dict[int, Set[int]]) -> torch.LongTensor:
    # create long tensor representing the membership
    # i.e. get the group of a specific member by index
    total_num_members = sum(map(len, groups.values()))
    membership = torch.empty((total_num_members,), dtype=torch.long)
    # fill the tensor with the membership information
    for group, members in groups.items():
        membership[list(members)] = group
    # return membership tensor
    return membership
