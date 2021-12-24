import torch
from .utils import build_sparse_tensor
from itertools import chain
from typing import Any, Dict, List, Tuple

def sparsify_predictions_and_targets(
    preds:torch.LongTensor,
    targets:torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Create sparse n-hot tensor from predictions and labels """
    # get the maximum label
    num_labels = max(
        preds.max().item(), 
        targets.max().item()
    ) + 1
    # build sparse targets
    sparse_targets = build_sparse_tensor(
        args=targets,
        mask=(targets >= 0),
        size=(targets.size(0), num_labels)
    )
    # build sparse predictions
    sparse_preds = build_sparse_tensor(
        args=preds,
        mask=(preds >= 0),
        size=sparse_targets.size()
    )
    # return both sparse predictions and targets
    return sparse_preds, sparse_targets

def precision(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    """ Computes the example-based Precision @ k """
    # build sparse targets and top-k predictions
    sparse_preds, sparse_targets = sparsify_predictions_and_targets(preds[:, :k], targets)
    # compute precision
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    return hits.item() / (preds.size(0) * k)

def recall(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    """ Computes the example-based Recall @ k """
    # build sparse targets and top-k predictions
    sparse_preds, sparse_targets = sparsify_predictions_and_targets(preds[:, :k], targets)
    # handle sparse.sum error when input is all zeros
    x = sparse_preds * sparse_targets
    if x._nnz() == 0:
        return 0.0
    # compute recall for each example
    hits = torch.sparse.sum(x, dim=-1).to_dense()
    recalls = hits / torch.sparse.sum(sparse_targets, dim=-1).to_dense()
    # return average recall
    return recalls.mean().item()

def f1_score(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    # build sparse targets and top-k predictions
    sparse_preds, sparse_targets = sparsify_predictions_and_targets(preds[:, :k], targets)
    # handle sparse.sum error when input is all zeros
    x = sparse_preds * sparse_targets
    if x._nnz() == 0:
        return 0.0
    # compute hits
    hits = torch.sparse.sum(x, dim=-1).to_dense()
    # compute precision
    precision = hits.sum().item() / (preds.size(0) * k)
    # compute recall
    recalls = hits / torch.sparse.sum(sparse_targets, dim=-1).to_dense()
    recall = recalls.mean().item()
    # compute f1-score
    return 2 * (precision * recall) / (precision + recall)    

def ndcg(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    """ Normalized Discounted Cumulative Gain """
    # get the top-k predictions
    preds = preds[:, :k]
    # here only the sparse targets are needed as the 
    # log-scaled predictions are generated later
    _, sparse_targets = sparsify_predictions_and_targets(preds, targets)
    # compute the inverse logs for each prediction
    inv_logs = 1 / torch.log2(torch.arange(k).float() + 2)
    inv_logs = inv_logs.unsqueeze(0).repeat(preds.size(0), 1)
    # build the sparse scaled prediction tensor
    sparse_scaled_preds = build_sparse_tensor(
        args=preds,
        mask=(preds >= 0),
        size=sparse_targets.size(),
        values=inv_logs    
    )
    # handle sparse.sum error when input is all zeros
    x = sparse_scaled_preds * sparse_targets
    if x._nnz() == 0:
        return 0.0
    # compute dcg
    dcg = torch.sparse.sum(x, dim=-1).to_dense()
    # compute normalization factor
    idx = torch.arange(k).unsqueeze(0).repeat(preds.size(0), 1)
    yn = torch.sparse.sum(sparse_targets, dim=-1).to_dense()
    target_inv_logs = torch.where(idx < yn.unsqueeze(1), inv_logs, torch.zeros_like(inv_logs))
    # normalize the gains
    ndcg = dcg / target_inv_logs.sum(dim=-1)
    # return average normalized gains
    return ndcg.mean().item()

def coverage(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    """ Computes Coverage @ k. How many of the targets are covered by the top-k predictions """
    # build sparse targets and top-k predictions
    sparse_preds, sparse_targets = sparsify_predictions_and_targets(preds[:, :k], targets)
    # compute coverage
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    return hits.item() / sparse_targets._nnz()

def hits(
    preds:torch.LongTensor,
    targets:torch.LongTensor,
    k:int
) -> float:
    """ Computes Hits @ k. How many of the top-k predictions are actually correct """
    # build sparse targets and top-k predictions
    sparse_preds, sparse_targets = sparsify_predictions_and_targets(preds[:, :k], targets)
    # compute hits
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    ks = torch.full((sparse_targets.size(0),), fill_value=k, dtype=torch.long)
    best = torch.sparse.sum(sparse_targets, dim=-1).to_dense()
    best = torch.minimum(best, ks)
    return hits.item() / best.sum().item()
