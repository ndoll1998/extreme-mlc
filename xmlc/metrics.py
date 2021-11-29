import torch

def precision(
    preds:torch.FloatTensor, 
    targets:torch.LongTensor, 
    k:int
) -> float:
    """ Computes Precision @ k """
    idx = torch.arange(preds.size(0)).unsqueeze(-1)
    predictions = torch.zeros_like(targets)
    predictions[idx, preds[:, :k]] = 1
    # compute precision
    hits = (predictions * targets).sum().item()
    return hits / (preds.size(0) * k)

def coverage(
    preds:torch.FloatTensor, 
    targets:torch.LongTensor, 
    k:int
) -> float:
    """ Computes Coverage @ k. How many of the targets are covered by the top-k predictions """
    idx = torch.arange(preds.size(0)).unsqueeze(-1)
    predictions = torch.zeros_like(targets)
    predictions[idx, preds[:, :k]] = 1
    # compute precision
    hits = (predictions * targets).sum().item()
    return hits / targets.sum().item()

def hits(
    preds:torch.FloatTensor, 
    targets:torch.LongTensor, 
    k:int
) -> float:
    """ Computes Hits @ k. How many of the top-k predictions are actually correct """
    idx = torch.arange(preds.size(0)).unsqueeze(-1)
    predictions = torch.zeros_like(targets)
    predictions[idx, preds[:, :k]] = 1
    # compute precision
    hits = (predictions * targets).sum().item()
    ks = torch.full((targets.size(0),), fill_value=k, dtype=torch.long)
    return hits / torch.minimum(targets.sum(dim=1), ks).sum().item()
