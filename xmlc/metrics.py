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
