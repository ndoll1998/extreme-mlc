import torch
from transformers import EvalPrediction
from .utils import build_sparse_tensor
from itertools import chain
from typing import Any, Dict, List

class MetricsTracker(object):
    
    def __init__(self):
        # save dict holding all metrics for all evaluations
        self.metrics = {'steps': [], 'loss': [], 'eval_loss': []}
    
    def prepare(self, predictions, targets) -> Any:
        """ prepare predictions and targets for metric computations """
        return eval_preds
        
    def compute_log_metrics(self, *args):
        """ Compute metrics that will be logged during trainig """
        return {}
    
    def compute_additional_metrics(self, *args):
        """ Compute additional metrics that won't be logged """
        return {}
    
    def evaluate(self, 
        step:int,
        train_loss:float,
        eval_loss:float,
        predictions:torch.Tensor,
        targets:torch.Tensor
    ) -> Dict[str, float]:
        # update default metrics
        self.metrics['steps'].append(step)
        self.metrics['loss'].append(train_loss)
        self.metrics['eval_loss'].append(eval_loss)
        # prepare
        prepared = self.prepare(predictions, targets)
        # compute metrics
        log_metrics = self.compute_log_metrics(*prepared)
        add_metrics = self.compute_additional_metrics(*prepared)
        # add to lists
        for key, value in chain(log_metrics.items(), add_metrics.items()):
            # add metric to dict if not already done yet
            if key not in self.metrics:
                self.metrics[key] = []
            # add metric value to list
            self.metrics[key].append(value)
        # return metrics that will be logged
        return {'loss': train_loss, 'eval_loss': eval_loss, **log_metrics}
    
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def __getitem__(self, key) -> List[Any]:
        return self.metrics[key]

    def __getattr__(self, name):
        if name in self.metrics:
            return self.metrics[name]


def precision(
    preds:torch.LongTensor,
    sparse_targets:torch.Tensor,
    k:int
) -> float:
    """ Computes Precision @ k """
    # build sparse topk predictions
    preds = preds[:, :k]
    sparse_preds = build_sparse_tensor(
        args=preds,
        mask=(preds >= 0),
        size=sparse_targets.size()
    )
    # compute precision
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    return hits.item() / (preds.size(0) * k)

def coverage(
    preds:torch.LongTensor,
    sparse_targets:torch.Tensor,
    k:int
) -> float:
    """ Computes Coverage @ k. How many of the targets are covered by the top-k predictions """
    # build sparse topk predictions
    preds = preds[:, :k]
    sparse_preds = build_sparse_tensor(
        args=preds,
        mask=(preds >= 0),
        size=sparse_targets.size()
    )
    # compute coverage
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    return hits.item() / sparse_targets._nnz()

def hits(
    preds:torch.LongTensor,
    sparse_targets:torch.Tensor,
    k:int
) -> float:
    """ Computes Hits @ k. How many of the top-k predictions are actually correct """
    # build sparse topk predictions
    preds = preds[:, :k]
    sparse_preds = build_sparse_tensor(
        args=preds,
        mask=(preds >= 0),
        size=sparse_targets.size()
    )
    # compute hits
    hits = torch.sparse.sum(sparse_preds * sparse_targets)
    ks = torch.full((sparse_targets.size(0),), fill_value=k, dtype=torch.long)
    best = torch.sparse.sum(sparse_targets, dim=-1).to_dense()
    best = torch.minimum(best, ks)
    return hits.item() / best.sum().item()

