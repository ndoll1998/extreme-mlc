import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from treelib import Tree
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Set, Callable
from .plt import ProbabilisticLabelTree
from .tree_utils import (
    propagate_labels_to_level,
    convert_labels_to_ids
)
from .dataset import (
    MultiLabelDataset, 
    GroupWeights,
    GroupWeightedMultiLabelDataset
)

@dataclass(frozen=True)
class InputsAndLabels(object):
    """ Helper class storing an input dataset together with the corresponding labels """
    inputs: Dataset
    labels: List[Set[str]]

    def __post_init__(self):
        # make sure the inputs and labels align
        n, m = len(self.inputs), len(self.labels)
        assert n == m, "Inputs (%i) and Labels (%i) do not align!" % (n, m)

class End2EndTrainerModule(pl.LightningModule):
    """ End-to-End Trainer for probabilistic label tree
        Trains all levels of the tree simultaneously
    """

    def __init__(self,
        tree:Tree,
        model:ProbabilisticLabelTree,
        train_data:InputsAndLabels,
        val_data:InputsAndLabels,
        num_candidates:int,
        topk:int,
        train_batch_size:int,
        val_batch_size:int,
        metrics:Callable
    ) -> None:
        # initialize lightning module
        super().__init__()
        # save arguments
        self.tree = tree
        self.metrics = metrics
        self.k = topk
        # batch sizes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        # save model
        self.model = model
        
        # create datasets
        self.train_dataset = MultiLabelDataset(
            input_dataset=train_data.inputs,
            labels=train_data.labels,
            label_pool=label_pool,
            num_candidates=num_candidates
        )        
        self.val_dataset = MultiLabelDataset(
            input_dataset=val_data.inputs,
            labels=val_data.labels,
            label_pool=label_pool,
            num_candidates=num_candidates
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_dataloader(self) -> DataLoader:
        # build the train dataloader
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        # build the validation dataloader
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4)

    def training_step(self, batch, batch_idx):
        # pop labels from batch and predict
        labels = batch.pop('labels')
        out = self.model(**batch)
        # compute loss and log it
        loss = F.binary_cross_entropy(out.probs[out.mask], labels[out.mask])
        self.log("train_loss", loss, on_step=True, prog_bar=False, logger=True)
        # return loss
        return loss

    @torch.no_grad() 
    def validation_step(self, batch, batch_idx):
        # compute validation loss
        # pop labels from batch and predict
        labels = batch.pop('labels')
        out = self.model(**batch)
        # compute loss
        loss = F.binary_cross_entropy(out.probs[out.mask], labels[out.mask])
        
        # compute validation metrics
        candidates = batch.pop("candidates")
        positives = torch.masked_fill(candidates, labels == 0, -1)
        # predict using all previous layers
        # note that this is no longer candidate based but
        # instead the model chooses the paths to follow during prediction
        self.model.eval()
        output = self.model(**batch, topk=self.k)
        preds = output.topk(k=100).candidates
        # return loss, predictions and labels
        return {
            'loss': loss, 
            'preds': preds.cpu(), 
            'targets': positives.cpu()
        }

    def validation_epoch_end(self, outputs):
        # compute average loss
        avg_loss = sum((out['loss'] for out in outputs)) / len(outputs)    
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # concatenate predictions and targets
        preds = torch.cat(tuple(out['preds'] for out in outputs), dim=0)
        targets = torch.cat(tuple(out['targets'] for out in outputs), dim=0)
        # compute metrics
        log_metrics, add_metrics = self.metrics(preds, targets)
        # log metrics
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(add_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return avg_loss


class LevelTrainerModule(pl.LightningModule):
    """ Trainer class to train a single level of a probabilistic label tree
        Note that the trainer assumes that all previous levels of the PLT are
        already trained
    """

    def __init__(self,
        level:int,
        tree:Tree,
        model:ProbabilisticLabelTree,
        train_data:InputsAndLabels,
        val_data:InputsAndLabels,
        num_candidates:int,
        topk:int,
        train_batch_size:int,
        val_batch_size:int,
        metrics:Callable
    ) -> None:
        # initialize lightning module
        super().__init__()
        # save arguments
        self.level = level
        self.tree = tree
        self.metrics = metrics
        self.num_candidates = num_candidates
        self.k = topk
        # batch sizes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        # save model and get classifier for specified level
        self.model = model
        self.classifier = model.get_classifier(level=level)
        # save data
        self.train_data = train_data
        self.val_data = val_data

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

    def train_dataloader(self) -> DataLoader:
        # build the dataset and the dataloader from it
        dataset = self.build_dataset(self.train_data)
        return DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        # build the dataset and the dataloader from it
        dataset = self.build_dataset(self.val_data)
        return DataLoader(dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4)

    def training_step(self, batch, batch_idx):
        # pop labels from batch and predict
        labels = batch.pop('labels')
        logits = self.classifier(**batch)
        # compute loss and log it
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=False, logger=True)
        # return loss
        return loss

    @torch.no_grad() 
    def validation_step(self, batch, batch_idx):
        # compute validation loss
        # pop labels from batch and predict
        labels = batch.pop('labels')
        logits = self.classifier(**batch)
        # compute loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # compute validation metrics
        candidates = batch.pop("candidates")
        positives = torch.masked_fill(candidates, labels == 0, -1)
        # predict using all previous layers
        # note that this is no longer candidate based but
        # instead the model chooses the paths to follow during prediction
        self.model.eval()
        output = self.model(**batch, topk=self.k, restrict_depth=self.level+1)
        preds = output.topk(k=100).candidates
        # return loss, predictions and labels
        return {
            'loss': loss, 
            'preds': preds.cpu(), 
            'targets': positives.cpu()
        }

    def validation_epoch_end(self, outputs):
        # compute average loss
        avg_loss = sum((out['loss'] for out in outputs)) / len(outputs)    
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # concatenate predictions and targets
        preds = torch.cat(tuple(out['preds'] for out in outputs), dim=0)
        targets = torch.cat(tuple(out['targets'] for out in outputs), dim=0)
        # compute metrics
        log_metrics, add_metrics = self.metrics(preds, targets)
        # log metrics
        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(add_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return avg_loss

    def build_dataset(self, data:InputsAndLabels) -> MultiLabelDataset:
        
        input_dataset, labels = data.inputs, data.labels
        # build a list of all labels in the current level
        label_pool = set(n.data.level_index for n in self.tree.filter_nodes(
            lambda n: self.tree.level(n.identifier) == (self.level + 1)
        ))
        
        # build training labels for current level
        labels = propagate_labels_to_level(self.tree, labels, level=self.level)
        labels = convert_labels_to_ids(self.tree, labels)
        # root level has no weighting
        # also if all labels are candidates then group weighting is unnecessary
        if (self.level == 0) or (self.num_candidates >= len(label_pool)):
            return MultiLabelDataset(
                input_dataset=input_dataset,
                labels=labels,
                label_pool=label_pool,
                num_candidates=self.num_candidates
            )
        else:
            # create dataloader for train input dataset
            # note that the validation-dataloader does not shuffle the dataset
            loader = DataLoader(input_dataset, batch_size=self.val_batch_size, shuffle=False)
            # get model predictions
            self.model.eval()
            outputs = []
            # move model to device
            with torch.no_grad():
                for inputs in tqdm(loader, "Building Group Weights"):
                    # apply model and collect all outputs
                    inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
                    model_out = self.model(**inputs, topk=self.k, restrict_depth=self.level+1)
                    outputs.append(model_out.topk(k=self.k).cpu())
            # concatenate all model outputs to build group weights
            weights = GroupWeights(
                weights=torch.cat([out.probs for out in outputs], dim=0),
                layout=torch.cat([out.candidates for out in outputs], dim=0),
                mask=torch.cat([out.mask for out in outputs], dim=0),
            )
            # build group mapping
            groups = {
                # map each group to their members
                # i.e. each node in the current level to it's children
                group.data.level_index: set(
                    n.data.level_index 
                    for n in self.tree.children(group.identifier)
                )
                for group in self.tree.filter_nodes(
                    lambda n: self.tree.level(n.identifier) == self.level
                )
            }
            # create train dataset
            return GroupWeightedMultiLabelDataset(
                input_dataset=input_dataset,
                labels=labels,
                label_pool=label_pool,
                num_candidates=self.num_candidates,
                groups=groups,
                group_weights=weights
            )
