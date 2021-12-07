import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
from treelib import Tree
from .metrics import MetricsTracker
from .dataset import MultiLabelDataset, GroupWeights, GroupWeightedMultiLabelDataset
from .plt import ProbabilisticLabelTree
from .tree_utils import propagate_labels_to_level, convert_labels_to_ids
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import List, Tuple, Set, Dict, Callable

@dataclass
class TrainingArgs(object):
    """ Helper class storing all kids of arguments that specify the training 
        regime
    """

    # saving
    save_dir:str
    save_interval:int
    # evaluation
    eval_interval:int
    # batch sizes
    train_batch_size:int
    eval_batch_size:int =None
    # which torch device to use
    device:torch.device =None
    # training loop
    num_epochs:int =None
    num_steps:int =None
    # optimization
    learning_rate:float =1e-3
    max_grad_norm:float =None
 
    def __post_init__(self):

        # get the device to use
        default_device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.device = self.device if self.device is not None else default_device
        # use same batch size for evaluation if not explicitely specified
        self.eval_batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.train_batch_size
        # make sure either `num_steps` or `num_epochs` is set but not both or none
        assert (self.num_steps is None) or (self.num_epochs is None)
        assert (self.num_steps is not None) or (self.num_epochs is not None)


@dataclass(frozen=True)
class InputsAndLabels(object):
    """ Helper class storing an input dataset together with the corresponding labels """
    inputs: Dataset
    labels: List[Set[str]]

    def __post_init__(self):
        # make sure the inputs and labels align
        n, m = len(self.inputs), len(self.labels)
        assert n == m, "Inputs (%i) and Labels (%i) do not align!" % (n, m)

class __Trainer(object):
    """ Abstract Trainer class defining the very basic training loop """

    def __init__(self,
        args:TrainingArgs,
        model:nn.Module,
        optim:Optimizer,
        train_dataset:Dataset,
        eval_dataset:Dataset,
        metrics:MetricsTracker
    ) -> None:
        # save training arguments
        self.args = args
        # save model and optimizer
        self.model = model.to(args.device)
        self.optim = optim
        # build train and test datalaoders
        self.train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
        # save metrics tracker
        self.metrics = metrics

    def loss(self, model:nn.Module, inputs:Dict[str, Tensor]) -> Tensor:
        """ Compute the loss for the given input """
        raise NotImplementedError()

    def predict(self, model:nn.Module, inputs:Dict[str, Tensor]) -> Tuple[Tensor]:
        """ Do a model prediction step and return the 3-tuple (loss, predictions, targets) """
        raise NotImplementedError()

    def train(self):
        """ Run the train loop """    
        
        self.model.train()
        global_step = 0
        running_loss = 0
        # compute the total number of training steps
        total_steps = self.args.num_steps if self.args.num_steps is not None else \
            (self.args.num_epochs * len(self.train_loader))

        # training loop
        with tqdm(total=total_steps, desc="Training") as pbar:

            while (global_step < total_steps):

                # start new training epoch
                for inputs in self.train_loader:
                    # compute loss
                    global_step += 1
                    loss = self.loss(self.model, inputs)
                    # optimize
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    # update running loss
                    running_loss += loss.item()

                    # evaluate the model
                    if (global_step % self.args.eval_interval == 0):
                        self.model.eval()
                        eval_running_loss = 0
                        predictions, targets = [], []
                        
                        with torch.no_grad():
                            # evaluation loop
                            for inputs in tqdm(self.eval_loader, desc="Evaluating", leave=False):
                                # do prediction step and store returns
                                loss, preds, positives = self.predict(self.model, inputs)
                                eval_running_loss += loss.item()
                                predictions.append(preds.cpu())
                                targets.append(positives.cpu())

                            # concatenate all predictions and targets
                            preds = torch.cat(predictions, dim=0)
                            targets = torch.cat(targets, dim=0)
                            # compute metrics
                            metrics_log = self.metrics(
                                step=global_step, 
                                train_loss=running_loss/self.args.eval_interval,
                                eval_loss=eval_running_loss/len(self.eval_loader),
                                predictions=preds, 
                                targets=targets
                            )
                            metrics_log = {
                                key: round(value, 5 if 'loss' in key else 3)
                                for key, value in metrics_log.items()
                            }
                            print("Step %i:" % global_step, metrics_log)

                            # reset running loss
                            running_loss = 0

                        # back to training
                        self.model.train()

                    if (global_step % self.args.save_interval == 0):
                        # save model and optimizer
                        os.makedirs(self.args.save_dir, exist_ok=True)
                        torch.save(self.optim.state_dict(), os.path.join(self.args.save_dir, "optimizer.bin"))
                        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model.bin"))

                    # update progress bar
                    pbar.update(1)
                    
                    # training done
                    if (global_step == total_steps):
                        break


class EndToEndTrainer(__Trainer):
    """ End-to-End Trainer for probabilistic label tree
        Trains all levels of the tree simultaneously
    """

    def __init__(self,
        tree:Tree,
        model:ProbabilisticLabelTree,
        train_data:InputsAndLabels,
        eval_data:InputsAndLabels,
        num_candidates:int,
        metrics:MetricsTracker,
        args:TrainingArgs,
        topk:int
    ):
        # save arguments as they are needed in create_optimizer
        self.args = args
        self.k = topk
        # build a list of all labels organized in the label tree
        label_pool = set(n.data.level_index for n in tree.leaves()) 
        # convert the labels to ids
        train_labels = convert_labels_to_ids(tree, train_data.labels)
        eval_labels = convert_labels_to_ids(tree, eval_data.labels)

        # create datasets
        train_dataset = MultiLabelDataset(
            input_dataset=train_data.inputs,
            labels=train_labels,
            label_pool=label_pool,
            num_candidates=num_candidates
        )        
        eval_dataset = MultiLabelDataset(
            input_dataset=eval_data.inputs,
            labels=eval_labels,
            label_pool=label_pool,
            num_candidates=num_candidates
        )

        # initialize trainer
        super(EndToEndTrainer, self).__init__(
            args=args,
            model=model,
            optim=self.create_optimizer(model),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metrics=metrics
        )


    def create_optimizer(self, model:nn.Module) -> Optimizer:
        return Adam(model.parameters(), lr=self.args.learning_rate)
    
    def loss(self, model, inputs):
        # move inputs to device and pop labels from inputs
        inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
        labels = inputs.pop('labels')
        # predict and compute loss
        out = model(**inputs)
        return F.binary_cross_entropy(out.probs[out.mask], labels[out.mask])

    @torch.no_grad()
    def predict(self, model, inputs):
        # move all inputs to device
        inputs = {n: t.to(self.args.device) for n, t in inputs.items()}
        # predict and compute loss
        labels = inputs.pop("labels")
        out = model.forward(**inputs)
        loss = F.binary_cross_entropy(out.probs[out.mask], labels[out.mask])

        # get all positives
        # TODO: this assumes that all positive labels are
        #       contained in the candidates which is not 
        #       neccessarily true
        candidates = inputs.pop("candidates")
        positives = torch.masked_fill(candidates, labels==0, -1)
        # predict using topk and track metrics
        out = model.forward(**inputs, topk=self.k)
        preds = out.topk(k=100).candidates
        # return
        return (loss, preds, positives)


class LevelTrainer(__Trainer):
    """ Trainer class to train a single level of a probabilistic label tree
        Note that the trainer assumes that all previous levels of the PLT are
        already trained
    """
    
    def __init__(self,
        level:int,
        tree:Tree,
        model:ProbabilisticLabelTree,
        train_data:InputsAndLabels,
        eval_data:InputsAndLabels,
        num_candidates:int,
        metrics:MetricsTracker,
        args:TrainingArgs,
        topk:int
    ) -> None:
        # save arguments as they are needed in build_dataset
        self.args = args
        # save tree and level to train
        self.tree = tree
        self.level = level
        # save candidate information
        self.num_candidates = num_candidates
        self.k = topk # used in evaluation
        
        # move the plt to the device and get the classifier to train
        self.plt = model.to(args.device)
        cls = self.plt.get_classifier(level=level)
        # create the optimizer
        optim = self.create_optimizer(cls)

        # initialize trainer
        super(LevelTrainer, self).__init__(
            args=args,
            # model and optimizer
            model=cls,
            optim=optim,
            # build training and evaluation dataset
            train_dataset=self.build_dataset(train_data),
            eval_dataset=self.build_dataset(eval_data),
            # metrics
            metrics=metrics
        )

    def create_optimizer(self, model:nn.Module) -> Optimizer:
        return Adam(model.parameters(), lr=self.args.learning_rate)

    def build_dataset(self, data:InputsAndLabels):
        
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
            # note that the test-dataloader does not shuffle the dataset
            loader = DataLoader(input_dataset, batch_size=self.args.eval_batch_size, shuffle=False)
            # get model predictions
            self.plt.eval()
            outputs = []
            # move model to device
            with torch.no_grad():
                for inputs in tqdm(loader, "Building Group Weights"):
                    # apply model and collect all outputs
                    inputs = {key: tensor.to(self.args.device) for key, tensor in inputs.items()}
                    model_out = self.plt(**inputs, topk=self.k, restrict_depth=self.level+1)
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
            
    def loss(self, classifier, inputs):
        # move inputs to device and pop labels from inputs
        inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
        labels = inputs.pop('labels')
        # predict and compute loss
        logits = classifier(**inputs)
        return F.binary_cross_entropy_with_logits(logits, labels)

    @torch.no_grad()
    def predict(self, classifier, inputs):
        # pop labels from inputs
        labels = inputs.pop('labels').to(self.args.device)
        # move inputs to device
        inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
        # predict and compute loss
        # note that this is still candidate based
        logits = classifier(**inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        # get all positives
        # TODO: this assumes that all positive labels are
        #       contained in the candidates which is not 
        #       neccessarily true
        candidates = inputs.pop("candidates")
        positives = torch.masked_fill(candidates, labels == 0, -1)
        # predict using all previous layers
        # note that this is no longer candidate based but
        # instead the model chooses the paths to follow during prediction
        self.plt.eval()
        output = self.plt(**inputs, topk=self.k, restrict_depth=self.level+1)
        preds = output.topk(k=100).candidates
        # return loss, predictions and labels
        return (loss, preds, positives)
