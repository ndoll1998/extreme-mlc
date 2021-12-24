import os
import yaml
import json
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from treelib import Tree
from typing import Dict, Tuple, List, Any, Callable
# import attention-xml
from xmlc.trainer import (
    LevelTrainerModule,
    InputsAndLabels
)
from xmlc.metrics import *
from xmlc.plt import ProbabilisticLabelTree
from xmlc.dataset import NamedTensorDataset
from xmlc.utils import build_sparse_tensor
from src.classifiers import LSTMClassifierFactory

class MetricsTracker(object):

    def __init__(self) -> None:
        self.history = {}

    def compute_metrics(self,
        preds:torch.LongTensor, 
        targets:torch.LongTensor
    ):
        return {
            # first only the metrics that will be logged
            # in the progress bar
            "F3": f1_score(preds, targets, k=3),
            "nDCG3": ndcg(preds, targets, k=3),
        }, {
            # now all additional metrics that will be logged
            # to the logger of choice
            # precision @ k
            "P1": precision(preds, targets, k=1),
            "P3": precision(preds, targets, k=3),
            "P5": precision(preds, targets, k=5),
            # recall @ k
            "R1": recall(preds, targets, k=1),
            "R3": recall(preds, targets, k=3),
            "R5": recall(preds, targets, k=5),
            # f-score @ k
            "F1": f1_score(preds, targets, k=1),
            "F5": f1_score(preds, targets, k=5),
            # ndcg @ k
            "nDCG1": ndcg(preds, targets, k=1),
            "nDCG5": ndcg(preds, targets, k=5),
            # coverage @ k
            "C1": coverage(preds, targets, k=1),
            "C3": coverage(preds, targets, k=3),
            "C5": coverage(preds, targets, k=5),
            # hits @ k
            "H1": hits(preds, targets, k=1),
            "H3": hits(preds, targets, k=3),
            "H5": hits(preds, targets, k=5),
        }

    def __call__(self, 
        preds:torch.LongTensor, 
        targets:torch.LongTensor
    ):
        # compute all metrics
        log_metrics, add_metrics = self.compute_metrics(preds, targets)
        # add all to history
        for name, value in chain(log_metrics.items(), add_metrics.items()):
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
        # return
        return log_metrics, add_metrics
            

def load_data(
    data_path:str, 
    padding_idx:int
) -> Tuple[InputsAndLabels, InputsAndLabels]:
    # load input ids and compute mask
    data = torch.load(data_path)
    input_ids, labels = data['input-ids'], data['labels']
    input_mask = (input_ids != padding_idx)
    # build data
    return InputsAndLabels(
        inputs=NamedTensorDataset(input_ids=input_ids, input_mask=input_mask),
        labels=labels
    )

def train_end2end(
    model:ProbabilisticLabelTree, 
    train_data:InputsAndLabels, 
    val_data:InputsAndLabels,
    params:Dict[str, Any]
):
    raise NotImplementedError()


def train_levelwise(
    tree:Tree,
    model:ProbabilisticLabelTree, 
    train_data:InputsAndLabels, 
    val_data:InputsAndLabels,
    params:Dict[str, Any],
    output_dir:str
) -> MetricsTracker:
    # use gpu if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train each level of the label-tree one after another
    for level in range(model.num_levels - 1):

        print("-" * 50)
        print("-" * 17 + ("Training Level %i" % level) + "-" * 17)
        print("-" * 50)

        # create logger
        # logger = pl.loggers.TensorBoardLogger("logs", name="attention-xml", sub_dir="level-%i" % level)
        logger = None # pl.loggers.MLFlowLogger()
        # create metrics tacker object
        metrics = MetricsTracker()
        # create the trainer module
        trainer_module = LevelTrainerModule(
            level=level,
            tree=tree,
            model=model,
            train_data=train_data,
            val_data=val_data,
            num_candidates=params['num_candidates'],
            topk=params['topk'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['eval_batch_size'],
            metrics=metrics
        )
        # create the trainer
        trainer = pl.Trainer(
            gpus=1,
            auto_select_gpus=True,
            max_steps=params['num_steps'],
            val_check_interval=params['eval_interval'],
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
            callbacks=[
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor="F3",
                    patience=50,
                    mode="max",
                    verbose=False
                )
            ]
        )
        # train the model
        trainer.fit(trainer_module)

    # return metrics tracker instance of very last layer
    return metrics        

if __name__ == '__main__':
   
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser("Train a model on the preprocessed data.")
    parser.add_argument("--train-data", type=str, help="Path to the preprocessed train data.")
    parser.add_argument("--val-data", type=str, help="Path to the preprocessed validation data.")
    parser.add_argument("--vocab", type=str, help="Path to the vocab file.")
    parser.add_argument("--embed", type=str, help="Path to the initial (pretrained) embedding vector file.")
    parser.add_argument("--label-tree", type=str, help="Path to the label tree file.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    # parse arguments    
    args = parser.parse_args()
 
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load model and trainer parameters
    with open("params.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)
    
    # load pretrained embedding
    with open(args.vocab, "r") as f:
        vocab = json.loads(f.read())
        padding_idx = vocab['[pad]']
    emb_init = np.load(args.embed)
    
    # load label tree
    with open(args.label_tree, "rb") as f:
        tree = pickle.load(f)

    # create the model
    model = ProbabilisticLabelTree(
        tree=tree,
        cls_factory=LSTMClassifierFactory.from_params(
            params=params['model'], 
            padding_idx=padding_idx,
            emb_init=emb_init
        )
    ) 

    # load train and validation data
    train_data = load_data(data_path=args.train_data, padding_idx=padding_idx)
    val_data = load_data(data_path=args.val_data, padding_idx=padding_idx)
    
    # check which training regime to use
    training_regime = {
        "levelwise": train_levelwise,
        "end2end": train_end2end
    }[params['trainer']['regime']]
    
    # train model
    metrics = training_regime(
        tree=tree,
        model=model, 
        train_data=train_data, 
        val_data=val_data, 
        params=params['trainer'],
        output_dir=args.output_dir
    )

    # save model to disk
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.bin"))
    
    # save final metrics
    final_scores = {name: values[-1] for name, values in metrics.history.items()}
    with open(os.path.join(args.output_dir, "validation-scores.json"), "w+") as f:
        f.write(json.dumps(final_scores))

    # plot metrics
    fig, axes = plt.subplots(6, 1, figsize=(12, 24), sharex=True)
    n, m = len(metrics.history["P1"]), params['trainer']['eval_interval']
    xs = list(range(1, n * m + 1, m)) 
    for ax, name in zip(axes, ["nDCG", "P", "R", "F", "C", "H"]): 
        # plot ndcg
        ax.plot(xs, metrics.history['%s1' % name], label="$k=1$")
        ax.plot(xs, metrics.history['%s3' % name], label="$k=3$")
        ax.plot(xs, metrics.history['%s5' % name], label="$k=5$")
        ax.set(
            title="%s @ k" % name,
            ylabel=name,
            xlabel="Global Step"
        )
        ax.legend()
        ax.grid()
    # save and show
    fig.savefig(os.path.join(args.output_dir, "validation-metrics.pdf"))
