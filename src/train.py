import os
import yaml
import json
import torch
import pickle
import numpy as np
from treelib import Tree
from typing import Dict, Tuple, List, Any
# import attention-xml
from xmlc.dataset import NamedTensorDataset
from xmlc.plt import ProbabilisticLabelTree
from xmlc.utils import build_sparse_tensor
from xmlc.trainer import (
    EndToEndTrainer,
    LevelTrainer,
    TrainingArgs,
    InputsAndLabels
)
from xmlc.modules import (
    MLP,
    SoftmaxAttention,
    MultiHeadAttention
)
from xmlc.metrics import (
    MetricsTracker,
    PrecisionCoverageHits
)
from src.classifiers import LSTMClassifier


class LSTMClassifierFactory(object):
    
    def __init__(self, 
        params:Dict[str, Any],
        padding_idx:int,
        emb_init:np.ndarray,
    ) -> None:

        # build classifier keyword-arguments
        self.cls_kwargs = dict(
            hidden_size=params['encoder']['hidden_size'],
            num_layers=params['encoder']['num_layers'],
            emb_init=emb_init,
            padding_idx=vocab['[pad]'],
            dropout=params['dropout']
        )

        # get attention type
        self.attention_module = {
            'softmax-attention': SoftmaxAttention,
            'multi-head-attention': MultiHeadAttention
        }[params['attention']['type']]

        # get classifier type
        self.mlp_layers = [
            params['encoder']['hidden_size'] * 2, 
            *params['mlp']['hidden_layers'], 
            1
        ]
        self.mlp_kwargs = dict(
            bias=params['mlp']['bias'],
            act_fn={
                'relu': torch.relu
            }[params['mlp']['activation']]
        )

    def create(self, num_labels:int) -> ProbabilisticLabelTree:

        # create attention module
        attention = self.attention_module()
        # create multi-layer perceptron
        mlp = MLP(*self.mlp_layers, **self.mlp_kwargs)
        # create classifier
        return LSTMClassifier(
            num_labels=num_labels,
            **self.cls_kwargs,
            attention=attention,
            mlp=mlp
        )

    def __call__(self, num_labels:int) -> ProbabilisticLabelTree:
        return self.create(num_labels)


def load_data(
    input_ids_path:str, 
    labels_path:str, 
    padding_idx:int
) -> Tuple[InputsAndLabels, InputsAndLabels]:

    # load input ids and compute mask
    input_ids = torch.load(input_ids_path)
    input_mask = (input_ids != padding_idx)
    # load labels
    with open(labels_path, "r") as f:
        labels = [l.strip().split() for l in f.readlines()]
    # build data
    return InputsAndLabels(
        inputs=NamedTensorDataset(input_ids=input_ids, input_mask=input_mask),
        labels=labels
    )

def train_end2end(
    model:ProbabilisticLabelTree, 
    train_data:InputsAndLabels, 
    test_data:InputsAndLabels,
    params:Dict[str, Any]
):
    raise NotImplementedError()

def train_levelwise(
    tree:Tree,
    model:ProbabilisticLabelTree, 
    train_data:InputsAndLabels, 
    test_data:InputsAndLabels,
    params:Dict[str, Any],
    output_dir:str
) -> List[MetricsTracker]:
    # use gpu if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_metrics = []
    # train each level of the label-tree one after another
    for level in range(model.num_levels - 1):
        print("# \n# Training Level %i \n#" % level)
        # create metrics tracker instance
        metrics = PrecisionCoverageHits()
        
        # set training arguments
        args = TrainingArgs(
            # saving
            save_interval=params['save_interval'],
            save_dir=os.path.join(output_dir, "level-%i" % level),
            # evaluation
            eval_interval=params['eval_interval'],
            # batch sizes
            train_batch_size=params['train_batch_size'],
            eval_batch_size=params['eval_batch_size'],
            # pytorch device
            device=device,
            # training loop
            num_steps=params['num_steps'] 
        )
        # build trainer
        LevelTrainer(
            level=level,
            tree=tree,
            model=model,
            train_data=train_data,
            eval_data=test_data,
            num_candidates=params['num_candidates'],
            args=args,
            topk=params['topk'],
            metrics=metrics
        ).train()
        # all metrics tacker to list
        all_metrics.append(metrics)

    # return tracked metrics per level
    return all_metrics

if __name__ == '__main__':
   
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser("Train a model on the preprocessed data.")
    parser.add_argument("--train-input-ids", type=str, help="Path to the preprocessed train input ids.")
    parser.add_argument("--test-input-ids", type=str, help="Path to the preprocessed test input ids.")
    parser.add_argument("--train-labels", type=str, help="Path to the file holding the train labels.")
    parser.add_argument("--test-labels", type=str, help="Path to the file holding the test labels.")
    parser.add_argument("--vocab", type=str, help="Path to the vocab file.")
    parser.add_argument("--embed", type=str, help="Path to the initial (pretrained) embedding vector file.")
    parser.add_argument("--label-tree", type=str, help="Path to the label tree file.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    # parse arguments    
    args = parser.parse_args()
 
    # load model and trainer parameters
    with open("params.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load pretrained embedding
    with open(args.vocab, "r") as f:
        vocab = json.loads(f.read())
        padding_idx = vocab['[pad]']
    emb_init = np.load(args.embed)
    
    # create classifier factory
    cls_factory = LSTMClassifierFactory(
        params=params['model'],
        padding_idx=padding_idx,
        emb_init=emb_init
    )

    # load train and test data
    train_data = load_data(input_ids_path=args.train_input_ids, labels_path=args.train_labels, padding_idx=padding_idx)
    test_data = load_data(input_ids_path=args.test_input_ids, labels_path=args.test_labels, padding_idx=padding_idx)

    # load label tree
    with open(args.label_tree, "rb") as f:
        tree = pickle.load(f)

    # create the model
    model = ProbabilisticLabelTree(
        tree=tree,
        cls_factory=cls_factory
    ) 
    # count the number of parameters to optimize during training
    n_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print("#Trainable Parameters: %i" % n_trainable_params)

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
        test_data=test_data, 
        params=params['trainer'],
        output_dir=args.output_dir
    )

    # save model to disk
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.bin"))
    
    final_metrics = metrics[-1].final_metrics()
    # for levelwise training regime add metrics of intermediate levels
    for layer, m in enumerate(metrics[:-1]):
        final_metrics["layer-%i" % layer] = m.final_metrics()
        
    # save final metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w+") as f:
        f.write(json.dumps(final_metrics))

    # save metrics as csv
    metrics[-1].to_csv(os.path.join(args.output_dir, "metrics.csv"))
    
