import torch
import torch.nn as nn
import torch.nn.functional as F
# import sentence transformer model
from sentence_transformers import SentenceTransformer
# import attention-xml modules
from xmlc.plt import ProbabilisticLabelTree
from xmlc.modules import ( 
    MLP,
    SoftmaxAttention,
    MultiHeadAttention,
    LabelAttentionClassifier
)

class SentenceTransformerClassifier(nn.Module):

    def __init__(self,
        num_labels:int,
        # sentence transformer
        pretrained_model:str,
        hidden_size:int,
        # attention module
        attention:nn.Module,
        # classifier module
        mlp:MLP
    ) -> None:
        # initialize module
        super(SentenceTransformerClassifier, self).__init__()
        # load sentence transformer model
        self.enc = SentenceTransformer(pretrained_model)
        # create the classifier 
        self.cls = LabelAttentionClassifier(
            hidden_size=hidden_size,
            num_labels=num_labels,
            attention=attention,
            mlp=mlp
        )

    def forward(self, input_ids, input_mask, candidates=None):
        # get the number of instances per bag
        n_batch, n_inst, _ = input_ids.size()
        # compute instance masks
        instance_mask = input_mask.sum(dim=-1) > 0
        # flatten ids and masks
        input_ids = input_ids.reshape(-1, input_ids.size(2))
        input_mask = input_mask.reshape(-1, input_mask.size(2))
        # apply encoder
        output = self.enc({'input_ids': input_ids, 'attention_mask': input_mask})
        embeds = output['sentence_embedding']
        # unflatten embeddings
        embeds = embeds.reshape(n_batch, n_inst, -1)
        # pass through classifier
        return self.cls(embeds, instance_mask, candidates)

class SentenceTransformerClassifierFactory(object):

    def __init__(self,
        # encoder
        pretrained_model:str,
        hidden_size:int,
        # attention module
        attention_type:str,
        # multi-layer perceptron
        mlp_hidden_layers:list,
        mlp_bias:bool,
        mlp_activation:str,
        # others
        padding_idx:int,
        dropout:float
    ):
        # build classifier kwargs
        self.cls_kwargs = dict(
            pretrained_model=pretrained_model,
            hidden_size=hidden_size
        )        

        # get attention type
        self.attention_module = {
            'softmax-attention': SoftmaxAttention,
            'multi-head-attention': MultiHeadAttention
        }[attention_type]

        # multi-layer perceptron setup
        self.mlp_layers = [hidden_size, *mlp_hidden_layers, 1]
        self.mlp_kwargs = dict(
            bias=mlp_bias,
            act_fn={
                'relu': torch.relu
            }[mlp_activation]
        )

    def create(self, num_labels:int):
        # create attention module
        attention = self.attention_module()
        # create multi-layer perceptron
        mlp = MLP(*self.mlp_layers, **self.mlp_kwargs)
        # create classifier
        return SentenceTransformerClassifier(
            num_labels=num_labels,
            **self.cls_kwargs,
            attention=attention,
            mlp=mlp
        )

    def __call__(self, num_labels:int):
        return self.create(num_labels)
