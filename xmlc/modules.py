import torch
import torch.nn as nn
import torch.nn.functional as F
from treelib import Tree
from torch import Tensor
from typing import Tuple, Callable
from .tree_utils import yield_tree_levels

class MLP(nn.Module):
    """ Multi-Layer Perceptron """

    def __init__(self,
        *layers:Tuple[int],
        act_fn:Callable[[Tensor], Tensor] =F.relu,
        bias:bool=True
    ) -> None:
        # initialize module
        super(MLP, self).__init__()
        # build all linear layers
        self.layers = nn.ModuleList([
            nn.Linear(n, m, bias=bias)
            for n, m in zip(layers[:-1], layers[1:])
        ])
        # initialize weights with xavier uniform
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        # save activation function
        self.act_fn = act_fn

    def forward(self, x:Tensor) -> Tensor:
        # apply all but the last layer
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))
        # apply last layer seperately to avoid activation function
        return self.layers[-1](x)


class SoftmaxAttention(nn.Module):
    """ Linear Softmax Attention Module used in original AttentionXML implementation """    

    def forward(self,
        x:torch.FloatTensor, 
        mask:torch.BoolTensor, 
        label_emb:torch.FloatTensor
    ) -> torch.FloatTensor:
        # compute attention scores
        scores = x @ label_emb.transpose(1, 2)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e5)
        scores = torch.softmax(scores, dim=-2)
        # compute label-aware embeddings
        return scores.transpose(1, 2) @ x

class MultiHeadAttention(nn.MultiheadAttention):
    """ Multi-Head Attention Module that can be used in a `LabelAttentionClassifier` Module """

    def forward(self,
        x:torch.FloatTensor, 
        mask:torch.BoolTensor, 
        label_emb:torch.FloatTensor
    ) -> torch.FloatTensor:
        # prepare inputs
        x = x.transpose(0, 1)
        label_emb = label_emb.transpose(0, 1)
        # apply multi-head attention
        attn_out, attn_weight = super(MultiHeadAttention, self).forward(
            query=label_emb,
            key=x,
            value=x,
            key_padding_mask=~mask
        )
        # reverse transpose
        return attn_out.transpose(0, 1)
 

class LabelAttentionClassifier(nn.Module):
    """ Label-attention based Multi-Label Classifier """

    def __init__(self,
        hidden_size:int,
        num_labels:int,
        attention:nn.Module,
        mlp:MLP
    ) -> None:
        # initialize module
        super(LabelAttentionClassifier, self).__init__()
        # save the attention and mlp module
        self.att = attention
        self.mlp = mlp
        # create label embedding
        self.label_embed = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=hidden_size,
            sparse=False
        )
        # use xavier uniform for initialization
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self,
        x:torch.FloatTensor,
        mask:torch.BoolTensor,
        candidates:torch.LongTensor =None
    ) -> torch.FloatTensor:
        # use all embeddings if no candidates are provided
        if candidates is None:
            n = self.label_embed.num_embeddings
            candidates = torch.arange(n).unsqueeze(0)
            candidates = candidates.repeat(x.size(0), 1)
            candidates = candidates.to(x.device)
        # get label embeddings and apply attention layer
        label_emb = self.label_embed(candidates)
        m = self.att(x, mask, label_emb)
        # apply classifier
        return self.mlp(m).squeeze(-1)



