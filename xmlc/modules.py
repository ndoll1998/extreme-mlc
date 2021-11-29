import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Callable

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
        # save activation function
        self.act_fn = act_fn

    def forward(self, x:Tensor) -> Tensor:
        # apply all but the last layer
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))
        # apply last layer seperately to avoid activation function
        return self.layers[-1](x)


class Attention(nn.Module):
    """ Linear Attention Module used in original AttentionXML implementation """    

    def __init__(self,
        dropout:float =0.2,
    ) -> None:
        # initialize module
        super(Attention, self).__init__()
        # save dropout
        self.dropout = dropout

    def forward(self,
        x:torch.FloatTensor, 
        mask:torch.BoolTensor, 
        label_emb:torch.FloatTensor
    ) -> torch.FloatTensor:
        # compute attention scores
        scores = x @ label_emb.transpose(1, 2)
        scores = torch.softmax(scores, dim=-1)
        scores = scores.masked_fill(~mask.unsqueeze(-1), 0)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        # compute label-aware embeddings
        return scores.transpose(1, 2) @ x


class MultiHeadAttention(nn.MultiheadAttention):
    """ Multi-Head Attention Module for PLT-Hierarchy Models """

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
 

class PLTHierarchy(nn.Module):
    """ Classifier Model for a Hierarchy of a Probability Label Tree """

    def __init__(self,
        hidden_size:int,
        num_labels:int,
        attention:nn.Module,
        classifier:nn.Module,
        dropout:float =0.2
    ) -> None:
        # initialize module
        super(PLTHierarchy, self).__init__()
        # save the attention and classifier module
        self.att = attention
        self.cls = classifier
        # create label embedding
        self.label_embed = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=hidden_size,
            sparse=False
        )
        # save dropout prob
        self.dropout = dropout

    def forward(self,
        x:torch.FloatTensor,
        mask:torch.BoolTensor,
        candidates:torch.LongTensor
    ) -> torch.FloatTensor:
        # get label embeddings and apply attention layer
        label_emb = self.label_embed(candidates)
        label_emb = F.dropout(label_emb, p=self.dropout, training=self.training)
        m = self.att(x, mask, label_emb)
        # apply classifier
        return self.cls(m).squeeze(-1)
