import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import attention-xml modules
from xmlc.plt import ProbabilisticLabelTree
from xmlc.modules import ( 
    MLP,
    SoftmaxAttention,
    MultiHeadAttention,
    LabelAttentionClassifier
)

class LSTMEncoder(nn.Module):
    """ Basic LSTM Encoder """
    
    def __init__(self,
        embed_size:int,
        hidden_size:int,
        num_layers:int,
        vocab_size:int,
        padding_idx:int,
        emb_init:torch.FloatTensor =None,
        dropout:float =0.2
    ) -> None:
        super(LSTMEncoder, self).__init__()
        self.dropout = dropout
        # create embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx,
            _weight=emb_init if emb_init is not None else None
        )
        # create lstm encoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # initial hidden and cell states for lstm
        self.h0 = nn.Parameter(torch.zeros(num_layers*2, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers*2, 1, hidden_size))
                
    def forward(self, 
        input_ids:torch.LongTensor, 
        input_mask:torch.BoolTensor
    ) -> torch.Tensor:
        # flatten parameters
        self.lstm.flatten_parameters()
        # pass through embedding
        b, s = input_ids.size()
        x = self.embedding.forward(input_ids)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pack padded sequences
        lengths = input_mask.sum(dim=-1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        # apply lstm encoder
        h0 = self.h0.repeat_interleave(b, dim=1)
        c0 = self.c0.repeat_interleave(b, dim=1)
        packed_x, _ = self.lstm(packed_x, (h0, c0))
        # unpack packed sequences
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=packed_x, 
            batch_first=True, 
            padding_value=0,
            total_length=s
        )
        return F.dropout(x, p=self.dropout, training=self.training)


class LSTMClassifier(nn.Module):
    """ Combination of a LSTM-encoder and a simple attention-based 
        Multi-label Classifier Module 
    """

    def __init__(self, 
        num_labels:int,
        # lstm
        hidden_size:int,
        num_layers:int,
        emb_init:np.ndarray,
        padding_idx:int,
        dropout:float,
        # attention module
        attention:nn.Module,
        # classifier module
        mlp:MLP
    ) -> None:
        # initialize module
        super(LSTMClassifier, self).__init__()
        # create lstm encoder
        self.enc = LSTMEncoder(
            vocab_size=emb_init.shape[0],
            embed_size=emb_init.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            padding_idx=padding_idx,
            emb_init=torch.from_numpy(emb_init).float(),
            dropout=dropout
        )
        # create label-attention classifier
        self.cls = LabelAttentionClassifier(
            hidden_size=hidden_size * 2, # x2 because lstm is bidirectional
            num_labels=num_labels,
            attention=attention,
            mlp=mlp
        )

    def forward(self, input_ids, input_mask, candidates=None):
        # apply encoder and pass output through classifer
        x = self.enc(input_ids, input_mask)
        return self.cls(x, input_mask, candidates)


class LSTMClassifierFactory(object):
    
    def __init__(self, 
        encoder_hidden_size:int,
        encoder_num_layers:int,
        attention_type:str,
        mlp_hidden_layers:list,
        mlp_bias:bool,
        mlp_activation:str,
        padding_idx:int,
        dropout:float,
        emb_init:np.ndarray,
    ) -> None:

        # build classifier keyword-arguments
        self.cls_kwargs = dict(
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            emb_init=emb_init,
            padding_idx=padding_idx,
            dropout=dropout
        )

        # get attention type
        self.attention_module = {
            'softmax-attention': SoftmaxAttention,
            'multi-head-attention': MultiHeadAttention
        }[attention_type]

        # get classifier type
        self.mlp_layers = [
            encoder_hidden_size * 2, 
            *mlp_hidden_layers, 
            1
        ]
        self.mlp_kwargs = dict(
            bias=mlp_bias,
            act_fn={
                'relu': torch.relu
            }[mlp_activation]
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

    @staticmethod
    def from_params(params, padding_idx:int, emb_init:np.ndarray):
        return LSTMClassifierFactory(
            encoder_hidden_size=params['encoder']['hidden_size'],
            encoder_num_layers=params['encoder']['num_layers'],
            attention_type=params['attention']['type'],
            mlp_hidden_layers=params['mlp']['hidden_layers'],
            mlp_bias=params['mlp']['bias'],
            mlp_activation=params['mlp']['activation'],
            dropout=params['dropout'],
            padding_idx=padding_idx,
            emb_init=emb_init
        )
