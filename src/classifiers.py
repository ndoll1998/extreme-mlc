import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmlc.modules import MLP, LabelAttentionClassifier

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




