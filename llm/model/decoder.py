from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from .resconnection import ResidualConnection
from .normlayer import RMSNorm
from .posencoding import PositionalEncoding, RoPE
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer
import numpy as np

class DecoderBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_heads, 
                 exp_factor,
                 dropout) -> None:
        
        super().__init__()
        
        self.block = nn.ModuleList([
            RMSNorm(emb_dim),
            MultiHeadAttention(emb_dim, emb_dim, num_heads, dropout=dropout),
            FeedForwardLayer(emb_dim, exp_factor)
        ])
    def forward(self, x, padding_mask):
        for i, layer in enumerate(self.block):
            if i == 0:
                x = layer(x)
                out = x.clone()
            if i == 1:
                x = layer(x, padding_mask) + out
            else:
                x = layer(x) + x
        return x
    


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_head, exp_factor, num_layers, dropout, *args, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.exp_factor = exp_factor,
        self.embedding = nn.Embedding(5001, self.emb_dim)

        self.num_layers = num_layers
        if kwargs["use_rope"]:
            self.rotary_base = kwargs["rotary_base"]
            self.pe = RoPE(emb_dim, self.rotary_base)
        else:
            self.pe = PositionalEncoding(emb_dim)

        zalupa_slonika = [
            DecoderBlock(emb_dim, num_head, exp_factor, dropout=dropout) for _ in range(num_layers)
        ]
        self.decoder = nn.ModuleList(zalupa_slonika)

        self.linear = nn.Linear(self.emb_dim, 5001)

    def forward(self, sentence):
        padding_mask = (sentence == 0).to(sentence.device)
        sentence = self.embedding(sentence)
        for layer in self.decoder:
            sentence = layer(sentence, padding_mask)
        return {"logits": self.linear(sentence)}
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)