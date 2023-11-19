from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from .resconnection import ResidualConnection
from .normlayer import RMSNorm
from .posencoding import PositionalEncoding, RoPE
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer


class DecoderBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_heads, 
                 exp_factor,
                 dropout) -> None:
        
        super().__init__()
        
        self.block = nn.Sequential(
            ResidualConnection(
                MultiHeadAttention(emb_dim, emb_dim, num_heads, dropout=dropout)
            ),
            RMSNorm(emb_dim),
            ResidualConnection(
                FeedForwardLayer(emb_dim, exp_factor)
            ),
            RMSNorm(emb_dim)
        )
    def forward(self, x):
        return self.block(x)
    


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_head, exp_factor, num_layers, dropout, *args, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.exp_factor = exp_factor,
        self.embedding = nn.Embedding(32000, self.emb_dim)

        self.num_layers = num_layers
        if kwargs["use_rope"]:
            self.rotary_base = kwargs["rotary_base"]
            self.pe = RoPE(emb_dim, self.rotary_base)
        else:
            self.pe = PositionalEncoding(emb_dim)

        zalupa_slonika = [
            DecoderBlock(emb_dim, num_head, exp_factor, dropout=dropout) for _ in range(num_layers)
        ]
        self.decoder = nn.Sequential(*zalupa_slonika)

        self.linear = nn.Linear(self.emb_dim, 32000)

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(self.decoder(x))