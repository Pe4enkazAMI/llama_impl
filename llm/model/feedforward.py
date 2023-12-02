from torch import Tensor
import torch.nn as nn
import torch
from .activation import SwiGLU

class FeedForwardLayer(nn.Module):
    def __init__(self, emb_dim, exp_factor) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(emb_dim, exp_factor*emb_dim),
            SwiGLU(dim=-1),
            nn.Linear(exp_factor * emb_dim // 2, emb_dim),
        )
    def forward(self, x):
        return self.block(x)
