import torch
import torch.nn as nn
from torch import Tensor


class ResidualConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x