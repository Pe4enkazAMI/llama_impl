from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(Tensor([1.]))
        self.register_parameter("beta", self.beta)
    def forward(self, x):
        return x * F.sigmoid(x * self.beta)
    
class SwiGLU(nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.swish = Swish()

    def forward(self, x):
        out, gate = torch.chunk(x, chunks=2, dim=-1)
        return self.swish(gate) * out
