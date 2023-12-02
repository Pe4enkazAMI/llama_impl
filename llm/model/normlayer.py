from torch import Tensor
import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, p=-1, eps=1e-8, bias=True) -> None:
        super().__init__()
        self.eps = eps
        self.p = p
        self.bias_ = bias
        self.emb_dim = emb_dim

        self.scale = nn.Parameter(torch.ones(self.emb_dim))
        self.register_parameter("scale", self.scale)
        if self.bias_:
            self.bias = nn.Parameter(torch.zeros(self.emb_dim))
            self.register_parameter("bias", self.bias)

    def forward(self, x):
        if self.p < 0 or self.p > 1:
            norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
            dim_x = self.emb_dim
        else:
            part = int(self.emb_dim * self.p)
            part_x, _ = torch.split(x, [part, self.emb_dim - part], dim=-1)
            norm_x = torch.norm(part_x, p=2, dim=-1, keepdim=True)
            dim_x = part

        rms_x = norm_x * (dim_x ** (-0.5))
        x_hat = x/ (rms_x + self.eps)

        if self.bias_:
            return x_hat * self.scale + self.bias
        return x_hat * self.scale
        
