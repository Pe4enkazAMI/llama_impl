from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F



class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe =  torch.zeros(size=(1, max_len, embed_dim))
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000))/embed_dim))
        pe[..., 0::2] = torch.sin(pos * div)
        pe[..., 1::2] = torch.cos(pos * div)
        pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[..., :x.shape[-2], :x.shape[-1]]
        return x
    

class RoPE(nn.Module):
    def __init__(self, dim, base) -> None:
        super().__init__()
        inv_freq = 1 / (base**torch.arange(0, dim, 2).float()/dim)
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cache = None
        self.cos_cache = None
        self.sin_cache = None

    def _cache(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cache:
            self.seq_len_cache = seq_len
            angle = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
            freq = angle[:, None] @ self.inv_freq[None, :]
            emb = torch.cat([freq, freq], dim=-1).to(x.device)
            self.cos_cache = emb.cos()
            self.sin_cache = emb.sin()

    def _neg_half(self, x):
        left, right = torch.chunk(x, chunks=2, dim=-1)
        return torch.cat([-right, left], dim=-1)
    
    def forward(self, x):
        self._cache(x)
        x_rope, x_pass = torch.chunk(x, 2, -1)
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cache[..., :x_rope.shape[-1]]) + (neg_half_x * self.sin_cache[..., :x_rope.shape[-1]])
        return torch.cat([x_rope, x_pass], dim=-1)
        