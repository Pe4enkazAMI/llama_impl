import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, self.embed_dim)
        pos = torch.arange(max_len).reshape(-1, 1)
        denom = torch.pow(10000, (torch.arange(self.embed_dim) - (torch.arange(self.embed_dim) % 2)) / embed_dim)
        pe = pos / denom
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[-2], :]
        return self.dropout(x)