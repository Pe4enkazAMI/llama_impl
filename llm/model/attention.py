from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, num_head, dropout):
        super().__init__()
        assert emb_dim % num_head == 0, "embedding dimension should be divisible by num_head"
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.head_dim = self.emb_dim // num_head

        self.qkv = nn.Linear(self.input_dim, 3*self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.emb_dim)
        self._reset_parameters()
        self.dropout = nn.Dropout(p=dropout)
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)

    @torch.no_grad()
    def make_attn_mask(self, x, device):
        return nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device)

    def forward(self, x, padding_mask=None, return_attention=False, use_flash=False):
        bs, seq_len, _  = x.shape
        mask = self.make_attn_mask(x, x.device)
        qkv = self.qkv(x)
        qkv = qkv.reshape(bs, seq_len, self.num_head, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        if use_flash:
            with torch.backends.cuda.enable_flash_sdp():
                value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
            value = value.permute(0, 2, 1, 3)
            value = value.reshape(bs, seq_len, self.emb_dim)
        else:
            score = q @ k.transpose(-1, -2)
            score = score / (q.shape[-1]**(0.5))
            if mask is not None:
                padding_mask = padding_mask.float().masked_fill(padding_mask == 1, float('-inf')).masked_fill(padding_mask == 0, float(0.0))
                score = score + mask + padding_mask.expand(-1, score.shape[1], score.shape[2])
            score = F.softmax(score, dim=-1)
            value = score @ v

        value = value.permute(0, 2, 1, 3)
        value = value.reshape(bs, seq_len, self.emb_dim)

        out = self.dropout(self.out(value))
        return out

