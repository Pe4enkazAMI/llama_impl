import torch
from torch import Tensor
import torch.nn as nn
import copy

nn.TransformerDecoderLayer
nn.TransformerDecoder

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, batch_first=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.attn_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask, padding_mask):
        x_ = self.norm1(x)
        x_, _ = self.attention(x, x, x, attn_mask=attention_mask, key_padding_mask=padding_mask)
        x_ = x + x_
        x_ = self.norm2(x_)
        x_ = x_ + self.feed_forward(x_)
        return x_

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    
    def forward(self, x, attention_mask, padding_mask):
        for layer in self.decoder:
            x = layer(x, attention_mask, padding_mask)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout: float = 0.1, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
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

class LLAMA(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size=5001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, dropout=0.1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = Decoder(
            DecoderLayer(embed_dim, num_heads, 4 * embed_dim, 0.1, batch_first=True),
            num_layers=num_layers
        )
        self.classification = nn.Linear(embed_dim, 5001)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        x = self.embedding(input_ids[:, :-1])
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return {"logits": self.classification(x)}
    
    def get_next_token(self, prefix: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        """ :returns: probabilities of next token """
        return self.forward(prefix, attention_mask, padding_mask)["logits"][:, -1, :]


def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(x, pad_idx, device):
    if len(x.shape) == 2:
        tgt_seq_len = x.shape[1]
    else:
        tgt_seq_len = x.shape[0]
    tgt_mask = generate_square_mask(tgt_seq_len, device)
    tgt_padding_mask = (x == pad_idx)
    return tgt_mask, tgt_padding_mask