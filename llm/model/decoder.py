import torch
from torch import Tensor
import torch.nn as nn
import copy
from .posenc import PositionalEncoding
from .utility import generate_square_mask, create_mask

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_head, ff_dim, dropout, batch_first=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(emb_dim, num_head, dropout=dropout, batch_first=batch_first)
        self.attn_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

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
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
    
    def forward(self, x, attention_mask, padding_mask):
        for layer in self.decoder:
            x = layer(x, attention_mask, padding_mask)
        return x

class LLaMa(nn.Module):
    def __init__(self, emb_dim, num_layers, num_head, exp_factor=4, vocab_size=5001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(embed_dim=emb_dim, dropout=0.1)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transformer = Decoder(
            DecoderLayer(emb_dim, num_head, exp_factor * emb_dim, 0.1, batch_first=True),
            num_layers=num_layers
        )
        self.classification = nn.Linear(emb_dim, vocab_size)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.shape)) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer(x, attention_mask, padding_mask)
        return {"logits": self.classification(x)}
    
    def get_next_token(self, prefix: Tensor, attention_mask: Tensor, padding_mask: Tensor):
        return self.forward(prefix, attention_mask, padding_mask)["logits"][:, -1, :]