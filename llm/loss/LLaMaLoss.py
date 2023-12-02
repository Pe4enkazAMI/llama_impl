from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn 

class LLaMaLoss(nn.CrossEntropyLoss):
    def __init__(self, 
                 weight: Tensor | None = None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0,
                 pad_id=0) -> None:
        ignore_index = pad_id
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    def forward(self, logits: Tensor, sentence: Tensor) -> Tensor:
        return super().forward(logits[..., :-1].transpose(1, 2),
                               sentence[..., 1:])
