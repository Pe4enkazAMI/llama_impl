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
                 label_smoothing: float = 0.1,
                 pad_id=0) -> None:
        ignore_index = pad_id
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    def forward(self, logits: Tensor, input_ids: Tensor, *args, **kwargs) -> Tensor:    
        return super().forward(logits.reshape(-1, logits.shape[-1]),
                               input_ids.long().reshape(-1))
