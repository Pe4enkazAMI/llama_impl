import json
from glob import glob
import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from llm.tokenizer import Tokenizer



class LLaMaDataset:
    def __init__(self, data_dir, *args, **kwargs) -> None:
        self.path = data_dir
        self.round_tensor = torch.load(self.path)

    def __getitem__(self, index):
        return self.round_tensor[index, ...]
    def __len__(self):
        return self.round_tensor.shape[0]