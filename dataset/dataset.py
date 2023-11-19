from torch.utils.data.dataset import Dataset
from tokenizer.tokenizer import Tokenizer
import torch 
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np

class LLaMaDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__() 
        self.root_dir = root_dir
        self.npy_files = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, index):
        return torch.from_numpy(np.load(self.root_dir + "/" + self.npy_files[index]))
    

def collate_fn(dataset_items):
    bos_tensor = torch.tensor([1])
    eos_tensor = torch.tensor([2])

    sentences = [torch.cat([bos_tensor, dataset_items[i]]) for i in range(len(dataset_items))]
    sentences = pad_sequence(sentences, batch_first=True, padding_value=3)
    sentences = torch.hstack([sentences, 2*torch.ones(sentences.shape[0])[:, None]])
    return {"sentence": sentences.int()}
