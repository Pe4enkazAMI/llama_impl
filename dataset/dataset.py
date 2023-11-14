from torch.utils.data.dataset import Dataset
from tokenizer.tokenizer import Tokenizer
import torch 
from torch.nn.utils.rnn import pad_sequence

class LLaMaDataset(Dataset):
    def __init__(self, txt_file, root_dir, vocab_size) -> None:
        super().__init__()
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.vocab = Tokenizer(path_to_corpora=root_dir + txt_file,
                               vocab_size=vocab_size, 
                               model_prefix="llama", 
                               model_type="bpe")
        self.vocab.fit()

        with open(root_dir + txt_file, "r") as f:
            self.corpora = f.readlines()
        

    def __len__(self):
        return len(self.corpora)

    def __getitem__(self, index):
        return torch.tensor(self.vocab.encode(self.corpora[index]))
    

def collate_fn(dataset_items):
    bos_tensor = torch.tensor([1])
    eos_tensor = torch.tensor([2])

    sentences = [torch.cat([bos_tensor, dataset_items[i]]) for i in range(len(dataset_items))]
    sentences = pad_sequence(sentences, batch_first=True, padding_value=3)
    sentences = torch.hstack([sentences, 2*torch.ones(sentences.shape[0])[:, None]])
    return {"sentence": sentences.int()}
