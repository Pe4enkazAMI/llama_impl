import json
from glob import glob
import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from llm.tokenizer import Tokenizer


# class LLaMaDataset:
#     def __init__(self, data_dir, tokenizer_model_path, config_parser=None):
#         self.data_dir = data_dir
#         self.tokenizer = Tokenizer(model_path=tokenizer_model_path)
#         # self.files = sorted(glob(os.path.join(self.data_dir, "*.npy")))
#         self.config_parser = config_parser

#     def __getitem__(self, index):
#         return {
#             "input_ids": torch.tensor(np.load(self.files[index]))
#         }
    
#     def __len__(self):
#         return len(self.files)

#     def get_collate_fn(self):
#         pad_id = self.tokenizer.processor.pad_id()
#         def collate_fn(dataset_items):
#             """
#             Collate and pad fields in dataset items
#             """
#             result_batch = {}
#             result_batch["input_ids"] = pad_sequence(
#                 [item["input_ids"] for item in dataset_items], 
#                 batch_first=True,
#                 padding_value=pad_id
#             )
#             result_batch["padding_mask"] = (result_batch["input_ids"] == pad_id)
#             return result_batch
#         return collate_fn


# def copy_stories_to_dir(data_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     json_files = sorted(glob(os.path.join(data_dir, "*.json")))
#     for file in tqdm(json_files, "Processing json files"):
#         with open(file) as f_in:
#             item_list = json.load(f_in)
#             output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file))[0]}.txt")
#             with open(output_path, "w") as f_out:
#                 f_out.write("\n".join([item["story"] for item in item_list]))
#             print(f"Wrote to {output_path}")


# def save_tokenized_dataset(data_dir, output_dir, tokenizer):
#     os.makedirs(output_dir, exist_ok=True)
#     json_files = glob(os.path.join(data_dir, "*.json"))
#     item_index = 0
#     for file in tqdm(json_files, "Processing json files"):
#         with open(file) as f:
#             item_list = json.load(f)
#             for item in item_list:
#                 tokenized_story = tokenizer.encode(item["story"])
#                 np.save(os.path.join(output_dir, f"{item_index}.npy"), tokenized_story)
#                 item_index += 1

class LLaMaDataset:
    def __init__(self, data_dir, *args, **kwargs) -> None:
        self.path = data_dir
        self.round_tensor = torch.load(self.path)

    def __getitem__(self, index):
        return self.round_tensor[index, ...]
    def __len__(self):
        return self.round_tensor.shape[0]