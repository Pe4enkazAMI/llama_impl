import torch
#from pytroch tutor.
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