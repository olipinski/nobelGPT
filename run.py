import os

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, random_split

from datasets import NobelDataset
from model import NobelLLama
from tokeniser import NobelGPTTokeniser

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

batch_size = 32
splits = [0.8, 0.2]
max_seq_len = 2048

# This will also train the tokeniser
# tokeniser = NobelGPTTokeniser(data_path=os.path.join(path, "data"))

# Load trained tokeniser
tokeniser = NobelGPTTokeniser(pretrained=os.path.join(path, "data"))

dataset = NobelDataset(tokeniser=tokeniser.get_tokeniser())

train, val = random_split(dataset=dataset, lengths=splits)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)

nobelgpt = NobelLLama(
    dim=2048,
    n_layers=16,
    n_heads=8,
    n_kv_heads=8,
    vocab_size=8000,
    multiple_of=128,
    ffn_dim_multiplier=None,
    norm_eps=1e-5,
    rope_theta=500000,
    use_scaled_rope=False,
    max_batch_size=batch_size,
    max_seq_len=max_seq_len,
)
nobelgpt = torch.compile(nobelgpt)

trainer = Trainer()
trainer.fit(nobelgpt, train_loader, val_loader)
