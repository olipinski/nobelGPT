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
# tokeniser = NobelGPTTokeniser(sentence_tokenised=True, data_path=os.path.join(path, "data"))

# Load trained tokeniser
tokeniser = NobelGPTTokeniser(
    sentence_tokenised=True, pretrained=os.path.join(path, "data")
)

# Truncation and padding params
tokeniser.get_tokeniser().enable_padding(
    pad_id=3, pad_token="[PAD]", pad_to_multiple_of=2, direction="left"
)
tokeniser.get_tokeniser().enable_truncation(max_length=max_seq_len)

dataset = NobelDataset(sentence_split=True, tokeniser=tokeniser)

train, val = random_split(dataset=dataset, lengths=splits)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

nobelgpt = NobelGPT(max_batch_size=batch_size)
#nobelgpt = torch.compile(nobelgpt)

trainer = Trainer()
trainer.fit(nobelgpt, train_loader, val_loader)
