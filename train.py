import os
import platform
import time

import shortuuid
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

from datasets import NobelDataset
from models import NobelGPT
from utils.file_utils import create_if_not_exist
from utils.tokeniser_utils import train_tokeniser

# ----------------
# Parameters
batch_size = 16
splits = [0.8, 0.2]
max_seq_len = 1024
vocab_size = 24000
d_embed = 1024
d_model = d_embed
n_head = 8
n_layer = 12
ffn_multiplier = 4
dropout = 0.2
# ----------------

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

log_dir = os.path.join(path, "logs")

run_uuid = shortuuid.uuid()[:8]

# Check whether the specified paths exist or not and create them
create_if_not_exist(log_dir)
create_if_not_exist(os.path.join(log_dir, "lightning_tensorboard"))
create_if_not_exist(os.path.join(log_dir, "lightning_wandb"))

# TODO check which dataset to set the path below to

data_path = os.path.join(path, "data")
text_path = os.path.join(data_path, "raw_txt")

# Check GPU capability for compile
compile_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap[0] >= 7:
        compile_ok = True
    if platform.uname()[0] == "Windows":
        compile_ok = False

# This will train the tokeniser
# tokeniser = train_tokeniser(data_path=data_path, text_path=text_path,vocab_size=vocab_size)

# Load trained tokeniser
tokeniser = Tokenizer.from_file(os.path.join(data_path, "tokeniser-ngpt.json"))

# If changing tokensier must delete the processsed folder!!!!!
# IMPORTANT
dataset = NobelDataset(tokeniser=tokeniser, max_seq_len=max_seq_len)

train, val = random_split(dataset=dataset, lengths=splits)

train_loader = DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
)
val_loader = DataLoader(
    val, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
)

nobelgpt = NobelGPT(
    vocab_size=vocab_size,
    d_embed=1024,
    d_model=1024,
    n_head=8,
    n_layer=12,
    ffn_multiplier=4,
    dropout=0.2,
    max_seq_len=max_seq_len,
)
if compile_ok:
    nobelgpt = torch.compile(nobelgpt)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/", save_top_k=2, monitor="val_loss"
)

trainer = Trainer(
    devices=-1,
    # strategy=DeepSpeedStrategy(
    #     stage=3,
    #     offload_optimizer=True,
    #     offload_parameters=True,
    # )
    strategy="fsdp",
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    precision=16,
    max_epochs=100,
)
trainer.fit(nobelgpt, train_loader, val_loader)
