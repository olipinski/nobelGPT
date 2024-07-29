import os
import platform

import shortuuid
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

from datasets import BookDataset, NobelDataset
from models import NobelGPT
from models.transformer_parts import TransformerBlock
from utils.file_utils import create_if_not_exist
from utils.tokeniser_utils import train_tokeniser
from callbacks import GenerationCallback

# ----------------
# Parameters
batch_size = 16
splits = [0.9, 0.1]
max_seq_len = 1024
vocab_size = 24000
d_embed = 1024
d_model = d_embed
n_head = 8
n_layer = 12
ffn_multiplier = 4
dropout = 0.2
dataset = "nobel"  # or "book"
scratch_train_tokeniser = False
model_type = "gpt"  # for now just this model
# ----------------

# Experiment name
run_uuid = shortuuid.uuid()[:8]
exp_name = f"{dataset}_{model_type}_{run_uuid}"


# All the paths to be created
# ----------------
full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

# Check whether the specified paths exist or not and create them
log_dir = os.path.join(path, "logs")
create_if_not_exist(log_dir)
tensorboard_dir = os.path.join(log_dir, "lightning_tensorboard")
create_if_not_exist(tensorboard_dir)
checkpoint_dir = os.path.join(log_dir, "checkpoints")
create_if_not_exist(checkpoint_dir)

create_if_not_exist(os.path.join(path, "data"))
data_path = os.path.join(path, "data", dataset)
create_if_not_exist(data_path)
text_path = os.path.join(data_path, "raw_txt")
create_if_not_exist(text_path)
# ----------------

# Check GPU capability for compile
compile_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap[0] >= 7:
        compile_ok = True
    if platform.uname()[0] == "Windows":
        compile_ok = False

# This will train the tokeniser
if scratch_train_tokeniser:
    tokeniser = train_tokeniser(
        data_path=data_path, text_path=text_path, vocab_size=vocab_size
    )
else:
    # Or load trained tokeniser
    tokeniser = Tokenizer.from_file(os.path.join(data_path, "tokeniser-ngpt.json"))

# If changing tokeniser must delete the processed folder!
# IMPORTANT
if dataset == "nobel":
    dataset = NobelDataset(tokeniser=tokeniser, max_seq_len=max_seq_len)
elif dataset == "book":
    dataset = BookDataset(tokeniser=tokeniser, max_seq_len=max_seq_len)
else:
    raise ValueError(f"Dataset {dataset} not supported.")

train, val = random_split(dataset=dataset, lengths=splits)

train_loader = DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True
)
val_loader = DataLoader(
    val, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    save_top_k=1,
    monitor="val_loss",
    save_last="link",
    mode="min",
    auto_insert_metric_name=True,
)
summary_callback = ModelSummary(max_depth=5)
gen_callback = GenerationCallback(generate_every_n_val=1)

tensorboard_logger = TensorBoardLogger(
    save_dir=tensorboard_dir,
    name=exp_name,
)

# Sharded training
policy = {TransformerBlock}
strategy = FSDPStrategy(
    auto_wrap_policy=policy,
)

precision = "bf16" if torch.cuda.is_bf16_supported() else "16"

trainer = Trainer(
    devices=-1,
    strategy=strategy,
    callbacks=[checkpoint_callback, summary_callback, gen_callback],
    logger=tensorboard_logger,
    accelerator="gpu",
    precision=precision,
    max_epochs=100,
    check_val_every_n_epoch=5,
)

# Automatically moves the model to device and with the correct precision
with trainer.init_module():
    if model_type == "gpt":
        model = NobelGPT(
            vocab_size=vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Model {model_type} not supported.")

if compile_ok:
    model = torch.compile(model)

trainer.fit(model, train_loader, val_loader)
