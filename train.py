import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

from datasets import NobelDataset
from models import NobelGPT
from utils.tokeniser_utils import train_tokeniser

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

batch_size = 32
splits = [0.8, 0.2]
max_seq_len = 256
vocab_size = 24000

data_path = os.path.join(path, "data")
text_path = os.path.join(data_path, "raw_txt")

# This will train the tokeniser
# tokeniser = train_tokeniser(data_path=data_path, text_path=text_path,vocab_size=vocab_size)

# Load trained tokeniser
tokeniser = Tokenizer.from_file(os.path.join(data_path, "tokeniser-ngpt.json"))

dataset = NobelDataset(tokeniser=tokeniser, max_seq_len=max_seq_len)

train, val = random_split(dataset=dataset, lengths=splits)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)

nobelgpt = NobelGPT(
    vocab_size=vocab_size,
    d_embed=384,
    d_model=384,
    n_head=6,
    n_layer=6,
    ffn_multiplier=4,
    dropout=0.2,
    max_seq_len=max_seq_len,
)
# nobelgpt = torch.compile(nobelgpt)

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
    strategy="ddp",
    callbacks=[checkpoint_callback],
    accelerator="gpu",
)
trainer.fit(nobelgpt, train_loader, val_loader)
