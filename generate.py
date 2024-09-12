"""The main file for generating text using the trained nobelGPT model."""

import os
from os.path import isfile

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from models import NobelGPT
from utils.file_utils import create_if_not_exist

# ----------------
# Parameters
max_seq_len = 1024
vocab_size = 24000
d_embed = 1024
d_model = d_embed
n_head = 8
n_layer = 12
ffn_multiplier = 4
dropout = 0.2
dataset = "nobel"  # or "book"
# ----------------


full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

# Paths
log_dir = os.path.join(path, "logs")
# Load first checkpoint to be found
checkpoint_dir = os.path.join(log_dir, "checkpoints")
checkpoints = [
    f for f in os.listdir(checkpoint_dir) if isfile(os.path.join(checkpoint_dir, f))
]
checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
data_path = os.path.join(path, "data", dataset)
eval_path = os.path.join(path, "data", "eval")
create_if_not_exist(eval_path)

tokeniser = Tokenizer.from_file(os.path.join(data_path, "tokeniser-ngpt.json"))
tokeniser.decoder = ByteLevel()

model = NobelGPT.load_from_checkpoint(
    checkpoint_path,
    map_location=torch.device("cpu"),
    vocab_size=vocab_size,
    d_embed=d_embed,
    d_model=d_model,
    n_head=n_head,
    n_layer=n_layer,
    ffn_multiplier=ffn_multiplier,
    dropout=dropout,
    max_seq_len=max_seq_len,
)

model.eval()

print(
    "Evaluating model, please input prompts below. If you would like to stop use Ctrl-C."
)
while True:
    prompt = input("Model Prompt >")
    tokens_to_gen = input("Number of tokens requested >")
    tokens = torch.tensor(tokeniser.encode(prompt).ids).unsqueeze(0)
    tokens_to_gen = int(tokens_to_gen)
    for _ in range(tokens_to_gen):
        logits = model((tokens, 0, 0))
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, 5)  # only get the indices
        choice_idx = torch.multinomial(top_probs, num_samples=1)
        next_token = top_idx[0, choice_idx]
        tokens = torch.cat([tokens, next_token], dim=-1)

    decoded = tokeniser.decode(tokens.squeeze().tolist())
    print(decoded)
