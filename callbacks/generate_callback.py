import torch
from lightning.pytorch.callbacks import Callback
from tokenizers import Tokenizer
import torch.nn.functional as F


class GenerationCallback(Callback):

    def __init__(self, tokeniser: Tokenizer, generate_every_n_val: int = 1, prompt: str = "Pewnego dnia ", n_tokens: int = 50):
        super().__init__()
        self.val_loops_count = 0
        self.generate_every_n_val = generate_every_n_val
        self.tokeniser = tokeniser
        self.prompt = prompt
        self.n_tokens = n_tokens
    def on_validation_start(self, trainer, pl_module):
        self.val_loops_count += 1

    def on_validation_end(self, trainer, pl_module):
        if self.val_loops_count % self.generate_every_n_val == 0:
            tokens = torch.tensor(self.tokeniser.encode(self.prompt).ids).unsqueeze(0)
            for _ in range(self.n_tokens):
                logits = pl_module((tokens, 0, 0))
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                top_probs, top_idx = torch.topk(probs, 5)  # only get the indices
                choice_idx = torch.multinomial(top_probs, num_samples=1)
                next_token = top_idx[0, choice_idx]
                tokens = torch.cat([tokens, next_token], dim=-1)

            decoded = self.tokeniser.decode(tokens.squeeze().tolist())
            print(decoded)