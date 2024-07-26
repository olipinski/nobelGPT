import math

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.transformer_parts import TransformerBlock


class NobelGPT(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        d_embed,
        d_model,
        max_seq_len,
        n_head,
        n_layer,
        ffn_multiplier,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.position_embedding = nn.Embedding(max_seq_len, d_embed)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_embed, d_model, n_head, ffn_multiplier, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.project_vocab = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=np.round(1 / math.sqrt(self.d_model), 2))
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        tokens, labels, masks = x
        tokens = tokens.int()
        tokens_embedded = self.token_embedding(tokens)
        position_embedded = self.position_embedding(
            torch.arange(tokens_embedded.shape[1], device=self.device)
        )
        tokens_processed = tokens_embedded + position_embedded
        out = self.blocks(tokens_processed)
        out = self.ln1(out)
        out = self.project_vocab(out)

        return out

    def training_step(self, batch, batch_idx):
        tokens, labels, _ = batch
        labels = labels.long()
        pred = self.forward(batch)
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))
        values = {
            "train_loss": loss,
        }
        self.log_dict(values, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, labels, _ = batch
        pred = self.forward(batch)
        labels = labels.long()
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))
        values = {"val_loss": loss}
        self.log_dict(values, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configure the optimizers to be used for the training.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Optimizer to be used for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=2.5e-4)
        return optimizer
