"""The main module for the transformer model."""
import math
import time

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
        stats: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.n_layer = n_layer
        self.ffn_multiplier = ffn_multiplier
        self.dropout = dropout
        self.stats = stats

        self.save_hyperparameters()

        # Initialised in configure model
        self.token_embedding = None
        self.position_embedding = None
        self.blocks = None
        self.ln1 = None
        self.project_vocab = None

    # For Lightning, layers should be instantiated in configure model.
    # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
    def configure_model(self) -> None:
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_embed)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    self.d_embed,
                    self.d_model,
                    self.n_head,
                    self.ffn_multiplier,
                    self.dropout,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.ln1 = nn.LayerNorm(self.d_model)
        self.project_vocab = nn.Linear(self.d_model, self.vocab_size)

        # Weight sharing
        self.project_vocab.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=np.round(1 / math.sqrt(self.d_model), 2))
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward a single batch through the network.

        Parameters
        ----------
        x:
            Batch.

        Returns
        -------
            Next token prediction.
        """
        tokens, _, _ = x
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
        """
        Run the training for a single batch, with a given batch id.

        Parameters
        ----------
        batch:
            Batch to process.
        batch_idx: int
            ID of the batch.
        """
        start = time.time()
        tokens, labels, _ = batch
        labels = labels.long()
        pred = self.forward(batch)
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))
        end = time.time() - start
        values = {
            "train_loss": loss,
            "ms_per_forward": end,
            "tokens_per_sec": 0,
        }

        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Run the validation for a single batch, with a given batch id.

        Parameters
        ----------
        batch:
            Batch to process.
        batch_idx: int
            ID of the batch.
        """
        _, labels, _ = batch
        pred = self.forward(batch)
        labels = labels.long()
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))
        values = {"val_loss": loss}
        self.log_dict(values, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizers to be used for the training.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Optimizer to be used for training.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=2.5e-4, betas=(0.9, 0.95), eps=1e-8
        )
        return optimizer
