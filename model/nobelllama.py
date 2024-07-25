import lightning as L
import torch
import torch.nn.functional as F

from model.model_defs import Transformer


class NobelGPT(L.LightningModule):
    def __init__(
        self,
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
        max_batch_size=32,
        max_seq_len=2048,
    ):
        super().__init__()
        self.nobelgpt = Transformer(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            use_scaled_rope=use_scaled_rope,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    def forward(self, x):
        tokens, start_pos, _, _, = x
        x = self.nobelgpt(tokens, start_pos)
        return x

    def training_step(self, batch, batch_idx):
        _, _, labels, mask = batch
        pred = self.forward(batch)
        loss = F.cross_entropy(pred, labels, reduction="none")
        # Mask out any loss related to padded tokens
        loss *= mask
        loss = loss.mean()
        values = {
            "train_loss": loss,
        }
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, labels = batch
        pred = self.forward(batch)
        loss = F.cross_entropy(pred, labels)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer
