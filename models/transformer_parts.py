import torch
import torch.nn.functional as F
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, d_embed, d_head, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.key_matrix = nn.Linear(d_embed, d_head, bias=False)
        self.query_matrix = nn.Linear(d_embed, d_head, bias=False)
        self.value_matrix = nn.Linear(d_embed, d_head, bias=False)

    def forward(self, x):
        k = self.key_matrix(x)
        q = self.query_matrix(x)
        v = self.value_matrix(x)
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout
        )
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, d_head, n_head, d_model, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(d_embed, d_head, dropout) for _ in range(n_head)]
        )
        self.project_up = nn.Linear(d_head * n_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.project_up(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_embed, ffn_multiplier: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_embed, d_embed * ffn_multiplier),
            nn.GELU(),
            nn.Linear(d_embed * ffn_multiplier, d_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_embed, d_model, n_head, ffn_multiplier, dropout=0.1):
        super().__init__()
        self.mh_attention = MultiHeadAttention(
            d_embed, d_model, n_head, d_model, dropout
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_embed, ffn_multiplier, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mh_attention(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x
