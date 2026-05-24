"""
Baseline Models for Pathfinder Comparison
==========================================
LSTM and Transformer baselines to compare against S4.

Expected results on Pathfinder-32 (synthetic):
  LSTM        ~55–60%  (near random — fails long-range)
  Transformer ~60–65%  (struggles with long sequences)
  S4          ~85–94%  (handles long-range dependencies)
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# LSTM Baseline
# ─────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    Standard LSTM for sequence classification.
    Limitation: vanishing gradients → poor long-range memory.
    """

    def __init__(
        self,
        d_input:     int = 1,
        d_model:     int = 128,
        num_layers:  int = 2,
        num_classes: int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.lstm    = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        """x: (B, L) or (B, L, C)"""
        if x.dim() == 2:
            x = x.unsqueeze(-1).float()
        x = self.encoder(x)                    # (B, L, d_model)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                            # last layer hidden state (B, d_model)
        return self.decoder(h)


# ─────────────────────────────────────────────
# Transformer Baseline
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Transformer encoder for sequence classification.
    Limitation: O(N²) attention — memory explodes for long sequences.
    """

    def __init__(
        self,
        d_input:     int = 1,
        d_model:     int = 128,
        num_layers:  int = 2,
        num_heads:   int = 4,
        num_classes: int = 2,
        seq_len:     int = 1024,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.encoder   = nn.Linear(d_input, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))

        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        """x: (B, L) or (B, L, C)"""
        if x.dim() == 2:
            x = x.unsqueeze(-1).float()
        B = x.size(0)

        x   = self.encoder(x)                          # (B, L, d_model)
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (B, L+1, d_model)
        x   = self.pos_enc(x)
        x   = self.transformer(x)
        cls_out = x[:, 0]                              # (B, d_model)
        return self.decoder(cls_out)


# ─────────────────────────────────────────────
# Model summary utility
# ─────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, L = 4, 1024

    lstm        = LSTMBaseline()
    transformer = TransformerBaseline(seq_len=L)

    x = torch.randn(B, L)
    print(f"LSTM        output: {lstm(x).shape}         params: {count_params(lstm):,}")
    print(f"Transformer output: {transformer(x).shape}  params: {count_params(transformer):,}")
    print("✓ Baselines OK")
