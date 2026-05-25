"""Encoder-only Transformer for sequence classification (reuses shared layers)."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer_lib.models.layers import (
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEmbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
)


class EncoderClassifier(nn.Module):
    """
    Encoder stack + pooled representation + linear head.
    Use for text classification, sentiment, etc.
    """

    def __init__(
        self,
        encoder: Encoder,
        src_embed: InputEmbeddings,
        src_pos_encoding: PositionalEncoding,
        num_classes: int,
        d_model: int,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        assert pool in ("cls", "mean"), "pool must be 'cls' (first token) or 'mean'"
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos_encoding = src_pos_encoding
        self.pool = pool
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src)
        x = self.src_pos_encoding(x)
        x = self.encoder(x, src_mask)

        if self.pool == "mean":
            # Mask-aware mean pooling over sequence dimension
            mask = src_mask.squeeze(1).squeeze(1).unsqueeze(-1).float()  # (B, seq, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = x[:, 0, :]

        return self.classifier(x)


def build_encoder_classifier(
    vocab_size: int,
    seq_len: int,
    num_classes: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
    pool: str = "cls",
) -> EncoderClassifier:
    src_embed = InputEmbeddings(d_model, vocab_size)
    src_pos_encoding = PositionalEncoding(d_model, seq_len, dropout)

    encoder_blocks = [
        EncoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        for _ in range(N)
    ]
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    model = EncoderClassifier(
        encoder, src_embed, src_pos_encoding, num_classes, d_model, pool=pool
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
