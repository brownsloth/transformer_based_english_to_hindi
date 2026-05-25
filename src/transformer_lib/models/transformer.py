"""Encoder-decoder Transformer for sequence-to-sequence tasks (e.g. translation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer_lib.models.layers import (
    Decoder,
    DecoderBlock,
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEmbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
    ProjectionLayer,
)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        target_embed: InputEmbeddings,
        src_pos_encoding: PositionalEncoding,
        target_pos_encoding: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos_encoding = src_pos_encoding
        self.target_pos_encoding = target_pos_encoding
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos_encoding(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_embed(target)
        target = self.target_pos_encoding(target)
        return self.decoder(target, encoder_output, target_mask, src_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)
    src_pos_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos_encoding = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
        )

    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
        )

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_pos_encoding,
        target_pos_encoding,
        projection_layer,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
