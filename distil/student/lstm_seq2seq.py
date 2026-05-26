"""Bahdanau attention LSTM seq2seq — fast CPU inference student."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # encoder_outputs: (B, S, enc_dim), decoder_hidden: (B, dec_dim)
        scores = self.v(torch.tanh(self.W_enc(encoder_outputs) + self.W_dec(decoder_hidden).unsqueeze(1)))
        scores = scores.squeeze(2)  # (B, S)
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        pad_id: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.bidirectional = True
        self.enc_out_dim = hidden_dim * 2

        self.src_embed = nn.Embedding(src_vocab, embed_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim, padding_idx=pad_id)

        enc_dropout = dropout if encoder_layers > 1 else 0.0
        dec_dropout = dropout if decoder_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=enc_dropout,
        )
        self.enc_hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.enc_cell_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.attention = BahdanauAttention(self.enc_out_dim, hidden_dim)
        self.decoder = nn.LSTM(
            embed_dim + self.enc_out_dim,
            hidden_dim,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dec_dropout,
        )
        self.out = nn.Linear(hidden_dim, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def _encode(
        self, src: torch.Tensor, src_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.dropout(self.src_embed(src))
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            enc_out_packed, (h, c) = self.encoder(packed)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
        else:
            enc_out, (h, c) = self.encoder(embedded)

        # (layers*2, B, H) -> (layers, B, H)
        h = h.view(self.encoder_layers, 2, h.size(1), h.size(2))
        c = c.view(self.encoder_layers, 2, c.size(1), c.size(2))
        h = torch.cat([h[:, 0], h[:, 1]], dim=2)
        c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        h = self.enc_hidden_proj(h)
        c = self.enc_cell_proj(c)
        return enc_out, (h, c)

    def _src_pad_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id).int()

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher forcing. Returns logits (B, seq_len, tgt_vocab)."""
        B, T = decoder_input.shape
        src_mask = self._src_pad_mask(encoder_input)
        src_lens = src_mask.sum(dim=1).clamp(min=1)

        enc_out, dec_state = self._encode(encoder_input, src_lens)
        logits = []

        for t in range(T):
            emb = self.dropout(self.tgt_embed(decoder_input[:, t]))
            context = self.attention(enc_out, dec_state[0][-1], src_mask)
            dec_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, dec_state = self.decoder(dec_in, dec_state)
            logits.append(self.out(out.squeeze(1)))

        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def greedy_decode(
        self,
        encoder_input: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        device = encoder_input.device
        src_mask = self._src_pad_mask(encoder_input)
        src_lens = src_mask.sum(dim=1).clamp(min=1)
        enc_out, dec_state = self._encode(encoder_input, src_lens)

        ys = torch.full((encoder_input.size(0), 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            emb = self.tgt_embed(ys[:, -1])
            context = self.attention(enc_out, dec_state[0][-1], src_mask)
            dec_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, dec_state = self.decoder(dec_in, dec_state)
            next_id = self.out(out.squeeze(1)).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return ys.squeeze(0)


def build_lstm_student(
    src_vocab: int,
    tgt_vocab: int,
    pad_id: int,
    **kwargs,
) -> LSTMSeq2Seq:
    return LSTMSeq2Seq(src_vocab, tgt_vocab, pad_id, **kwargs)
