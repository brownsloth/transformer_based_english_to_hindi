"""Greedy decoding for validation and inference."""

from __future__ import annotations

import torch
from tokenizers import Tokenizer

from transformer_lib.models.layers import causal_mask
from transformer_lib.models.transformer import Transformer


def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    from transformer_lib.config import EOS_TOKEN, SOS_TOKEN

    sos_id = tokenizer_tgt.token_to_id(SOS_TOKEN)
    eos_id = tokenizer_tgt.token_to_id(EOS_TOKEN)

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1, dtype=source.dtype, device=device).fill_(sos_id)

    while decoder_input.size(1) < max_len:
        dec_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        dec_out = model.decode(decoder_input, encoder_output, dec_mask, source_mask)
        prob = model.project(dec_out[:, -1, :])  # (batch, vocab)
        _, next_word = torch.max(prob, dim=-1)  # (batch,)
        next_token = next_word.view(1, 1).to(decoder_input.dtype)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_word.item() == eos_id:
            break

    return decoder_input.squeeze(0)
