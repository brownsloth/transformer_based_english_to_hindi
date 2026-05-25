"""Bilingual parallel corpus dataset for seq2seq training."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN
from transformer_lib.models.layers import causal_mask


class BilingualDataset(Dataset):
    def __init__(
        self,
        ds: Any,
        tokenizer_src: Any,
        tokenizer_tgt: Any,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
        truncate_long: bool = False,
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.truncate_long = truncate_long

        self.sos_token_id = tokenizer_src.token_to_id(SOS_TOKEN)
        self.eos_token_id = tokenizer_src.token_to_id(EOS_TOKEN)
        self.pad_token_id = tokenizer_src.token_to_id(PAD_TOKEN)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.ds[index]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        enc_ids = self.tokenizer_src.encode(src_text).ids
        dec_ids = self.tokenizer_tgt.encode(tgt_text).ids

        if self.truncate_long:
            max_enc = self.seq_len - 2
            max_dec = self.seq_len - 1
            if len(enc_ids) > max_enc:
                enc_ids = enc_ids[:max_enc]
            if len(dec_ids) > max_dec:
                dec_ids = dec_ids[:max_dec]

        enc_pad = self.seq_len - len(enc_ids) - 2
        dec_pad = self.seq_len - len(dec_ids) - 1

        if enc_pad < 0 or dec_pad < 0:
            raise ValueError(
                f"Sentence too long for seq_len={self.seq_len}. "
                f"src_len={len(enc_ids)}, tgt_len={len(dec_ids)}"
            )

        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_token_id], dtype=torch.int64),
                torch.tensor(enc_ids, dtype=torch.int64),
                torch.tensor([self.eos_token_id], dtype=torch.int64),
                torch.tensor([self.pad_token_id] * enc_pad, dtype=torch.int64),
            ]
        )

        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_token_id], dtype=torch.int64),
                torch.tensor(dec_ids, dtype=torch.int64),
                torch.tensor([self.pad_token_id] * dec_pad, dtype=torch.int64),
            ]
        )

        labels = torch.cat(
            [
                torch.tensor(dec_ids, dtype=torch.int64),
                torch.tensor([self.eos_token_id], dtype=torch.int64),
                torch.tensor([self.pad_token_id] * dec_pad, dtype=torch.int64),
            ]
        )

        pad_id = self.pad_token_id
        encoder_mask = (encoder_input != pad_id).int().unsqueeze(0).unsqueeze(0)
        decoder_mask = (
            (decoder_input != pad_id).int().unsqueeze(0).unsqueeze(0)
            & causal_mask(decoder_input.size(0))
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": labels,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
