"""Frozen transformer teacher — returns log-probs for distillation."""

from __future__ import annotations

import torch

from transformer_lib.models.transformer import Transformer, build_transformer
from transformer_lib.config import Config as TeacherConfig, PAD_TOKEN


class TransformerTeacher:
    def __init__(self, config: TeacherConfig, checkpoint: str, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.pad_id: int | None = None

        from tokenizers import Tokenizer

        tok_src = Tokenizer.from_file(str(config.tokenizer_path(config.data.lang_src)))
        tok_tgt = Tokenizer.from_file(str(config.tokenizer_path(config.data.lang_tgt)))
        self.pad_id = tok_tgt.token_to_id(PAD_TOKEN)

        m = config.model
        self.model = build_transformer(
            tok_src.get_vocab_size(),
            tok_tgt.get_vocab_size(),
            m.seq_len,
            m.seq_len,
            d_model=m.d_model,
            N=m.num_layers,
            h=m.num_heads,
            dropout=m.dropout,
            d_ff=m.d_ff,
        ).to(device)

        ckpt = torch.load(
            config.weights_path(checkpoint),
            map_location=device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def log_probs(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return teacher log-probs (B, seq_len, vocab)."""
        enc_out = self.model.encode(encoder_input, encoder_mask)
        dec_out = self.model.decode(decoder_input, enc_out, decoder_mask, encoder_mask)
        return self.model.project(dec_out)
