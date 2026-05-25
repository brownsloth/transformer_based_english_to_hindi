#!/usr/bin/env python3
"""Translate a single English sentence using a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, load_config
from transformer_lib.models.transformer import build_transformer
from transformer_lib.training.decode import greedy_decode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "en_hi_translation.yaml"))
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Epoch index (0 or 00) — maps to smoke_00.pt / tmodel_05.pt",
    )
    p.add_argument("--text", required=True, help="Source sentence")
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_src = Tokenizer.from_file(str(config.tokenizer_path(config.data.lang_src)))
    tok_tgt = Tokenizer.from_file(str(config.tokenizer_path(config.data.lang_tgt)))

    m = config.model
    model = build_transformer(
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

    ckpt_path = config.weights_path(args.checkpoint)
    if not ckpt_path.exists():
        available = sorted(config.weights_dir.glob("*.pt")) if config.weights_dir.exists() else []
        hint = ", ".join(p.name for p in available) or "(none)"
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nAvailable: {hint}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sos = tok_src.token_to_id(SOS_TOKEN)
    eos = tok_src.token_to_id(EOS_TOKEN)
    pad = tok_src.token_to_id(PAD_TOKEN)

    ids = tok_src.encode(args.text).ids
    pad_len = max(0, m.seq_len - len(ids) - 2)
    enc = torch.cat(
        [
            torch.tensor([sos]),
            torch.tensor(ids),
            torch.tensor([eos]),
            torch.tensor([pad] * pad_len),
        ]
    )[: m.seq_len].unsqueeze(0).to(device)

    mask = (enc != pad).int().unsqueeze(0).unsqueeze(0).to(device)
    out = greedy_decode(model, enc, mask, tok_tgt, m.seq_len, device)
    print(tok_tgt.decode(out.cpu().tolist()))


if __name__ == "__main__":
    main()
