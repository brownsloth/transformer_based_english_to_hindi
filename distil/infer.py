#!/usr/bin/env python3
"""Translate English → Hindi with a distilled LSTM student checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import load_distil_config
from distil.student.lstm_seq2seq import build_lstm_student
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN


def encode_source(text: str, tok_src: Tokenizer, seq_len: int) -> torch.Tensor:
    sos = tok_src.token_to_id(SOS_TOKEN)
    eos = tok_src.token_to_id(EOS_TOKEN)
    pad = tok_src.token_to_id(PAD_TOKEN)

    ids = tok_src.encode(text).ids
    max_body = seq_len - 2
    if len(ids) > max_body:
        ids = ids[:max_body]

    pad_len = max(0, seq_len - len(ids) - 2)
    enc = torch.cat(
        [
            torch.tensor([sos]),
            torch.tensor(ids),
            torch.tensor([eos]),
            torch.tensor([pad] * pad_len),
        ]
    )[:seq_len]
    return enc.unsqueeze(0)


def translate(
    text: str,
    model,
    tok_src: Tokenizer,
    tok_tgt: Tokenizer,
    seq_len: int,
    device: torch.device,
) -> str:
    sos_id = tok_src.token_to_id(SOS_TOKEN)
    eos_id = tok_src.token_to_id(EOS_TOKEN)
    pad_id = tok_tgt.token_to_id(PAD_TOKEN)

    enc = encode_source(text, tok_src, seq_len).to(device)
    out = model.greedy_decode(enc, sos_id, eos_id, seq_len)
    ids = [t for t in out.cpu().tolist() if t not in (sos_id, eos_id, pad_id)]
    return tok_tgt.decode(ids)


def main() -> None:
    p = argparse.ArgumentParser(description="LSTM student inference (en → hi)")
    p.add_argument("--config", default=str(ROOT / "distil" / "configs" / "lstm_kd.yaml"))
    p.add_argument("--checkpoint", required=True, help="Epoch index, e.g. 9 or 09")
    p.add_argument("--text", action="append", help="English sentence (repeat for multiple)")
    p.add_argument("--teacher-artifacts", default=None, help="Path to outputs/en_hi (tokenizers)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.text:
        p.error("Provide at least one --text \"...\"")

    cfg = load_distil_config(args.config)
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts

    teacher_cfg = cfg.load_teacher_run_config()
    device = torch.device(args.device)
    seq_len = cfg.student.seq_len

    tok_src = Tokenizer.from_file(str(teacher_cfg.tokenizer_path(teacher_cfg.data.lang_src)))
    tok_tgt = Tokenizer.from_file(str(teacher_cfg.tokenizer_path(teacher_cfg.data.lang_tgt)))
    pad_id = tok_tgt.token_to_id(PAD_TOKEN)

    scfg = cfg.student
    model = build_lstm_student(
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
        pad_id,
        embed_dim=scfg.embed_dim,
        hidden_dim=scfg.hidden_dim,
        encoder_layers=scfg.encoder_layers,
        decoder_layers=scfg.decoder_layers,
        dropout=0.0,
    ).to(device)

    ckpt_path = cfg.student_weights_path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for text in args.text:
        hi = translate(text, model, tok_src, tok_tgt, seq_len, device)
        print(f"EN: {text}")
        print(f"HI: {hi}")
        print()


if __name__ == "__main__":
    main()
