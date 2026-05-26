#!/usr/bin/env python3
"""Quick phrase eval for dictionary-mode LSTM student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import load_distil_config
from distil.infer import translate
from distil.student.lstm_seq2seq import build_lstm_student
from tokenizers import Tokenizer
from transformer_lib.config import PAD_TOKEN

DEFAULT_PHRASES = [
    "hello",
    "good morning",
    "thank you",
    "good night",
    "water",
    "food",
    "yes",
    "no",
    "please",
    "sorry",
    "how are you",
    "I love you",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Translate fixed short phrases (dictionary eval)")
    p.add_argument("--config", default=str(ROOT / "distil" / "configs" / "lstm_kd_dict.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--teacher-artifacts", default=None)
    p.add_argument("--text", action="append", help="Extra phrase(s) to test")
    args = p.parse_args()

    cfg = load_distil_config(args.config)
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts

    teacher_cfg = cfg.load_teacher_run_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    phrases = DEFAULT_PHRASES + (args.text or [])
    print(f"Checkpoint: {ckpt_path}\n")
    for phrase in phrases:
        hi = translate(phrase, model, tok_src, tok_tgt, seq_len, device)
        print(f"  {phrase:20s}  →  {hi}")


if __name__ == "__main__":
    main()
