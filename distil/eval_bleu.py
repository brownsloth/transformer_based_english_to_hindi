#!/usr/bin/env python3
"""BLEU eval for distilled LSTM student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import load_distil_config
from distil.data import get_distil_dataloaders
from distil.student.lstm_seq2seq import build_lstm_student
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "distil" / "configs" / "lstm_kd.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--teacher-artifacts", default=None)
    args = p.parse_args()

    import sacrebleu

    cfg = load_distil_config(args.config)
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts

    teacher_cfg = cfg.load_teacher_run_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, tok_src, tok_tgt = get_distil_dataloaders(cfg, teacher_cfg)
    pad_id = tok_tgt.token_to_id(PAD_TOKEN)
    sos_id = tok_src.token_to_id(SOS_TOKEN)
    eos_id = tok_src.token_to_id(EOS_TOKEN)

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

    ckpt = torch.load(cfg.student_weights_path(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hyps, refs = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=args.num_samples)):
            if i >= args.num_samples:
                break
            enc = batch["encoder_input"].unsqueeze(0).to(device)
            out = model.greedy_decode(enc, sos_id, eos_id, scfg.seq_len)
            hyps.append(tok_tgt.decode(out.cpu().tolist()))
            refs.append(batch["tgt_text"][0])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    print(f"Checkpoint: {cfg.student_weights_path(args.checkpoint)}")
    print(f"Samples: {len(hyps)}")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")


if __name__ == "__main__":
    main()
