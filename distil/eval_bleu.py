#!/usr/bin/env python3
"""BLEU eval for distilled LSTM student."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import DistilRunConfig, load_distil_config
from distil.data import get_distil_dataloaders
from distil.student.lstm_seq2seq import build_lstm_student
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN


def list_student_checkpoints(cfg: DistilRunConfig) -> list[tuple[int, Path]]:
    """Return (epoch, path) sorted by epoch for all student weight files."""
    weights_dir = cfg.weights_dir
    pattern = re.compile(rf"^{re.escape(cfg.paths.student_basename)}(\d+)\.pt$")
    found: list[tuple[int, Path]] = []
    for path in weights_dir.glob(f"{cfg.paths.student_basename}*.pt"):
        m = pattern.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found, key=lambda x: x[0])


def build_student_model(cfg: DistilRunConfig, tok_src, tok_tgt, device: torch.device):
    pad_id = tok_tgt.token_to_id(PAD_TOKEN)
    scfg = cfg.student
    return build_lstm_student(
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
        pad_id,
        embed_dim=scfg.embed_dim,
        hidden_dim=scfg.hidden_dim,
        encoder_layers=scfg.encoder_layers,
        decoder_layers=scfg.decoder_layers,
        dropout=0.0,
    ).to(device)


def evaluate_checkpoint(
    model,
    val_loader,
    tok_src,
    tok_tgt,
    seq_len: int,
    device: torch.device,
    num_samples: int,
) -> tuple[float, float]:
    import sacrebleu

    pad_id = tok_tgt.token_to_id(PAD_TOKEN)
    sos_id = tok_src.token_to_id(SOS_TOKEN)
    eos_id = tok_src.token_to_id(EOS_TOKEN)

    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            enc = batch["encoder_input"].to(device)
            out = model.greedy_decode(enc, sos_id, eos_id, seq_len)
            ids = out.cpu().tolist()
            # Drop SOS/EOS/pad so hypotheses match reference text style.
            ids = [t for t in ids if t not in (sos_id, eos_id, pad_id)]
            hyps.append(tok_tgt.decode(ids))
            refs.append(batch["tgt_text"][0])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return bleu.score, chrf.score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "distil" / "configs" / "lstm_kd.yaml"))
    p.add_argument("--checkpoint", default=None, help="Epoch index, e.g. 6 or 06")
    p.add_argument("--all-checkpoints", action="store_true", help="Eval every lstm_*.pt")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--teacher-artifacts", default=None)
    args = p.parse_args()

    if not args.all_checkpoints and args.checkpoint is None:
        p.error("Provide --checkpoint N or --all-checkpoints")

    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        sys.exit(1)

    cfg = load_distil_config(args.config)
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts

    teacher_cfg = cfg.load_teacher_run_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, tok_src, tok_tgt = get_distil_dataloaders(cfg, teacher_cfg)
    model = build_student_model(cfg, tok_src, tok_tgt, device)
    seq_len = cfg.student.seq_len

    if args.all_checkpoints:
        checkpoints = list_student_checkpoints(cfg)
        if not checkpoints:
            print(f"No checkpoints in {cfg.weights_dir}")
            sys.exit(1)

        print(f"Evaluating {len(checkpoints)} checkpoints ({args.num_samples} val samples each)\n")
        print(f"{'epoch':>5}  {'BLEU':>7}  {'chrF':>7}  checkpoint")
        print("-" * 52)

        best_bleu, best_epoch = -1.0, -1
        for epoch, path in checkpoints:
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            bleu, chrf = evaluate_checkpoint(
                model, val_loader, tok_src, tok_tgt, seq_len, device, args.num_samples
            )
            print(f"{epoch:5d}  {bleu:7.2f}  {chrf:7.2f}  {path.name}")
            if bleu > best_bleu:
                best_bleu, best_epoch = bleu, epoch

        print("-" * 52)
        print(f"Best: epoch {best_epoch}  BLEU {best_bleu:.2f}")
        return

    ckpt_path = cfg.student_weights_path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    bleu, chrf = evaluate_checkpoint(
        model, val_loader, tok_src, tok_tgt, seq_len, device, args.num_samples
    )
    print(f"Checkpoint: {ckpt_path}")
    print(f"Samples: {min(args.num_samples, len(val_loader.dataset))}")
    print(f"BLEU: {bleu:.2f}")
    print(f"chrF: {chrf:.2f}")


if __name__ == "__main__":
    main()
