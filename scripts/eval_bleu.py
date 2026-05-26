#!/usr/bin/env python3
"""Compute corpus BLEU on a validation subset (translation quality metric)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_lib.config import load_config
from transformer_lib.data.tokenization import get_translation_dataloaders
from transformer_lib.models.transformer import build_transformer
from transformer_lib.training.decode import greedy_decode


def main() -> None:
    p = argparse.ArgumentParser(description="BLEU eval on validation set subset")
    p.add_argument("--config", default=str(ROOT / "configs/runpod_en_hi.yaml"))
    p.add_argument("--checkpoint", required=True, help="Epoch index, e.g. 7 or 07")
    p.add_argument("--num-samples", type=int, default=200, help="Val sentences to decode")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    try:
        import sacrebleu
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        sys.exit(1)

    config = load_config(args.config)
    device = torch.device(args.device)

    _, val_loader, tok_src, tok_tgt = get_translation_dataloaders(config)
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
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    hypotheses: list[str] = []
    references: list[str] = []

    print(f"Checkpoint: {ckpt_path}")
    print(f"Evaluating up to {args.num_samples} validation sentences...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=min(args.num_samples, len(val_loader)))):
            if i >= args.num_samples:
                break
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            out_ids = greedy_decode(model, enc_in, enc_mask, tok_tgt, m.seq_len, device)
            pred = tok_tgt.decode(out_ids.cpu().tolist())
            ref = batch["tgt_text"][0]
            hypotheses.append(pred)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    print(f"\nSamples: {len(hypotheses)}")
    print(f"BLEU:  {bleu.score:.2f}")
    print(f"chrF:  {chrf.score:.2f}")
    print("\n--- Example ---")
    print("SRC:", batch["src_text"][0][:120] if hypotheses else "n/a")
    print("REF:", references[-1][:120])
    print("HYP:", hypotheses[-1][:120])


if __name__ == "__main__":
    main()
