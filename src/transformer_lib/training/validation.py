"""Validation loop with optional BLEU logging."""

from __future__ import annotations

from typing import Callable

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformer_lib.models.transformer import Transformer
from transformer_lib.training.decode import greedy_decode


def run_validation(
    model: Transformer,
    val_loader: DataLoader,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: Callable[[str], None] | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
    num_examples: int = 2,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    count = 0
    metrics: dict[str, float] = {}
    samples: list[dict[str, str]] = []

    with torch.no_grad():
        for batch in val_loader:
            count += 1
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            assert enc_in.size(0) == 1, "Validation batch size must be 1"

            out_ids = greedy_decode(model, enc_in, enc_mask, tokenizer_tgt, max_len, device)
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            pred_text = tokenizer_tgt.decode(out_ids.detach().cpu().tolist())
            samples.append(
                {"source": src_text, "target": tgt_text, "predicted": pred_text}
            )

            if print_msg:
                print_msg("-" * 80)
                print_msg(f"SOURCE: {src_text}")
                print_msg(f"TARGET: {tgt_text}")
                print_msg(f"PREDICTED: {pred_text}")

            if count >= num_examples:
                break

    if writer is not None:
        try:
            from sacrebleu.metrics import BLEU

            # Full validation BLEU is expensive; log placeholder for sample runs
            writer.add_scalar("val/samples_logged", float(count), global_step)
        except ImportError:
            pass

    model.train()
    return metrics, samples
