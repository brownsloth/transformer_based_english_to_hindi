#!/usr/bin/env python3
"""Analyze dataset and recommend seq_len for config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset

from transformer_lib.config import load_config
from transformer_lib.data.tokenization import (
    analyze_max_seq_lengths,
    get_or_build_tokenizer,
    recommended_seq_len,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "en_hi_translation.yaml"))
    args = p.parse_args()

    config = load_config(args.config)
    subset = f"{config.data.lang_src}-{config.data.lang_tgt}"
    ds = load_dataset(config.data.dataset_name, subset, split="train")

    tok_src = get_or_build_tokenizer(config, ds, config.data.lang_src)
    tok_tgt = get_or_build_tokenizer(config, ds, config.data.lang_tgt)

    max_src, max_tgt = analyze_max_seq_lengths(
        ds, tok_src, tok_tgt, config.data.lang_src, config.data.lang_tgt
    )
    rec = recommended_seq_len(max_src, max_tgt)

    print(f"Max source tokens: {max_src}")
    print(f"Max target tokens: {max_tgt}")
    print(f"Recommended seq_len (with buffer): {rec}")
    print(f"Current config seq_len: {config.model.seq_len}")
    if config.model.seq_len < rec:
        print("WARNING: Increase model.seq_len in your YAML config.")


if __name__ == "__main__":
    main()
