#!/usr/bin/env python3
"""Distill transformer teacher → LSTM student (Path D + teacher logits)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import DistilRunConfig, load_distil_config
from distil.training.trainer import DistillationTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Knowledge distillation: Transformer → LSTM")
    p.add_argument(
        "--config",
        default=str(ROOT / "distil" / "configs" / "lstm_kd.yaml"),
    )
    p.add_argument("--teacher-checkpoint", default=None)
    p.add_argument("--teacher-artifacts", default=None, help="Path to outputs/en_hi")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None, help="KD weight (0-1)")
    p.add_argument("--preload", default=None)
    return p.parse_args()


def apply_overrides(cfg: DistilRunConfig, args: argparse.Namespace) -> DistilRunConfig:
    if args.teacher_checkpoint:
        cfg.teacher.checkpoint = args.teacher_checkpoint
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.alpha is not None:
        cfg.distillation.alpha = args.alpha
    if args.preload is not None:
        cfg.training.preload = args.preload
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_distil_config(args.config)
    cfg = apply_overrides(cfg, args)

    print(f"Student output: {cfg.output_dir}")
    print(f"Teacher checkpoint: {cfg.teacher.checkpoint}")
    print(f"KD alpha={cfg.distillation.alpha} T={cfg.distillation.temperature}")

    DistillationTrainer(cfg).train()


if __name__ == "__main__":
    main()
