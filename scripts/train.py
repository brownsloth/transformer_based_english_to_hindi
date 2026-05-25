#!/usr/bin/env python3
"""Train English->Hindi (or any opus-100 pair) Transformer translation model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path when running as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_lib.config import apply_cli_overrides, load_config
from transformer_lib.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer translation model")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "en_hi_translation.yaml"),
        help="Path to YAML config",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Override paths.output_dir")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--preload", type=str, default=None, help="Checkpoint epoch to resume, e.g. 05")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    overrides: dict = {}
    if args.output_dir:
        overrides["paths.output_dir"] = args.output_dir
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["training.num_epochs"] = args.epochs
    if args.lr is not None:
        overrides["training.lr"] = args.lr
    if args.preload is not None:
        overrides["training.preload"] = args.preload
    if args.seq_len is not None:
        overrides["model.seq_len"] = args.seq_len
    if args.no_amp:
        overrides["training.amp"] = False

    if overrides:
        config = apply_cli_overrides(config, overrides)

    webhook = os.environ.get("ALERT_WEBHOOK_URL")
    if webhook:
        config.monitoring.webhook_url = webhook

    webhook_type = os.environ.get("ALERT_WEBHOOK_TYPE")
    if webhook_type:
        config.monitoring.webhook_type = webhook_type

    if os.environ.get("WANDB_PROJECT"):
        config.wandb.project = os.environ["WANDB_PROJECT"]
    if os.environ.get("WANDB_ENTITY"):
        config.wandb.entity = os.environ["WANDB_ENTITY"]
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        config.wandb.enabled = False

    print(f"Output directory: {config.output_dir}")
    print(f"TensorBoard logs: {config.tensorboard_dir}")
    print(f"Status file: {config.status_path}")
    print(f"W&B enabled: {config.wandb.enabled}")
    print(f"Slack/webhook alerts: {bool(config.monitoring.webhook_url)}")

    Trainer(config).train()


if __name__ == "__main__":
    main()
