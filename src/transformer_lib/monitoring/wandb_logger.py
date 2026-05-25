"""Optional Weights & Biases logging (runs alongside TensorBoard)."""

from __future__ import annotations

import logging
from typing import Any

from transformer_lib.config import Config

logger = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.run = None
        wcfg = config.wandb

        if not wcfg.enabled:
            return

        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; pip install wandb or disable wandb.enabled")
            return

        run_name = wcfg.run_name or f"{config.data.lang_src}-{config.data.lang_tgt}"
        self.run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity or None,
            name=run_name,
            config=config.to_dict(),
            dir=str(config.output_dir),
            resume="allow",
        )
        logger.info("W&B run: %s", self.run.url if self.run else "n/a")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled:
            import wandb

            wandb.log({tag: value}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.enabled:
            import wandb

            wandb.log({tag: text}, step=step)

    def log_validation_samples(
        self,
        step: int,
        rows: list[dict[str, str]],
    ) -> None:
        if not self.enabled or not self.config.wandb.log_val_samples or not rows:
            return
        import wandb

        table = wandb.Table(columns=["source", "target", "predicted"])
        for r in rows:
            table.add_data(r["source"], r["target"], r["predicted"])
        wandb.log({"val/translations": table}, step=step)

    def finish(self) -> None:
        if self.enabled:
            import wandb

            wandb.finish()
