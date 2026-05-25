"""TensorBoard logging with hyperparameter tracking."""

from __future__ import annotations

from typing import Any

from torch.utils.tensorboard import SummaryWriter

from transformer_lib.config import Config


class TensorBoardLogger:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.writer: SummaryWriter | None = None

        if config.tensorboard.enabled:
            self.writer = SummaryWriter(
                log_dir=str(config.tensorboard_dir),
                flush_secs=config.tensorboard.flush_secs,
            )

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        if not self.enabled or not self.config.tensorboard.log_hparams:
            return
        try:
            self.writer.add_hparams(hparams, metrics or {})
        except Exception:
            # Some metric keys required; skip if incompatible
            pass

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer:
            self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
