"""Webhook alerts for remote training (Slack, Discord, or generic POST)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class AlertManager:
    """Post training alerts to Slack, Discord, or any JSON webhook."""

    def __init__(
        self,
        webhook_url: str | None = None,
        webhook_type: str = "slack",
    ) -> None:
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type.lower()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def _build_payload(self, title: str, message: str) -> dict[str, Any]:
        body = f"*{title}*\n{message}"
        if self.webhook_type == "discord":
            return {"content": f"**{title}**\n{message}"}
        if self.webhook_type == "slack":
            return {"text": body}
        return {"text": body, "title": title, "message": message}

    def send(self, title: str, message: str, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        payload = self._build_payload(title, message)
        if extra:
            payload.update(extra)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    logger.warning("Webhook returned status %s", resp.status)
        except urllib.error.URLError as e:
            logger.warning("Failed to send webhook alert: %s", e)

    def training_started(self, config_summary: str) -> None:
        self.send("Training started", config_summary)

    def epoch_complete(self, epoch: int, loss: float | None, elapsed_h: float) -> None:
        loss_str = f"{loss:.4f}" if loss is not None else "n/a"
        self.send(
            f"Epoch {epoch} complete",
            f"Average loss: {loss_str}\nElapsed: {elapsed_h:.2f}h",
        )

    def training_finished(self, epochs: int, best_loss: float | None) -> None:
        self.send(
            "Training finished",
            f"Completed {epochs} epochs. Final avg loss: {best_loss or 'n/a'}",
        )

    def training_failed(self, error: str) -> None:
        self.send("Training failed", error)
