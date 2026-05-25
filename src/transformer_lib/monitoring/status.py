"""JSON heartbeat file for remote training progress monitoring."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingStatus:
    state: str  # starting | training | validating | epoch_end | finished | failed
    epoch: int
    global_step: int
    total_steps: int | None
    train_loss: float | None
    lr: float | None
    device: str
    message: str
    updated_at: float
    elapsed_sec: float
    steps_per_sec: float | None = None
    eta_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatusReporter:
    """Writes status.json so you can `ssh vm 'cat outputs/.../status.json'`."""

    def __init__(self, path: Path, run_name: str = "translation") -> None:
        self.path = path
        self.run_name = run_name
        self._start = time.time()
        self._last_step_time = self._start
        self._last_step = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, status: TrainingStatus) -> None:
        payload = {"run_name": self.run_name, **status.to_dict()}
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.path)

    def update(
        self,
        state: str,
        epoch: int,
        global_step: int,
        *,
        total_steps: int | None = None,
        train_loss: float | None = None,
        lr: float | None = None,
        device: str = "cpu",
        message: str = "",
    ) -> None:
        now = time.time()
        elapsed = now - self._start
        steps_per_sec = None
        eta_sec = None

        if global_step > self._last_step:
            dt = now - self._last_step_time
            dstep = global_step - self._last_step
            if dt > 0 and dstep > 0:
                steps_per_sec = dstep / dt
                if total_steps is not None and steps_per_sec > 0:
                    remaining = total_steps - global_step
                    eta_sec = remaining / steps_per_sec
            self._last_step = global_step
            self._last_step_time = now

        self._write(
            TrainingStatus(
                state=state,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
                train_loss=train_loss,
                lr=lr,
                device=device,
                message=message,
                updated_at=now,
                elapsed_sec=elapsed,
                steps_per_sec=steps_per_sec,
                eta_sec=eta_sec,
            )
        )
