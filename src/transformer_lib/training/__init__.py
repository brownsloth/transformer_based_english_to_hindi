"""Training utilities (Trainer imported lazily to avoid heavy deps at import time)."""

__all__ = ["Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        from transformer_lib.training.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

