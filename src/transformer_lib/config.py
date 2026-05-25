"""Configuration loading from YAML and CLI overrides."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

UNK_TOKEN = "<" + "UNK" + ">"
PAD_TOKEN = "<" + "PAD" + ">"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "<" + "EOS" + ">"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]


@dataclass
class ModelConfig:
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    seq_len: int = 450


@dataclass
class DataConfig:
    dataset_name: str = "Helsinki-NLP/opus-100"
    lang_src: str = "en"
    lang_tgt: str = "hi"
    train_split_ratio: float = 0.9
    batch_size: int = 8
    num_workers: int = 2
    max_train_samples: int | None = None  # limit rows for local smoke tests
    truncate_long: bool = False  # truncate instead of error when seq_len is too small


@dataclass
class TrainingConfig:
    num_epochs: int = 20
    lr: float = 1e-4
    label_smoothing: float = 0.1
    amp: bool = True
    grad_clip: float = 1.0
    val_every_n_steps: int = 500
    val_num_examples: int = 2
    save_every_n_epochs: int = 1
    preload: str | None = None


@dataclass
class PathsConfig:
    output_dir: str = "outputs/en_hi"
    model_basename: str = "tmodel_"
    tokenizer_pattern: str = "tokenizers/tokenizer_{lang}.json"


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = "runs"
    flush_secs: int = 30
    log_hparams: bool = True


@dataclass
class MonitoringConfig:
    status_file: str = "status.json"
    heartbeat_every_n_steps: int = 50
    log_file: str = "train.log"
    webhook_url: str | None = None
    webhook_type: str = "slack"  # slack | discord | generic
    alert_on_epoch_end: bool = True
    alert_on_start: bool = True
    alert_on_finish: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "transformer-en-hi"
    entity: str | None = None
    run_name: str | None = None
    log_every_n_steps: int = 10
    log_val_samples: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    @property
    def tokenizers_dir(self) -> Path:
        return self.output_dir / "tokenizers"

    @property
    def tensorboard_dir(self) -> Path:
        return self.output_dir / self.tensorboard.log_dir

    @property
    def status_path(self) -> Path:
        return self.output_dir / self.monitoring.status_file

    @property
    def log_path(self) -> Path:
        return self.output_dir / self.monitoring.log_file

    def tokenizer_path(self, lang: str) -> Path:
        pattern = self.paths.tokenizer_pattern.replace("{lang}", lang)
        if "{0}" in pattern:
            pattern = pattern.format(lang)
        return self.output_dir / pattern

    def weights_path(self, epoch: int | str) -> Path:
        if isinstance(epoch, str) and epoch.isdigit():
            epoch = int(epoch)
        if isinstance(epoch, int):
            name = f"{self.paths.model_basename}{epoch:02d}.pt"
        else:
            name = f"{self.paths.model_basename}{epoch}.pt"
        return self.weights_dir / name

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizers_dir.mkdir(parents=True, exist_ok=True)
        if self.tensorboard.enabled:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | None = None) -> Path:
        path = path or (self.output_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _merge_dict(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def section(name: str, cls: type):
        return cls(**raw.get(name, {}))

    return Config(
        model=section("model", ModelConfig),
        data=section("data", DataConfig),
        training=section("training", TrainingConfig),
        paths=section("paths", PathsConfig),
        tensorboard=section("tensorboard", TensorBoardConfig),
        monitoring=section("monitoring", MonitoringConfig),
        wandb=section("wandb", WandbConfig),
        seed=raw.get("seed", 42),
    )


def apply_cli_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Apply flat overrides like {'data.batch_size': 16}."""
    data = config.to_dict()
    for key, value in overrides.items():
        parts = key.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return Config(
        model=ModelConfig(**data["model"]),
        data=DataConfig(**data["data"]),
        training=TrainingConfig(**data["training"]),
        paths=PathsConfig(**data["paths"]),
        tensorboard=TensorBoardConfig(**data["tensorboard"]),
        monitoring=MonitoringConfig(**data["monitoring"]),
        wandb=WandbConfig(**data.get("wandb", {})),
        seed=data.get("seed", 42),
    )
