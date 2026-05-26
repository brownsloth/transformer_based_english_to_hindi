"""Distillation config (teacher transformer + LSTM student)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from transformer_lib.config import load_config as load_teacher_config


@dataclass
class TeacherConfig:
    yaml_path: str = "configs/runpod_en_hi.yaml"
    checkpoint: str = "8"
    artifacts_dir: str | None = None  # override paths.output_dir in teacher yaml


@dataclass
class StudentConfig:
    embed_dim: int = 128
    hidden_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 128


@dataclass
class DistillConfig:
    temperature: float = 3.0
    alpha: float = 0.6  # weight on KD; (1-alpha) on CE


@dataclass
class DistilDataConfig:
    dataset_name: str = "Helsinki-NLP/opus-100"
    lang_src: str = "en"
    lang_tgt: str = "hi"
    train_split_ratio: float = 0.9
    batch_size: int = 128
    num_workers: int = 8
    truncate_long: bool = True
    max_train_samples: int | None = None
    # Keep pairs where source/target word counts are within these bounds (dictionary mode).
    min_src_words: int | None = None
    max_src_words: int | None = None
    max_tgt_words: int | None = None


@dataclass
class DistilTrainingConfig:
    num_epochs: int = 25
    lr: float = 1e-3
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    save_every_n_epochs: int = 1
    preload: str | None = None
    preload_path: str | None = None  # full path to .pt; loads weights only (fine-tune)
    val_bleu_samples: int = 200


@dataclass
class DistilPathsConfig:
    output_dir: str = "distil/outputs/lstm_kd"
    student_basename: str = "lstm_"


@dataclass
class DistilRunConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distillation: DistillConfig = field(default_factory=DistillConfig)
    data: DistilDataConfig = field(default_factory=DistilDataConfig)
    training: DistilTrainingConfig = field(default_factory=DistilTrainingConfig)
    paths: DistilPathsConfig = field(default_factory=DistilPathsConfig)
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    def student_weights_path(self, epoch: int | str) -> Path:
        if isinstance(epoch, str) and epoch.isdigit():
            epoch = int(epoch)
        name = f"{self.paths.student_basename}{epoch:02d}.pt" if isinstance(epoch, int) else f"{self.paths.student_basename}{epoch}.pt"
        return self.weights_dir / name

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def load_teacher_run_config(self):
        cfg = load_teacher_config(self.teacher.yaml_path)
        if self.teacher.artifacts_dir:
            cfg.paths.output_dir = self.teacher.artifacts_dir
        return cfg


def load_distil_config(path: str | Path) -> DistilRunConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def sec(name: str, cls: type):
        return cls(**raw.get(name, {}))

    return DistilRunConfig(
        teacher=sec("teacher", TeacherConfig),
        student=sec("student", StudentConfig),
        distillation=sec("distillation", DistillConfig),
        data=sec("data", DistilDataConfig),
        training=sec("training", DistilTrainingConfig),
        paths=sec("paths", DistilPathsConfig),
        seed=raw.get("seed", 42),
    )
