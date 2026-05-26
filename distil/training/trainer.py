"""Knowledge distillation trainer: LSTM student + frozen transformer teacher."""

from __future__ import annotations

import logging
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from distil.config import DistilRunConfig
from distil.data import get_distil_dataloaders
from distil.student.lstm_seq2seq import build_lstm_student
from distil.teacher.wrapper import TransformerTeacher
from transformer_lib.config import PAD_TOKEN

logger = logging.getLogger(__name__)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    temperature: float,
    alpha: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined KD (KL on teacher logits) + CE on ground truth."""
    mask = labels != pad_id  # (B, T)

    # CE on hard labels
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )

    # KD: teacher outputs log_softmax; convert for KL(student || teacher)
    T = temperature
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_log_probs / T, dim=-1)

    kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)  # (B, T)
    kl = (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)
    kl = kl * (T * T)

    loss = alpha * kl + (1.0 - alpha) * ce
    return loss, {"loss": loss.item(), "ce": ce.item(), "kl": kl.item()}


class DistillationTrainer:
    def __init__(self, config: DistilRunConfig) -> None:
        self.config = config
        config.ensure_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log_path = config.output_dir / "distill.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )

    def _set_seed(self) -> None:
        s = self.config.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def train(self) -> None:
        self._set_seed()
        cfg = self.config
        tcfg = cfg.training
        dcfg = cfg.distillation

        teacher_cfg = cfg.load_teacher_run_config()
        logger.info("Teacher artifacts: %s", teacher_cfg.output_dir)
        logger.info("Teacher checkpoint: %s", cfg.teacher.checkpoint)

        train_loader, val_loader, tok_src, tok_tgt = get_distil_dataloaders(cfg, teacher_cfg)
        pad_id = tok_tgt.token_to_id(PAD_TOKEN)

        teacher = TransformerTeacher(teacher_cfg, cfg.teacher.checkpoint, self.device)
        scfg = cfg.student
        student = build_lstm_student(
            tok_src.get_vocab_size(),
            tok_tgt.get_vocab_size(),
            pad_id,
            embed_dim=scfg.embed_dim,
            hidden_dim=scfg.hidden_dim,
            encoder_layers=scfg.encoder_layers,
            decoder_layers=scfg.decoder_layers,
            dropout=scfg.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in student.parameters())
        logger.info("Student parameters: %s", f"{n_params:,}")

        optimizer = torch.optim.Adam(student.parameters(), lr=tcfg.lr)
        use_amp = tcfg.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        initial_epoch = 0
        if tcfg.preload_path:
            path = Path(tcfg.preload_path)
            state = torch.load(path, map_location=self.device, weights_only=False)
            student.load_state_dict(state["model_state_dict"])
            logger.info("Fine-tune: loaded student weights from %s", path)
        elif tcfg.preload:
            path = cfg.student_weights_path(tcfg.preload)
            state = torch.load(path, map_location=self.device, weights_only=False)
            student.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            initial_epoch = state["epoch"] + 1
            logger.info("Resumed from %s", path)

        for epoch in range(initial_epoch, tcfg.num_epochs):
            student.train()
            epoch_losses: list[float] = []
            batch_iter = tqdm(train_loader, desc=f"Distill epoch {epoch:02d}")

            for batch in batch_iter:
                enc_in = batch["encoder_input"].to(self.device)
                dec_in = batch["decoder_input"].to(self.device)
                enc_mask = batch["encoder_mask"].to(self.device)
                dec_mask = batch["decoder_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

                with torch.inference_mode():
                    teacher_log_probs = teacher.log_probs(enc_in, dec_in, enc_mask, dec_mask)

                with amp_ctx:
                    student_logits = student(enc_in, dec_in)
                    loss, parts = distillation_loss(
                        student_logits,
                        teacher_log_probs,
                        labels,
                        pad_id,
                        dcfg.temperature,
                        dcfg.alpha,
                        tcfg.label_smoothing,
                    )

                scaler.scale(loss).backward()
                if tcfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(parts["loss"])
                batch_iter.set_postfix(ce=f"{parts['ce']:.3f}", kl=f"{parts['kl']:.3f}")

            avg = sum(epoch_losses) / max(len(epoch_losses), 1)
            logger.info("Epoch %d | avg loss %.4f", epoch, avg)

            if (epoch + 1) % tcfg.save_every_n_epochs == 0:
                self._save(student, optimizer, epoch)

        self._save(student, optimizer, tcfg.num_epochs - 1)

    def _save(self, student: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> Path:
        path = self.config.student_weights_path(epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("Saved student %s", path)
        return path
