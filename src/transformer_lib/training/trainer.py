"""Main training loop with AMP, checkpointing, and monitoring."""

from __future__ import annotations

import logging
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from transformer_lib.config import PAD_TOKEN, Config
from transformer_lib.data.tokenization import get_translation_dataloaders
from transformer_lib.models.transformer import Transformer, build_transformer
from transformer_lib.monitoring.alerts import AlertManager
from transformer_lib.monitoring.status import StatusReporter
from transformer_lib.monitoring.tensorboard_logger import TensorBoardLogger
from transformer_lib.monitoring.wandb_logger import WandbLogger
from transformer_lib.training.validation import run_validation

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        config.save_json()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        mon = config.monitoring
        self.alerts = AlertManager(mon.webhook_url, webhook_type=mon.webhook_type)
        self.status = StatusReporter(
            config.status_path,
            run_name=f"{config.data.lang_src}-{config.data.lang_tgt}",
        )
        self.tb = TensorBoardLogger(config)
        self.wandb = WandbLogger(config)

        self._setup_logging()

    def _setup_logging(self) -> None:
        log_path = self.config.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
        )

    def _set_seed(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(
        self, src_vocab: int, tgt_vocab: int
    ) -> Transformer:
        m = self.config.model
        return build_transformer(
            src_vocab,
            tgt_vocab,
            m.seq_len,
            m.seq_len,
            d_model=m.d_model,
            N=m.num_layers,
            h=m.num_heads,
            dropout=m.dropout,
            d_ff=m.d_ff,
        )

    def _load_checkpoint(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, int]:
        preload = self.config.training.preload
        if not preload:
            return 0, 0

        path = self.config.weights_path(preload)
        logger.info("Loading checkpoint %s", path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"] + 1, state["global_step"]

    def train(self) -> None:
        self._set_seed()
        cfg = self.config
        tcfg = cfg.training

        if cfg.monitoring.alert_on_start and self.alerts.enabled:
            self.alerts.training_started(
                f"{cfg.data.lang_src}->{cfg.data.lang_tgt} | "
                f"epochs={tcfg.num_epochs} batch={cfg.data.batch_size} "
                f"seq_len={cfg.model.seq_len} output={cfg.output_dir}"
            )

        train_loader, val_loader, tok_src, tok_tgt = get_translation_dataloaders(cfg)
        model = self._build_model(
            tok_src.get_vocab_size(), tok_tgt.get_vocab_size()
        ).to(self.device)

        pad_id = tok_tgt.token_to_id(PAD_TOKEN)
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_id, label_smoothing=tcfg.label_smoothing
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr, eps=1e-9)
        use_amp = tcfg.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        initial_epoch, global_step = self._load_checkpoint(model, optimizer)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * tcfg.num_epochs

        hparams = {
            "lr": tcfg.lr,
            "batch_size": cfg.data.batch_size,
            "d_model": cfg.model.d_model,
            "num_layers": cfg.model.num_layers,
            "seq_len": cfg.model.seq_len,
        }
        self.tb.log_hparams(hparams)

        self.status.update(
            "starting",
            initial_epoch,
            global_step,
            total_steps=total_steps,
            device=str(self.device),
            message="Training started",
        )

        logger.info("Device: %s | Steps/epoch: %d", self.device, steps_per_epoch)

        try:
            for epoch in range(initial_epoch, tcfg.num_epochs):
                model.train()
                epoch_losses: list[float] = []
                batch_iter = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch:02d}",
                    leave=True,
                )

                for batch in batch_iter:
                    enc_in = batch["encoder_input"].to(self.device)
                    dec_in = batch["decoder_input"].to(self.device)
                    enc_mask = batch["encoder_mask"].to(self.device)
                    dec_mask = batch["decoder_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    amp_ctx = (
                        torch.amp.autocast("cuda") if scaler.is_enabled() else nullcontext()
                    )
                    with amp_ctx:
                        enc_out = model.encode(enc_in, enc_mask)
                        dec_out = model.decode(dec_in, enc_out, dec_mask, enc_mask)
                        proj = model.project(dec_out)
                        loss = loss_fn(
                            proj.view(-1, tok_tgt.get_vocab_size()),
                            labels.view(-1),
                        )

                    scaler.scale(loss).backward()

                    if tcfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    batch_iter.set_postfix(loss=f"{loss_val:.4f}")

                    self.tb.log_scalar("train/loss", loss_val, global_step)
                    self.tb.log_scalar("train/lr", tcfg.lr, global_step)
                    if global_step % cfg.wandb.log_every_n_steps == 0:
                        self.wandb.log_scalar("train/loss", loss_val, global_step)
                        self.wandb.log_scalar("train/lr", tcfg.lr, global_step)

                    if global_step % cfg.monitoring.heartbeat_every_n_steps == 0:
                        self.status.update(
                            "training",
                            epoch,
                            global_step,
                            total_steps=total_steps,
                            train_loss=loss_val,
                            lr=tcfg.lr,
                            device=str(self.device),
                        )

                    if (
                        global_step > 0
                        and global_step % tcfg.val_every_n_steps == 0
                    ):
                        self.status.update(
                            "validating",
                            epoch,
                            global_step,
                            device=str(self.device),
                            message="Running validation",
                        )
                        _, val_samples = run_validation(
                            model,
                            val_loader,
                            tok_tgt,
                            cfg.model.seq_len,
                            self.device,
                            print_msg=batch_iter.write,
                            writer=self.tb.writer,
                            global_step=global_step,
                            num_examples=tcfg.val_num_examples,
                        )
                        self.wandb.log_validation_samples(global_step, val_samples)

                    global_step += 1

                avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                self.tb.log_scalar("train/epoch_loss", avg_loss, epoch)
                self.wandb.log_scalar("train/epoch_loss", avg_loss, global_step)
                self.tb.flush()

                _, val_samples = run_validation(
                    model,
                    val_loader,
                    tok_tgt,
                    cfg.model.seq_len,
                    self.device,
                    print_msg=logger.info,
                    writer=self.tb.writer,
                    global_step=global_step,
                    num_examples=tcfg.val_num_examples,
                )
                self.wandb.log_validation_samples(global_step, val_samples)

                if (epoch + 1) % tcfg.save_every_n_epochs == 0:
                    self._save_checkpoint(model, optimizer, epoch, global_step)

                elapsed_h = (time.time() - self.status._start) / 3600
                self.status.update(
                    "epoch_end",
                    epoch,
                    global_step,
                    train_loss=avg_loss,
                    device=str(self.device),
                    message=f"Epoch {epoch} avg loss {avg_loss:.4f}",
                )

                if cfg.monitoring.alert_on_epoch_end and self.alerts.enabled:
                    self.alerts.epoch_complete(epoch, avg_loss, elapsed_h)

                logger.info("Epoch %d complete | avg loss: %.4f", epoch, avg_loss)

            self.status.update(
                "finished",
                tcfg.num_epochs - 1,
                global_step,
                device=str(self.device),
                message="Training finished successfully",
            )
            if cfg.monitoring.alert_on_finish and self.alerts.enabled:
                msg = f"Completed {tcfg.num_epochs} epochs. Final avg loss: {avg_loss:.4f}"
                if self.wandb.enabled and self.wandb.run is not None:
                    msg += f"\nW&B: {self.wandb.run.url}"
                msg += f"\nOutputs: {cfg.output_dir}"
                self.alerts.send("Training finished", msg)

        except Exception as e:
            self.status.update(
                "failed",
                0,
                global_step,
                device=str(self.device),
                message=str(e),
            )
            if self.alerts.enabled:
                self.alerts.training_failed(str(e))
            raise
        finally:
            self.tb.close()
            self.wandb.finish()

    def _save_checkpoint(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
    ) -> Path:
        path = self.config.weights_path(epoch)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint %s", path)
        return path
