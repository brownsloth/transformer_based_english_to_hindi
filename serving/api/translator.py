"""Load fast (LSTM student) and accurate (quantized teacher) backends."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.student.lstm_seq2seq import build_lstm_student
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN
from transformer_lib.models.transformer import build_transformer
from transformer_lib.training.decode import greedy_decode

TranslateMode = Literal["fast", "accurate"]


@dataclass
class ServeConfig:
    artifacts_dir: Path
    lang_src: str
    lang_tgt: str
    tokenizer_pattern: str
    fast: dict
    accurate: dict

    @classmethod
    def load(cls, path: Path) -> ServeConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        paths = raw.get("paths", {})
        artifacts = Path(
            os.environ.get("SERVE_ARTIFACTS_DIR", paths.get("artifacts_dir", "serving/artifacts"))
        )
        if not artifacts.is_absolute():
            artifacts = ROOT / artifacts
        data = raw.get("data", {})
        return cls(
            artifacts_dir=artifacts,
            lang_src=data.get("lang_src", "en"),
            lang_tgt=data.get("lang_tgt", "hi"),
            tokenizer_pattern=paths.get("tokenizer_pattern", "tokenizers/tokenizer_{lang}.json"),
            fast=raw.get("fast", {}),
            accurate=raw.get("accurate", {}),
        )

    def tokenizer_path(self, lang: str) -> Path:
        return self.artifacts_dir / self.tokenizer_pattern.format(lang=lang)

    def weights_path(self, mode: TranslateMode) -> Path:
        section = self.fast if mode == "fast" else self.accurate
        return self.artifacts_dir / section["weights"]


def _encode_source(
    text: str,
    tok_src: Tokenizer,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sos = tok_src.token_to_id(SOS_TOKEN)
    eos = tok_src.token_to_id(EOS_TOKEN)
    pad = tok_src.token_to_id(PAD_TOKEN)

    ids = tok_src.encode(text.strip()).ids
    max_body = seq_len - 2
    if len(ids) > max_body:
        ids = ids[:max_body]

    enc = torch.cat(
        [
            torch.tensor([sos]),
            torch.tensor(ids, dtype=torch.int64),
            torch.tensor([eos]),
            torch.tensor([pad] * max(0, seq_len - len(ids) - 2), dtype=torch.int64),
        ]
    )[:seq_len].unsqueeze(0).to(device)
    mask = (enc != pad).int().unsqueeze(0).unsqueeze(0).to(device)
    return enc, mask


class _FastBackend:
    def __init__(self, cfg: ServeConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.seq_len = int(cfg.fast["seq_len"])
        self.description = cfg.fast.get("description", "LSTM student")

        tok_src_path = cfg.tokenizer_path(cfg.lang_src)
        tok_tgt_path = cfg.tokenizer_path(cfg.lang_tgt)
        self.tok_src = Tokenizer.from_file(str(tok_src_path))
        self.tok_tgt = Tokenizer.from_file(str(tok_tgt_path))

        pad_id = self.tok_tgt.token_to_id(PAD_TOKEN)
        scfg = cfg.fast
        self.model = build_lstm_student(
            self.tok_src.get_vocab_size(),
            self.tok_tgt.get_vocab_size(),
            pad_id,
            embed_dim=int(scfg["embed_dim"]),
            hidden_dim=int(scfg["hidden_dim"]),
            encoder_layers=int(scfg["encoder_layers"]),
            decoder_layers=int(scfg["decoder_layers"]),
            dropout=0.0,
        )

        weights_path = cfg.weights_path("fast")
        if not weights_path.exists():
            raise FileNotFoundError(f"Fast weights not found: {weights_path}")

        payload = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.half()
        else:
            raise ValueError(f"Unrecognized fast checkpoint format: {weights_path}")

        self.model.eval().to(device)

        self.sos = self.tok_src.token_to_id(SOS_TOKEN)
        self.eos = self.tok_src.token_to_id(EOS_TOKEN)
        self.pad = self.tok_tgt.token_to_id(PAD_TOKEN)

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        enc, _ = _encode_source(text, self.tok_src, self.seq_len, self.device)
        out = self.model.greedy_decode(enc, self.sos, self.eos, self.seq_len)
        ids = [t for t in out.cpu().tolist() if t not in (self.sos, self.eos, self.pad)]
        return self.tok_tgt.decode(ids)


class _AccurateBackend:
    def __init__(self, cfg: ServeConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.description = cfg.accurate.get("description", "Transformer teacher")

        if device.type == "cpu":
            torch.backends.quantized.engine = "qnnpack"

        tok_src_path = cfg.tokenizer_path(cfg.lang_src)
        tok_tgt_path = cfg.tokenizer_path(cfg.lang_tgt)
        self.tok_src = Tokenizer.from_file(str(tok_src_path))
        self.tok_tgt = Tokenizer.from_file(str(tok_tgt_path))

        acfg = cfg.accurate
        self.seq_len = int(acfg["seq_len"])

        weights_path = cfg.weights_path("accurate")
        if not weights_path.exists():
            raise FileNotFoundError(f"Accurate weights not found: {weights_path}")

        payload = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
        else:
            self.model = build_transformer(
                self.tok_src.get_vocab_size(),
                self.tok_tgt.get_vocab_size(),
                self.seq_len,
                self.seq_len,
                d_model=int(acfg["d_model"]),
                N=int(acfg["num_layers"]),
                h=int(acfg["num_heads"]),
                dropout=float(acfg.get("dropout", 0.1)),
                d_ff=int(acfg["d_ff"]),
            )
            state = payload["model_state_dict"] if isinstance(payload, dict) else payload
            self.model.load_state_dict(state)

        self.model.eval().to(device)

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        enc, mask = _encode_source(text, self.tok_src, self.seq_len, self.device)
        out_ids = greedy_decode(self.model, enc, mask, self.tok_tgt, self.seq_len, self.device)
        return self.tok_tgt.decode(out_ids.cpu().tolist())


class TranslatorService:
    def __init__(self) -> None:
        config_path = os.environ.get(
            "SERVE_CONFIG",
            str(ROOT / "serving" / "config" / "serve.yaml"),
        )
        self.cfg = ServeConfig.load(Path(config_path))
        self.device = torch.device(
            os.environ.get("SERVE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.backends: dict[TranslateMode, _FastBackend | _AccurateBackend] = {
            "fast": _FastBackend(self.cfg, self.device),
            "accurate": _AccurateBackend(self.cfg, self.device),
        }

    def translate(self, text: str, mode: TranslateMode = "fast") -> dict:
        if mode not in self.backends:
            raise ValueError(f"Unknown mode: {mode}")

        backend = self.backends[mode]
        t0 = time.perf_counter()
        translation = backend.translate(text)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "translation": translation,
            "latency_ms": latency_ms,
            "mode": mode,
        }

    def health(self) -> dict:
        return {
            "device": str(self.device),
            "modes": {
                name: {
                    "description": backend.description,
                    "weights": str(self.cfg.weights_path(name)),
                }
                for name, backend in self.backends.items()
            },
        }
