#!/usr/bin/env python3
"""
Quantize transformer teacher with several methods and compare BLEU / latency on CPU.

Dynamic INT8 runs on CPU only (PyTorch). Transformer has no LSTM — methods focus on
Linear layers (attention + feed-forward).

Usage:
  python distil/quantize_teacher.py \\
    --teacher-artifacts /workspace/outputs/en_hi \\
    --checkpoint 10

  python distil/quantize_teacher.py \\
    --checkpoint latest \\
    --num-samples 50 \\
    --methods fp32_baseline fp16 dynamic_int8_linear \\
    --save-models
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from transformer_lib.config import load_config
from transformer_lib.data.tokenization import get_translation_dataloaders
from transformer_lib.models.transformer import build_transformer
from transformer_lib.training.decode import greedy_decode


METHODS = (
    "fp32_baseline",
    "fp16",
    "dynamic_int8_linear",
    "static_int8",
)

PHRASES_FILE = ROOT / "distil" / "report" / "phrases_benchmark.txt"


def list_teacher_checkpoints(config) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(config.paths.model_basename)}(\d+)\.pt$")
    found: list[tuple[int, Path]] = []
    for path in config.weights_dir.glob(f"{config.paths.model_basename}*.pt"):
        m = pattern.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found, key=lambda x: x[0])


def resolve_checkpoint(config, checkpoint: str) -> tuple[int, Path]:
    if checkpoint == "latest":
        ckpts = list_teacher_checkpoints(config)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {config.weights_dir}")
        return ckpts[-1]
    path = config.weights_path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    epoch = int(checkpoint) if str(checkpoint).isdigit() else -1
    return epoch, path


def build_teacher_model(config, tok_src: Tokenizer, tok_tgt: Tokenizer, device: torch.device):
    m = config.model
    return build_transformer(
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
        m.seq_len,
        m.seq_len,
        d_model=m.d_model,
        N=m.num_layers,
        h=m.num_heads,
        dropout=m.dropout,
        d_ff=m.d_ff,
    ).to(device)


def model_weight_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.state_dict().values():
        if torch.is_tensor(p):
            total += p.numel() * p.element_size()
    return total


def calibrate_static(model: nn.Module, train_loader, device: torch.device, batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= batches:
                break
            enc = batch["encoder_input"].to(device)
            dec = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)
            enc_out = model.encode(enc, enc_mask)
            dec_out = model.decode(dec, enc_out, dec_mask, enc_mask)
            model.project(dec_out)


def apply_method(model: nn.Module, method: str, train_loader, device: torch.device) -> nn.Module:
    m = copy.deepcopy(model)
    m.eval()
    m.to(device)

    if method == "fp32_baseline":
        return m

    if method == "fp16":
        return m.half()

    if method == "dynamic_int8_linear":
        return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)

    if method == "static_int8":
        backend = "qnnpack" if device.type == "cpu" else "fbgemm"
        torch.backends.quantized.engine = backend
        m.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        prepared = torch.ao.quantization.prepare(m, inplace=True)
        calibrate_static(prepared, train_loader, device, batches=32)
        return torch.ao.quantization.convert(prepared, inplace=True)

    raise ValueError(f"Unknown method: {method}")


def evaluate_bleu(
    model: nn.Module,
    val_loader,
    tok_tgt: Tokenizer,
    seq_len: int,
    device: torch.device,
    num_samples: int,
) -> dict:
    import sacrebleu

    model.eval()
    hyps, refs = [], []
    latencies_ms: list[float] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            ref = batch["tgt_text"][0]

            t0 = time.perf_counter()
            out_ids = greedy_decode(model, enc_in, enc_mask, tok_tgt, seq_len, device)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            hyps.append(tok_tgt.decode(out_ids.cpu().tolist()))
            refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "num_samples": len(hyps),
        "latency_ms_mean": round(sum(latencies_ms) / max(len(latencies_ms), 1), 2),
        "latency_ms_p50": round(sorted(latencies_ms)[len(latencies_ms) // 2], 2) if latencies_ms else 0,
    }


def evaluate_phrases(
    model: nn.Module,
    phrases: list[str],
    tok_src: Tokenizer,
    tok_tgt: Tokenizer,
    seq_len: int,
    device: torch.device,
) -> dict:
    from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN

    sos = tok_src.token_to_id(SOS_TOKEN)
    eos = tok_src.token_to_id(EOS_TOKEN)
    pad = tok_src.token_to_id(PAD_TOKEN)

    model.eval()
    latencies_ms: list[float] = []
    outputs: list[dict] = []

    with torch.no_grad():
        for phrase in phrases:
            ids = tok_src.encode(phrase).ids
            max_body = seq_len - 2
            if len(ids) > max_body:
                ids = ids[:max_body]
            enc = torch.cat(
                [
                    torch.tensor([sos]),
                    torch.tensor(ids),
                    torch.tensor([eos]),
                    torch.tensor([pad] * max(0, seq_len - len(ids) - 2)),
                ]
            )[:seq_len].unsqueeze(0).to(device)
            mask = (enc != pad).int().unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            out_ids = greedy_decode(model, enc, mask, tok_tgt, seq_len, device)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            out_list = out_ids.cpu().tolist()
            sos_t = tok_tgt.token_to_id(SOS_TOKEN)
            eos_t = tok_tgt.token_to_id(EOS_TOKEN)
            pad_t = tok_tgt.token_to_id(PAD_TOKEN)
            out_list = [t for t in out_list if t not in (sos_t, eos_t, pad_t)]
            outputs.append({"en": phrase, "hi": tok_tgt.decode(out_list)})

    return {
        "phrase_count": len(outputs),
        "latency_ms_mean": round(sum(latencies_ms) / max(len(latencies_ms), 1), 2),
        "phrases": outputs,
    }


def save_quantized_model(path: Path, method: str, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"quantization_method": method, "model": model, "meta": meta}, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize transformer teacher and compare quality")
    p.add_argument("--config", default=str(ROOT / "configs" / "runpod_en_hi.yaml"))
    p.add_argument("--teacher-artifacts", default="/workspace/outputs/en_hi")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--num-samples", type=int, default=200, help="Val sentences for BLEU")
    p.add_argument(
        "--eval-mode",
        choices=("bleu", "phrases", "both"),
        default="bleu",
        help="bleu=corpus BLEU; phrases=phrases_benchmark.txt; both=run both",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=METHODS,
    )
    p.add_argument("--output-dir", default=str(ROOT / "distil/outputs/quantized"))
    p.add_argument("--save-models", action="store_true")
    args = p.parse_args()

    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Eval device: {device} (required for INT8 dynamic quant)")

    config = load_config(args.config)
    config.paths.output_dir = args.teacher_artifacts

    epoch, ckpt_path = resolve_checkpoint(config, args.checkpoint)
    print(f"Source checkpoint: {ckpt_path} (epoch {epoch})")

    train_loader, val_loader, tok_src, tok_tgt = get_translation_dataloaders(config)
    seq_len = config.model.seq_len

    phrases: list[str] = []
    if args.eval_mode in ("phrases", "both") and PHRASES_FILE.exists():
        phrases = [
            ln.strip()
            for ln in PHRASES_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    base = build_teacher_model(config, tok_src, tok_tgt, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    base.load_state_dict(state["model_state_dict"])
    base.eval()

    fp32_size_mb = round(model_weight_bytes(base) / 1e6, 2)
    print(f"FP32 weight size: {fp32_size_mb} MB\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"{'method':<28} {'BLEU':>7} {'chrF':>7} {'lat_ms':>8} {'size_MB':>8}  status")
    print("-" * 72)

    for method in args.methods:
        row = {"method": method, "source_checkpoint": str(ckpt_path), "source_epoch": epoch}
        try:
            t0 = time.perf_counter()
            if method == "fp32_baseline":
                qmodel = base
            else:
                qmodel = apply_method(base, method, train_loader, device)

            metrics: dict = {}
            if args.eval_mode in ("bleu", "both"):
                metrics.update(evaluate_bleu(qmodel, val_loader, tok_tgt, seq_len, device, args.num_samples))
            if args.eval_mode in ("phrases", "both") and phrases:
                phrase_metrics = evaluate_phrases(qmodel, phrases, tok_src, tok_tgt, seq_len, device)
                metrics["phrase_latency_ms_mean"] = phrase_metrics["latency_ms_mean"]
                row["phrase_outputs"] = phrase_metrics["phrases"]

            elapsed = time.perf_counter() - t0
            size_mb = round(model_weight_bytes(qmodel) / 1e6, 2)

            row.update(
                {
                    "status": "ok",
                    "size_mb": size_mb,
                    "eval_wall_seconds": round(elapsed, 1),
                    **metrics,
                }
            )
            bleu_s = f"{metrics.get('bleu', 0):7.2f}" if "bleu" in metrics else f"{'n/a':>7}"
            chrf_s = f"{metrics.get('chrf', 0):7.2f}" if "chrf" in metrics else f"{'n/a':>7}"
            lat = metrics.get("latency_ms_mean", metrics.get("phrase_latency_ms_mean", 0))
            print(f"{method:<28} {bleu_s} {chrf_s} {lat:8.2f} {size_mb:8.2f}  ok")

            if args.save_models:
                tag = f"{config.paths.model_basename}{epoch:02d}_{method}.pt"
                save_path = out_dir / "models" / "teacher" / tag
                save_quantized_model(
                    save_path,
                    method,
                    qmodel,
                    {"source": str(ckpt_path), "bleu": metrics.get("bleu"), "size_mb": size_mb},
                )
                row["saved_path"] = str(save_path)

        except Exception as exc:
            row["status"] = f"failed: {exc}"
            print(f"{method:<28} {'—':>7} {'—':>7} {'—':>8} {'—':>8}  FAILED: {exc}")

        rows.append(row)

    print("-" * 72)
    ok_bleu = [r for r in rows if r.get("status") == "ok" and r.get("bleu") is not None]
    if ok_bleu:
        best = max(ok_bleu, key=lambda r: r["bleu"])
        print(f"Best BLEU: {best['method']} ({best['bleu']})")

    stem = f"quant_compare_teacher_{epoch:02d}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    csv_rows = [{k: v for k, v in r.items() if k != "phrase_outputs"} for r in rows]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "teacher_transformer",
                "config": args.config,
                "checkpoint": str(ckpt_path),
                "epoch": epoch,
                "fp32_size_mb": fp32_size_mb,
                "num_samples": args.num_samples,
                "eval_mode": args.eval_mode,
                "device": str(device),
                "results": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "method",
        "status",
        "bleu",
        "chrf",
        "latency_ms_mean",
        "latency_ms_p50",
        "phrase_latency_ms_mean",
        "size_mb",
        "eval_wall_seconds",
        "saved_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    if any(r.get("phrase_outputs") for r in rows):
        phrase_path = out_dir / f"{stem}_phrases.json"
        phrase_path.write_text(
            json.dumps(
                {r["method"]: r.get("phrase_outputs") for r in rows if r.get("status") == "ok"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {phrase_path}")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    if args.save_models:
        print(f"Models: {out_dir / 'models' / 'teacher'}/")


if __name__ == "__main__":
    main()
