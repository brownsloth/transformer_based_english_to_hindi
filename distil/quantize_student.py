#!/usr/bin/env python3
"""
Quantize LSTM student with several methods and compare BLEU / latency on CPU.

Dynamic INT8 runs on CPU only (PyTorch). All variants are evaluated on CPU for
fair comparison.

Usage:
  python distil/quantize_student.py \\
    --config distil/configs/lstm_kd_dict.yaml \\
    --checkpoint latest \\
    --teacher-artifacts /workspace/outputs/en_hi \\
    --dict-max-src-words 3

  python distil/quantize_student.py --checkpoint 14 --save-models
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import load_distil_config
from distil.data import get_distil_dataloaders
from distil.eval_bleu import build_student_model, list_student_checkpoints
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN


METHODS = (
    "fp32_baseline",
    "fp16",
    "dynamic_int8_lstm",
    "dynamic_int8_linear",
    "dynamic_int8_lstm_linear",
    "static_int8",
)


def resolve_checkpoint(cfg, checkpoint: str) -> tuple[int, Path]:
    if checkpoint == "latest":
        ckpts = list_student_checkpoints(cfg)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {cfg.weights_dir}")
        return ckpts[-1]
    path = cfg.student_weights_path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    epoch = int(checkpoint) if str(checkpoint).isdigit() else -1
    return epoch, path


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
            model(enc, dec)


def apply_method(model: nn.Module, method: str, train_loader, device: torch.device) -> nn.Module:
    """Return a model ready for eval (may share structure with input for some methods)."""
    m = copy.deepcopy(model)
    m.eval()
    m.to(device)

    if method == "fp32_baseline":
        return m

    if method == "fp16":
        return m.half()

    if method == "dynamic_int8_lstm":
        return torch.ao.quantization.quantize_dynamic(m, {nn.LSTM}, dtype=torch.qint8)

    if method == "dynamic_int8_linear":
        return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)

    if method == "dynamic_int8_lstm_linear":
        return torch.ao.quantization.quantize_dynamic(m, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

    if method == "static_int8":
        backend = "qnnpack" if device.type == "cpu" else "fbgemm"
        torch.backends.quantized.engine = backend
        m.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        prepared = torch.ao.quantization.prepare(m, inplace=True)
        calibrate_static(prepared, train_loader, device, batches=32)
        return torch.ao.quantization.convert(prepared, inplace=True)

    raise ValueError(f"Unknown method: {method}")


def evaluate_model(
    model: nn.Module,
    val_loader,
    tok_src,
    tok_tgt,
    seq_len: int,
    device: torch.device,
    num_samples: int,
) -> dict:
    import sacrebleu

    pad_id = tok_tgt.token_to_id(PAD_TOKEN)
    sos_id = tok_src.token_to_id(SOS_TOKEN)
    eos_id = tok_src.token_to_id(EOS_TOKEN)

    model.eval()
    hyps, refs = [], []
    latencies_ms: list[float] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            enc = batch["encoder_input"].to(device)
            ref = batch["tgt_text"][0]

            t0 = time.perf_counter()
            out = model.greedy_decode(enc, sos_id, eos_id, seq_len)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            ids = out.cpu().tolist()
            if ids and ids[0] == sos_id:
                ids = ids[1:]
            ids = [t for t in ids if t not in (sos_id, eos_id, pad_id)]
            hyps.append(tok_tgt.decode(ids))
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


def save_quantized_model(path: Path, method: str, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "quantization_method": method,
            "model": model,
            "meta": meta,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize student and compare BLEU")
    p.add_argument("--config", default=str(ROOT / "distil/configs/lstm_kd_dict.yaml"))
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--teacher-artifacts", default=None)
    p.add_argument("--dict-max-src-words", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=METHODS,
        help="Quantization methods to compare",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "distil/outputs/quantized"),
        help="Comparison tables and optional saved models",
    )
    p.add_argument("--save-models", action="store_true", help="Save each quantized model .pt")
    args = p.parse_args()

    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        sys.exit(1)

    # Quantized inference is CPU-only in PyTorch.
    device = torch.device("cpu")
    print(f"Eval device: {device} (required for INT8 dynamic quant)")

    cfg = load_distil_config(args.config)
    if args.teacher_artifacts:
        cfg.teacher.artifacts_dir = args.teacher_artifacts
    if args.dict_max_src_words is not None:
        cfg.data.max_src_words = args.dict_max_src_words

    epoch, ckpt_path = resolve_checkpoint(cfg, args.checkpoint)
    print(f"Source checkpoint: {ckpt_path} (epoch {epoch})")

    teacher_cfg = cfg.load_teacher_run_config()
    train_loader, val_loader, tok_src, tok_tgt = get_distil_dataloaders(cfg, teacher_cfg)
    seq_len = cfg.student.seq_len

    base = build_student_model(cfg, tok_src, tok_tgt, device)
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

            metrics = evaluate_model(qmodel, val_loader, tok_src, tok_tgt, seq_len, device, args.num_samples)
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
            print(
                f"{method:<28} {metrics['bleu']:7.2f} {metrics['chrf']:7.2f} "
                f"{metrics['latency_ms_mean']:8.2f} {size_mb:8.2f}  ok"
            )

            if args.save_models:
                tag = f"{cfg.paths.student_basename}{epoch:02d}_{method}.pt"
                save_path = out_dir / "models" / tag
                save_quantized_model(
                    save_path,
                    method,
                    qmodel,
                    {"source": str(ckpt_path), "bleu": metrics["bleu"], "size_mb": size_mb},
                )
                row["saved_path"] = str(save_path)

        except Exception as exc:
            row["status"] = f"failed: {exc}"
            row["bleu"] = None
            row["chrf"] = None
            print(f"{method:<28} {'—':>7} {'—':>7} {'—':>8} {'—':>8}  FAILED: {exc}")

        rows.append(row)

    print("-" * 72)

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("bleu") is not None]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["bleu"])
        print(f"Best BLEU: {best['method']} ({best['bleu']})")

    stem = f"quant_compare_{cfg.paths.student_basename}{epoch:02d}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": str(ckpt_path),
                "epoch": epoch,
                "fp32_size_mb": fp32_size_mb,
                "num_samples": args.num_samples,
                "device": str(device),
                "results": rows,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "method",
        "status",
        "bleu",
        "chrf",
        "latency_ms_mean",
        "latency_ms_p50",
        "size_mb",
        "eval_wall_seconds",
        "saved_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    if args.save_models:
        print(f"Models: {out_dir / 'models'}/")


if __name__ == "__main__":
    main()
