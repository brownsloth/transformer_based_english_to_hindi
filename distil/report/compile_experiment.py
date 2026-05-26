#!/usr/bin/env python3
"""
Compile a full distillation experiment export for blogging / continuation.

Run on RunPod after training finishes. Produces metrics, loss curves, phrase
benchmarks (teacher + all student checkpoints), val examples, and a tarball.

Usage (on pod):
  bash distil/report/run_on_runpod.sh

  # or
  python distil/report/compile_experiment.py \\
    --teacher-artifacts /workspace/outputs/en_hi \\
    --teacher-checkpoint 10 \\
    --tarball
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from distil.config import load_distil_config
from distil.eval_bleu import (
    build_student_model,
    evaluate_checkpoint,
    list_student_checkpoints,
)
from distil.infer import encode_source, translate
from distil.data import get_distil_dataloaders
from transformer_lib.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, load_config
from transformer_lib.models.transformer import build_transformer
from transformer_lib.training.decode import greedy_decode

from tokenizers import Tokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def collect_metadata(out: Path) -> dict:
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    nvidia = run_cmd(["nvidia-smi"])
    (meta_dir / "nvidia_smi.txt").write_text(nvidia or "(nvidia-smi unavailable)\n", encoding="utf-8")

    info = {
        "collected_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)

    for name, cmd in [
        ("df_h.txt", ["df", "-h"]),
        ("free_h.txt", ["free", "-h"]),
        ("uname_a.txt", ["uname", "-a"]),
    ]:
        text = run_cmd(cmd)
        if text:
            (meta_dir / name).write_text(text, encoding="utf-8")

    pip = run_cmd([sys.executable, "-m", "pip", "freeze"])
    (meta_dir / "pip_freeze.txt").write_text(pip or "", encoding="utf-8")
    (meta_dir / "system.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def parse_distill_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"Epoch (\d+) \| avg loss ([\d.]+)", line)
        if m:
            rows.append({"epoch": int(m.group(1)), "avg_loss": float(m.group(2))})
    return rows


def export_loss_curves(out: Path, runs: dict[str, Path]) -> dict[str, list[dict]]:
    train_dir = out / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    all_losses: dict[str, list[dict]] = {}

    for name, log_path in runs.items():
        rows = parse_distill_log(log_path)
        all_losses[name] = rows
        if not rows:
            continue
        csv_path = train_dir / f"{name}_loss.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "avg_loss"])
            w.writeheader()
            w.writerows(rows)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [r["epoch"] for r in rows]
            losses = [r["avg_loss"] for r in rows]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, losses, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Avg loss")
            plt.title(f"Training loss — {name}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(train_dir / f"{name}_loss.png", dpi=120)
            plt.close()
        except ImportError:
            pass

    return all_losses


def teacher_translate(text: str, model, tok_src, tok_tgt, seq_len, device) -> tuple[str, float]:
    sos = tok_src.token_to_id(SOS_TOKEN)
    eos = tok_src.token_to_id(EOS_TOKEN)
    pad = tok_src.token_to_id(PAD_TOKEN)

    t0 = time.perf_counter()
    ids = tok_src.encode(text).ids
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
    out = greedy_decode(model, enc, mask, tok_tgt, seq_len, device)
    ms = (time.perf_counter() - t0) * 1000

    out_ids = out.cpu().tolist()
    sos_t = tok_tgt.token_to_id(SOS_TOKEN)
    eos_t = tok_tgt.token_to_id(EOS_TOKEN)
    pad_t = tok_tgt.token_to_id(PAD_TOKEN)
    out_ids = [t for t in out_ids if t not in (sos_t, eos_t, pad_t)]
    return tok_tgt.decode(out_ids), ms


def student_translate_timed(text, model, tok_src, tok_tgt, seq_len, device) -> tuple[str, float]:
    t0 = time.perf_counter()
    hi = translate(text, model, tok_src, tok_tgt, seq_len, device)
    ms = (time.perf_counter() - t0) * 1000
    return hi, ms


def load_teacher(teacher_cfg_path: str, teacher_artifacts: str, checkpoint: str, device):
    cfg = load_config(teacher_cfg_path)
    cfg.paths.output_dir = teacher_artifacts
    tok_src = Tokenizer.from_file(str(cfg.tokenizer_path(cfg.data.lang_src)))
    tok_tgt = Tokenizer.from_file(str(cfg.tokenizer_path(cfg.data.lang_tgt)))
    m = cfg.model
    model = build_transformer(
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
    ckpt = torch.load(cfg.weights_path(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return cfg, model, tok_src, tok_tgt, n_params


def eval_student_bleu_all(
    config_path: str,
    teacher_artifacts: str,
    device,
    num_samples: int,
    max_src_words: int | None = None,
) -> list[dict]:
    cfg = load_distil_config(config_path)
    cfg.teacher.artifacts_dir = teacher_artifacts
    if max_src_words is not None:
        cfg.data.max_src_words = max_src_words

    teacher_cfg = cfg.load_teacher_run_config()
    _, val_loader, tok_src, tok_tgt = get_distil_dataloaders(cfg, teacher_cfg)
    model = build_student_model(cfg, tok_src, tok_tgt, device)
    seq_len = cfg.student.seq_len

    rows = []
    for epoch, path in list_student_checkpoints(cfg):
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        t0 = time.perf_counter()
        bleu, chrf = evaluate_checkpoint(
            model, val_loader, tok_src, tok_tgt, seq_len, device, num_samples
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "epoch": epoch,
                "bleu": round(bleu, 2),
                "chrf": round(chrf, 2),
                "checkpoint": path.name,
                "eval_seconds": round(elapsed, 1),
            }
        )
    return rows


def eval_teacher_bleu(teacher_cfg_path, teacher_artifacts, checkpoint, device, num_samples) -> dict:
    import sacrebleu

    cfg = load_config(teacher_cfg_path)
    cfg.paths.output_dir = teacher_artifacts
    from transformer_lib.data.tokenization import get_translation_dataloaders

    _, val_loader, tok_src, tok_tgt = get_translation_dataloaders(cfg)
    m = cfg.model
    model = build_transformer(
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
    state = torch.load(cfg.weights_path(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    hyps, refs = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            out_ids = greedy_decode(model, enc_in, enc_mask, tok_tgt, m.seq_len, device)
            hyps.append(tok_tgt.decode(out_ids.cpu().tolist()))
            refs.append(batch["tgt_text"][0])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {
        "checkpoint": f"tmodel_{checkpoint}.pt",
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "num_samples": len(hyps),
        "eval_seconds": round(time.perf_counter() - t0, 1),
    }


def run_phrase_benchmark(
    phrases: list[str],
    teacher_model,
    teacher_cfg,
    tok_src,
    tok_tgt,
    student_runs: list[tuple[str, str, int | None]],
    device,
    epochs_only: dict[str, set[int]] | None = None,
) -> dict:
    """student_runs: (label, config_path, max_src_words override or None)"""
    results = {
        "teacher": {
            "checkpoint": teacher_cfg.paths.model_basename,
            "phrases": [],
        },
        "students": {},
    }

    for phrase in phrases:
        hi, ms = teacher_translate(
            phrase, teacher_model, tok_src, tok_tgt, teacher_cfg.model.seq_len, device
        )
        results["teacher"]["phrases"].append({"en": phrase, "hi": hi, "latency_ms": round(ms, 2)})

    if epochs_only is not None and not epochs_only:
        return results

    for label, config_path, max_words in student_runs:
        cfg = load_distil_config(config_path)
        cfg.teacher.artifacts_dir = str(teacher_cfg.paths.output_dir)
        if max_words is not None:
            cfg.data.max_src_words = max_words

        teacher_run_cfg = cfg.load_teacher_run_config()
        _, _, ts, tt = get_distil_dataloaders(cfg, teacher_run_cfg)
        model = build_student_model(cfg, ts, tt, device)
        seq_len = cfg.student.seq_len

        ckpts = list_student_checkpoints(cfg)
        results["students"][label] = {"checkpoints": {}}

        for epoch, path in ckpts:
            if epochs_only is not None:
                allowed = epochs_only.get(label)
                if allowed is not None and epoch not in allowed:
                    continue
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            phrase_rows = []
            for phrase in phrases:
                hi, ms = student_translate_timed(phrase, model, ts, tt, seq_len, device)
                phrase_rows.append({"en": phrase, "hi": hi, "latency_ms": round(ms, 2)})
            results["students"][label]["checkpoints"][str(epoch)] = {
                "file": path.name,
                "phrases": phrase_rows,
            }

    return results


def export_val_examples(
    teacher_model,
    teacher_cfg,
    tok_tgt,
    student_cfg_path,
    student_epoch,
    teacher_artifacts,
    device,
    num_examples,
) -> list[dict]:
    cfg = load_distil_config(student_cfg_path)
    cfg.teacher.artifacts_dir = teacher_artifacts
    teacher_run = cfg.load_teacher_run_config()
    _, val_loader, tok_src, tok_tgt_s = get_distil_dataloaders(cfg, teacher_run)

    student = build_student_model(cfg, tok_src, tok_tgt_s, device)
    ckpt = cfg.student_weights_path(student_epoch)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    student.load_state_dict(state["model_state_dict"])
    student.eval()

    seq_len = cfg.student.seq_len
    sos_id = tok_src.token_to_id(SOS_TOKEN)
    eos_id = tok_src.token_to_id(EOS_TOKEN)
    pad_id = tok_tgt_s.token_to_id(PAD_TOKEN)
    m = teacher_cfg.model

    examples = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_examples:
                break
            src = batch["src_text"][0]
            ref = batch["tgt_text"][0]

            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            t_out = greedy_decode(teacher_model, enc_in, enc_mask, tok_tgt, m.seq_len, device)
            teacher_hi = tok_tgt.decode(t_out.cpu().tolist())

            s_out = student.greedy_decode(enc_in, sos_id, eos_id, seq_len)
            ids = [t for t in s_out.cpu().tolist() if t not in (sos_id, eos_id, pad_id)]
            student_hi = tok_tgt_s.decode(ids)

            examples.append(
                {
                    "src": src,
                    "reference": ref,
                    "teacher": teacher_hi,
                    "student": student_hi,
                }
            )
    return examples


def copy_artifacts(
    out: Path,
    teacher_artifacts: Path,
    teacher_checkpoint: str,
    include_all_teacher_weights: bool,
    include_all_student_weights: bool,
) -> list[dict]:
    manifest = []
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst: Path):
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append(
            {
                "path": str(dst.relative_to(out)),
                "size_mb": round(dst.stat().st_size / 1e6, 2),
            }
        )

    # Tokenizers (small, required for local inference)
    for lang in ("en", "hi"):
        src = teacher_artifacts / "tokenizers" / f"tokenizer_{lang}.json"
        _copy(src, art / "tokenizers" / f"tokenizer_{lang}.json")

    # Teacher weights
    if include_all_teacher_weights:
        wdir = teacher_artifacts / "weights"
        if wdir.exists():
            for pt in sorted(wdir.glob("tmodel_*.pt")):
                _copy(pt, art / "teacher_weights" / pt.name)
    else:
        src = teacher_artifacts / "weights" / f"tmodel_{int(teacher_checkpoint):02d}.pt"
        if not src.exists():
            src = teacher_artifacts / "weights" / f"tmodel_{teacher_checkpoint}.pt"
        _copy(src, art / "teacher_weights" / src.name)

    # Student weights
    for run in ("lstm_kd", "lstm_dict"):
        src_dir = ROOT / "distil" / "outputs" / run / "weights"
        if not src_dir.exists():
            continue
        pts = sorted(src_dir.glob("*.pt"))
        if not include_all_student_weights and pts:
            # keep all by default for experiment continuity
            pass
        for pt in pts:
            _copy(pt, art / "student_weights" / run / pt.name)

    # Logs
    for name in ("lstm_kd", "lstm_dict"):
        log = ROOT / "distil" / "outputs" / name / "distill.log"
        _copy(log, art / "logs" / f"{name}_distill.log")
    train_log = teacher_artifacts / "train.log"
    _copy(train_log, art / "logs" / "teacher_train.log")

    # Configs snapshot
    for rel in (
        "distil/configs/lstm_kd.yaml",
        "distil/configs/lstm_kd_dict.yaml",
        "configs/runpod_en_hi.yaml",
    ):
        src = ROOT / rel
        _copy(src, art / "configs" / Path(rel).name)

    return manifest


def write_report_md(
    out: Path,
    meta: dict,
    losses: dict,
    bleu_general: list[dict],
    bleu_dict: list[dict],
    bleu_teacher: dict,
    phrase_results: dict,
    manifest: list[dict],
    timings: dict,
    args,
) -> None:
    def best_bleu(rows):
        if not rows:
            return None
        return max(rows, key=lambda r: r["bleu"])

    bg = best_bleu(bleu_general)
    bd = best_bleu(bleu_dict)
    skip_bleu = getattr(args, "skip_bleu", False)

    lines = [
        "# Distillation Experiment Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Summary",
        "",
        "This export documents the English→Hindi transformer teacher and LSTM student",
        "knowledge-distillation experiments (general KD + short-phrase dictionary fine-tune).",
        "",
        "### What we tried",
        "",
        "1. **Teacher:** ~50M param transformer (6 layers, d=512), trained on opus-100 en-hi.",
        "2. **Student (general):** ~8.8M LSTM + Bahdanau attention, KD (α=0.7, T=3) + CE.",
        "   Weight tying + 128-dim embeddings.",
        "3. **Student (dict):** Fine-tuned from general checkpoint on opus pairs with",
        "   1–3 English words (179k pairs), α=0.4, lower LR.",
        "",
        "### Key finding",
        "",
        "High corpus BLEU on filtered short val does not guarantee good idiomatic",
        "phrase translations. Manual phrase eval is essential.",
        "",
        "## Hardware",
        "",
        f"- GPU: {meta.get('gpu_name', 'n/a')}",
        f"- GPU memory: {meta.get('gpu_total_memory_gb', 'n/a')} GB",
        f"- PyTorch: {meta.get('torch')} / CUDA {meta.get('cuda_version')}",
        f"- Platform: {meta.get('platform')}",
        "",
        "## Compile timings (seconds)",
        "",
        "```json",
        json.dumps(timings, indent=2),
        "```",
        "",
    ]

    if skip_bleu or not bleu_teacher:
        lines.extend(
            [
                "## Metrics",
                "",
                "Corpus BLEU was skipped. Primary eval: **phrase benchmark** (~40 fixed phrases).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"## BLEU metrics ({args.bleu_samples} val samples)",
                "",
                "### Teacher (general val)",
                "",
                f"- Checkpoint: `{bleu_teacher.get('checkpoint', 'n/a')}`",
                f"- BLEU: **{bleu_teacher.get('bleu', 'n/a')}** | chrF: {bleu_teacher.get('chrf', 'n/a')}",
                "",
                "### Student — general KD (`lstm_kd`)",
                "",
                "| epoch | BLEU | chrF | checkpoint |",
                "|------:|-----:|-----:|------------|",
            ]
        )
        for r in bleu_general:
            lines.append(f"| {r['epoch']} | {r['bleu']} | {r['chrf']} | {r['checkpoint']} |")
        if bg:
            lines.extend(
                [
                    "",
                    f"**Best general:** epoch {bg['epoch']} — BLEU {bg['bleu']}",
                    "",
                    "### Student — dictionary fine-tune (`lstm_dict`)",
                    "",
                    "| epoch | BLEU | chrF | checkpoint |",
                    "|------:|-----:|-----:|------------|",
                ]
            )
        for r in bleu_dict:
            lines.append(f"| {r['epoch']} | {r['bleu']} | {r['chrf']} | {r['checkpoint']} |")
        if bd:
            lines.append(f"\n**Best dict:** epoch {bd['epoch']} — BLEU {bd['bleu']}")
        lines.append("")

    lines.extend(
        [
            "",
            "## Phrase benchmark",
            "",
            "See `inferences/phrases_benchmark.json` for phrase-level outputs.",
            "See `inferences/phrases_comparison.md` for a readable side-by-side table.",
            f"Phrase mode: `{getattr(args, 'phrase_mode', 'all')}`",
            "",
            "## Training loss",
            "",
            "CSV and PNG (if matplotlib available) in `training/`.",
            "",
            "## Artifacts in this bundle",
            "",
            "| file | size (MB) |",
            "|------|----------:|",
        ]
    )
    for m in sorted(manifest, key=lambda x: x["path"]):
        lines.append(f"| `{m['path']}` | {m['size_mb']} |")

    lines.extend(
        [
            "",
            "## Suggested next steps",
            "",
            "- Increase `seq_len` (128 → 256) for general student",
            "- KD into 2-layer transformer student instead of LSTM",
            "- Mixed curated + opus short-phrase training",
            "- Beam search at inference (+2–5 BLEU)",
            "- BPE tokenization to shrink embeddings",
            "",
            "## Quantization",
            "",
            "Run separately or with `--quantize` on this script:",
            "",
            "```bash",
            "python distil/quantize_teacher.py --checkpoint 10 --eval-mode phrases --save-models",
            "python distil/quantize_student.py --checkpoint latest --save-models",
            "```",
            "",
            "Results land in `quantized/quant_compare_teacher_*.csv` and `quant_compare_dict_*.csv`.",
            "",
            "## Local setup after download",
            "",
            "```bash",
            "tar -xzf experiment_bundle.tar.gz",
            "cd transformer_from_scratch_translation",
            "# tokenizers in experiment_export/.../artifacts/tokenizers/",
            "python distil/infer.py --config distil/configs/lstm_kd_dict.yaml \\",
            "  --checkpoint 09 --teacher-artifacts ./experiment_export/.../artifacts \\",
            "  --text \"good morning\"",
            "```",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_phrases_comparison_md(out: Path, phrase_results: dict) -> None:
    phrases = [p["en"] for p in phrase_results["teacher"]["phrases"]]
    teacher_map = {p["en"]: p["hi"] for p in phrase_results["teacher"]["phrases"]}

    student_cols: list[tuple[str, str, list]] = []
    for label, data in phrase_results.get("students", {}).items():
        ckpts = data.get("checkpoints", {})
        for ep_str in sorted(ckpts.keys(), key=int):
            student_cols.append((label, ep_str, ckpts[ep_str]["phrases"]))

    header = "| English | Teacher |" + "".join(
        f" {label} ep{ep} |" for label, ep, _ in student_cols
    )
    lines = ["# Phrase comparison", "", header, "|" + "---|" * (2 + len(student_cols))]

    for phrase in phrases:
        row = f"| {phrase} | {teacher_map.get(phrase, '')} |"
        for _, _, prows in student_cols:
            hm = {p["en"]: p["hi"] for p in prows}
            row += f" {hm.get(phrase, '')} |"
        lines.append(row)

    (out / "inferences" / "phrases_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def create_tarball(export_dir: Path, tarball_path: Path) -> None:
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)


def run_quantization_exports(export_dir: Path, args, teacher_artifacts: str) -> float:
    """Run teacher + student quant comparison on CPU."""
    import subprocess

    quant_dir = export_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    qmethods = ["fp32_baseline", "fp16", "dynamic_int8_linear"]

    print("\n=== Quantization comparison (CPU) ===")
    t0 = time.perf_counter()

    subprocess.run(
        [
            py,
            str(ROOT / "distil/quantize_teacher.py"),
            "--teacher-artifacts",
            teacher_artifacts,
            "--checkpoint",
            args.teacher_checkpoint,
            "--num-samples",
            "50",
            "--eval-mode",
            "phrases",
            "--methods",
            *qmethods,
            "--output-dir",
            str(quant_dir),
            "--save-models",
        ],
        check=False,
    )
    subprocess.run(
        [
            py,
            str(ROOT / "distil/quantize_student.py"),
            "--config",
            str(ROOT / "distil/configs/lstm_kd_dict.yaml"),
            "--teacher-artifacts",
            teacher_artifacts,
            "--checkpoint",
            "latest",
            "--dict-max-src-words",
            str(args.dict_max_src_words),
            "--num-samples",
            "50",
            "--methods",
            *qmethods,
            "--output-dir",
            str(quant_dir),
            "--save-models",
        ],
        check=False,
    )

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"Quantization exports finished in {elapsed}s → {quant_dir}")
    return elapsed


def latest_checkpoint_epoch(config_path: str, teacher_artifacts: str) -> int | None:
    cfg = load_distil_config(config_path)
    cfg.teacher.artifacts_dir = teacher_artifacts
    ckpts = list_student_checkpoints(cfg)
    return ckpts[-1][0] if ckpts else None


def main() -> None:
    p = argparse.ArgumentParser(description="Compile distillation experiment export")
    p.add_argument("--export-dir", default=None, help="Output dir (default: distil/exports/experiment_TIMESTAMP)")
    p.add_argument("--teacher-artifacts", default="/workspace/outputs/en_hi")
    p.add_argument("--teacher-config", default=str(ROOT / "configs" / "runpod_en_hi.yaml"))
    p.add_argument("--teacher-checkpoint", default="10")
    p.add_argument("--bleu-samples", type=int, default=200)
    p.add_argument("--val-examples", type=int, default=50)
    p.add_argument("--dict-max-src-words", type=int, default=3, help="Match dict training filter")
    p.add_argument("--include-all-teacher-weights", action="store_true")
    p.add_argument("--tarball", action="store_true")
    p.add_argument(
        "--phrase-mode",
        choices=("all", "best", "teacher"),
        default="all",
        help="all=every student ckpt; best=teacher+best BLEU epoch each run; teacher=teacher only",
    )
    p.add_argument(
        "--lightweight",
        action="store_true",
        help="Fast export (~5-15 min): phrase translations only, all student epochs",
    )
    p.add_argument(
        "--skip-bleu",
        action="store_true",
        help="Skip corpus BLEU; use phrase benchmark as primary metric",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Run teacher+student quant comparison (CPU, phrase eval, ~15-40 min)",
    )
    args = p.parse_args()

    if args.lightweight:
        args.skip_bleu = True
        args.phrase_mode = "all"
        args.val_examples = 0
        print("Lightweight mode: phrase translations on all checkpoints (no corpus BLEU)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(args.export_dir or ROOT / "distil" / "exports" / f"experiment_{ts}")
    export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings: dict[str, float] = {}
    teacher_artifacts = Path(args.teacher_artifacts)

    print(f"Export directory: {export_dir}")
    print(f"Device: {device}")

    t0 = time.perf_counter()
    meta = collect_metadata(export_dir)
    timings["metadata_seconds"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    losses = export_loss_curves(
        export_dir,
        {
            "lstm_kd": ROOT / "distil" / "outputs" / "lstm_kd" / "distill.log",
            "lstm_dict": ROOT / "distil" / "outputs" / "lstm_dict" / "distill.log",
        },
    )
    timings["loss_curves_seconds"] = round(time.perf_counter() - t0, 1)

    bleu_teacher: dict = {}
    bleu_general: list[dict] = []
    bleu_dict: list[dict] = []

    if not args.skip_bleu:
        t0 = time.perf_counter()
        bleu_teacher = eval_teacher_bleu(
            args.teacher_config,
            str(teacher_artifacts),
            args.teacher_checkpoint,
            device,
            args.bleu_samples,
        )
        timings["teacher_bleu_seconds"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        bleu_general = eval_student_bleu_all(
            str(ROOT / "distil/configs/lstm_kd.yaml"),
            str(teacher_artifacts),
            device,
            args.bleu_samples,
        )
        timings["student_general_bleu_seconds"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        bleu_dict = eval_student_bleu_all(
            str(ROOT / "distil/configs/lstm_kd_dict.yaml"),
            str(teacher_artifacts),
            device,
            args.bleu_samples,
            max_src_words=args.dict_max_src_words,
        )
        timings["student_dict_bleu_seconds"] = round(time.perf_counter() - t0, 1)

        metrics_dir = export_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "bleu_teacher.json").write_text(json.dumps(bleu_teacher, indent=2), encoding="utf-8")
        for name, rows in [("bleu_general", bleu_general), ("bleu_dict", bleu_dict)]:
            with open(metrics_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
    else:
        timings["teacher_bleu_seconds"] = 0
        timings["student_general_bleu_seconds"] = 0
        timings["student_dict_bleu_seconds"] = 0
        print("Skipping corpus BLEU (--skip-bleu / --lightweight)")

    phrase_file = ROOT / "distil" / "report" / "phrases_benchmark.txt"
    phrases = [ln.strip() for ln in phrase_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    t0 = time.perf_counter()
    teacher_cfg, teacher_model, tok_src, tok_tgt, teacher_params = load_teacher(
        args.teacher_config, str(teacher_artifacts), args.teacher_checkpoint, device
    )
    teacher_cfg.paths.output_dir = str(teacher_artifacts)
    timings["load_teacher_seconds"] = round(time.perf_counter() - t0, 1)

    student_runs = [
        ("general_kd", str(ROOT / "distil/configs/lstm_kd.yaml"), None),
        ("dict_finetune", str(ROOT / "distil/configs/lstm_kd_dict.yaml"), args.dict_max_src_words),
    ]

    epochs_only: dict[str, set[int]] | None = None
    if args.phrase_mode == "teacher":
        epochs_only = {}
    elif args.phrase_mode == "best":
        epochs_only = {}
        if bleu_general:
            epochs_only["general_kd"] = {max(bleu_general, key=lambda r: r["bleu"])["epoch"]}
        elif (ep := latest_checkpoint_epoch(str(ROOT / "distil/configs/lstm_kd.yaml"), str(teacher_artifacts))) is not None:
            epochs_only["general_kd"] = {ep}
        if bleu_dict:
            epochs_only["dict_finetune"] = {max(bleu_dict, key=lambda r: r["bleu"])["epoch"]}
        elif (ep := latest_checkpoint_epoch(str(ROOT / "distil/configs/lstm_kd_dict.yaml"), str(teacher_artifacts))) is not None:
            epochs_only["dict_finetune"] = {ep}

    t0 = time.perf_counter()
    phrase_results = run_phrase_benchmark(
        phrases,
        teacher_model,
        teacher_cfg,
        tok_src,
        tok_tgt,
        student_runs,
        device,
        epochs_only=epochs_only,
    )
    timings["phrase_benchmark_seconds"] = round(time.perf_counter() - t0, 1)

    inf_dir = export_dir / "inferences"
    inf_dir.mkdir(parents=True, exist_ok=True)
    (inf_dir / "phrases_benchmark.json").write_text(
        json.dumps(phrase_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_phrases_comparison_md(export_dir, phrase_results)

    if bleu_general:
        best_general = max(bleu_general, key=lambda r: r["bleu"])["epoch"]
    else:
        best_general = latest_checkpoint_epoch(str(ROOT / "distil/configs/lstm_kd.yaml"), str(teacher_artifacts)) or 0

    if args.val_examples > 0:
        t0 = time.perf_counter()
        val_examples = export_val_examples(
            teacher_model,
            teacher_cfg,
            tok_tgt,
            str(ROOT / "distil/configs/lstm_kd.yaml"),
            best_general,
            str(teacher_artifacts),
            device,
            args.val_examples,
        )
        timings["val_examples_seconds"] = round(time.perf_counter() - t0, 1)
        (inf_dir / "val_examples_general.json").write_text(
            json.dumps(val_examples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        timings["val_examples_seconds"] = 0

    t0 = time.perf_counter()
    manifest = copy_artifacts(
        export_dir,
        teacher_artifacts,
        args.teacher_checkpoint,
        args.include_all_teacher_weights,
        include_all_student_weights=True,
    )
    timings["copy_artifacts_seconds"] = round(time.perf_counter() - t0, 1)

    summary = {
        "teacher_params": teacher_params,
        "teacher_checkpoint": args.teacher_checkpoint,
        "loss_epochs": {k: len(v) for k, v in losses.items()},
        "bleu_teacher": bleu_teacher or None,
        "best_general_epoch": best_general,
        "best_dict_epoch": (
            max(bleu_dict, key=lambda r: r["bleu"])["epoch"]
            if bleu_dict
            else latest_checkpoint_epoch(str(ROOT / "distil/configs/lstm_kd_dict.yaml"), str(teacher_artifacts))
        ),
        "phrase_count": len(phrases),
        "skip_bleu": args.skip_bleu,
        "phrase_mode": args.phrase_mode,
        "timings": timings,
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.quantize:
        timings["quantization_seconds"] = run_quantization_exports(
            export_dir, args, str(teacher_artifacts)
        )

    write_report_md(
        export_dir,
        meta,
        losses,
        bleu_general,
        bleu_dict,
        bleu_teacher,
        phrase_results,
        manifest,
        timings,
        args,
    )

    tarball_path = ROOT / "distil" / "exports" / "experiment_bundle.tar.gz"
    if args.tarball:
        t0 = time.perf_counter()
        create_tarball(export_dir, tarball_path)
        timings["tarball_seconds"] = round(time.perf_counter() - t0, 1)
        size_gb = tarball_path.stat().st_size / 1e9
        print(f"\nTarball: {tarball_path} ({size_gb:.2f} GB)")

    print(f"\nDone. Report: {export_dir / 'REPORT.md'}")
    print("\n--- Download to your Mac (replace POD_IP and PORT) ---")
    print(f"scp -P PORT root@POD_IP:{tarball_path if args.tarball else export_dir} ~/Downloads/")


if __name__ == "__main__":
    main()
