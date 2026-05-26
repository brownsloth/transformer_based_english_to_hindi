# Knowledge distillation — Transformer teacher → LSTM student

Distill your trained transformer (**teacher**) into a **small LSTM seq2seq** (**student**) using **teacher logits (KD) + ground-truth CE**.

Optimized for **low-latency CPU inference** (Railway) while keeping as much accuracy as possible.

---

## Architecture

```
Teacher (frozen)     Student (trained)
Transformer ~50M  →  LSTM + attention ~5–15M
log_probs per step   KL + CE loss
```

**Loss:** `α · KL(student || teacher) · T² + (1-α) · CE(labels)`

Default: `α=0.6`, `T=3.0`

---

## Quick start (RunPod)

```bash
cd /workspace/transformer_from_scratch_translation
source .venv/bin/activate

# Teacher must exist at /workspace/outputs/en_hi/weights/tmodel_08.pt
bash distil/run_runpod.sh
```

In tmux:

```bash
tmux new -s distill
bash distil/run_runpod.sh
# Ctrl+B, D
```

---

## Evaluate student BLEU

```bash
python distil/eval_bleu.py --checkpoint 24 --num-samples 200 \
  --teacher-artifacts /workspace/outputs/en_hi
```

Compare to teacher:

```bash
python scripts/eval_bleu.py --config configs/runpod_en_hi.yaml \
  --checkpoint 8 --num-samples 200
```

---

## Config

Edit `distil/configs/lstm_kd.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `student.seq_len` | 128 | Shorter = faster CPU infer |
| `student.hidden_dim` | 256 | ↑ accuracy, ↑ latency |
| `distillation.alpha` | 0.6 | ↑ = trust teacher more |
| `distillation.temperature` | 3.0 | Softer teacher distribution |
| `data.batch_size` | 128 | LSTM is lighter than transformer |

CLI overrides:

```bash
python distil/train.py \
  --teacher-artifacts /workspace/outputs/en_hi \
  --teacher-checkpoint 8 \
  --batch-size 256 \
  --epochs 30 \
  --alpha 0.7
```

---

## Outputs

```
distil/outputs/lstm_kd/
  weights/lstm_00.pt … lstm_24.pt
  distill.log
```

---

## After distillation — serve on Railway (CPU)

Point `serving/` at the LSTM student (separate serving update) or use:

```python
from distil.student.lstm_seq2seq import build_lstm_student
# load lstm_XX.pt, greedy_decode on CPU — ~100–500ms typical
```

---

## Tuning for max accuracy (still Path D)

1. **`alpha=0.7–0.8`** — more teacher imitation  
2. **`seq_len=128`** — good speed; try **160** if BLEU drops on long sentences  
3. **`hidden_dim=384`** — +accuracy, still CPU-ok  
4. **Train 25–30 epochs** — stop when `eval_bleu.py` plateaus  
5. Optional Phase B: pseudo-label extra English (not included yet)

**Skip RL** — KD + CE is the right tool here.

---

## Layout

```
distil/
  README.md
  train.py              Entry point
  eval_bleu.py          Student BLEU
  run_runpod.sh         One-command RunPod
  config.py
  configs/lstm_kd.yaml
  data.py
  teacher/wrapper.py    Frozen transformer → log_probs
  student/lstm_seq2seq.py
  training/trainer.py
```

---

## Smoke test (local, small)

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
python distil/train.py \
  --teacher-artifacts serving/artifacts \
  --teacher-checkpoint 8 \
  --batch-size 32 \
  --epochs 1
```

(Requires `bash serving/package_artifacts.sh` first.)
