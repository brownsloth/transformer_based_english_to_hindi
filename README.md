# Transformer Translation (EN → HI)

Production-style codebase converted from the Colab notebook. Trains an encoder–decoder Transformer on [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (`en-hi`) with reusable modules for future tasks (e.g. sequence classification).

## Project layout

```
configs/                 YAML experiment configs
scripts/
  train.py               Main training entrypoint
  analyze_seq_len.py     Recommend seq_len from data
  infer.py               Greedy decode one sentence
deploy/
  setup_remote.sh        VM bootstrap
  run_train.sh           Background training (nohup)
src/transformer_lib/
  models/
    layers.py            Shared blocks (attention, FFN, etc.)
    transformer.py       Seq2seq model + build_transformer()
    sequence_classifier.py  Encoder-only classifier (reuse layers)
  data/                  Dataset + tokenizers
  training/              Trainer, decode, validation
  monitoring/            TensorBoard, status.json, webhooks
```

## Quick start (local or GPU VM)

**First time on your machine?** Follow **[docs/LOCAL_RUN.md](docs/LOCAL_RUN.md)** for a step-by-step smoke test (training + monitoring + TensorBoard).

```bash
cd transformer_from_scratch_translation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Quick local verification (~minutes)
python scripts/train.py --config configs/local_smoke_test.yaml

# Full training (en → hi)
python scripts/train.py --config configs/en_hi_translation.yaml
```

## RunPod / cloud GPU

See **[docs/RUNPOD_DETAILED.md](docs/RUNPOD_DETAILED.md)** (step-by-step) or [docs/RUNPOD.md](docs/RUNPOD.md) — W&B, Slack, TensorBoard, tmux.

```bash
bash deploy/runpod_start.sh   # uses configs/runpod_en_hi.yaml + .env secrets
```

## Remote GPU VM deployment

1. Copy the repo to the VM (`git clone`, `rsync`, etc.).
2. Run `bash deploy/setup_remote.sh`.
3. Start training in the background:

```bash
export ALERT_WEBHOOK_URL="https://hooks.slack.com/services/..."  # optional
bash deploy/run_train.sh
```

4. Monitor from your laptop:

```bash
# Progress heartbeat (epoch, step, loss, ETA)
ssh user@gpu-vm 'cat ~/transformer_from_scratch_translation/outputs/en_hi/status.json'

# Live log
ssh user@gpu-vm 'tail -f ~/transformer_from_scratch_translation/outputs/en_hi/train.log'

# TensorBoard (port-forward)
ssh -L 6006:localhost:6006 user@gpu-vm
# on VM: tensorboard --logdir outputs/en_hi/runs --bind_all --port 6006
# open http://localhost:6006
```

### Alerts

Set `ALERT_WEBHOOK_URL` to a Slack/Discord/generic webhook. Alerts fire on start, each epoch end, finish, and failure.

Or set `monitoring.webhook_url` in `configs/en_hi_translation.yaml`.

## Configuration

Edit `configs/en_hi_translation.yaml` or override via CLI:

| Flag | Effect |
|------|--------|
| `--output-dir` | Artifact root |
| `--batch-size` | Training batch size |
| `--epochs` | Number of epochs |
| `--preload 05` | Resume from `tmodel_05.pt` |
| `--seq-len` | Max sequence length |
| `--no-amp` | Disable mixed precision |

Artifacts under `outputs/en_hi/`:

- `weights/tmodel_XX.pt` — checkpoints
- `tokenizers/` — source/target tokenizers
- `runs/` — TensorBoard logs
- `status.json` — remote monitoring heartbeat
- `train.log` — structured training log
- `config.json` — resolved config snapshot

## Reusing models for other tasks

### Translation (seq2seq)

```python
from transformer_lib import build_transformer
model = build_transformer(src_vocab, tgt_vocab, seq_len, seq_len)
```

### Sequence classification

```python
from transformer_lib import build_encoder_classifier
model = build_encoder_classifier(vocab_size, seq_len, num_classes=3, pool="mean")
```

Shares `layers.py` (embeddings, encoder blocks, attention) with the translation model.

## Inference

```bash
python scripts/infer.py \
  --checkpoint 05 \
  --text "Hello, how are you?"
```

## Notebook

The original notebook is kept as `[Umar_Jamil]_Training_a_transformer_Translation_task.ipynb` for reference. Use this repo for training and deployment.

## Memory notes

The notebook OOM'd with `batch_size=32` and `seq_len=450` on a T4 (16GB). Default config uses `batch_size=8` and AMP. Reduce further with `--batch-size 4` if needed.
