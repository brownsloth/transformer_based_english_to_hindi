# Local run guide

Follow these steps on your Mac/Linux machine to verify install, training, TensorBoard, and monitoring before deploying to a GPU VM.

## Prerequisites

- Python 3.10+ (3.9 may work)
- ~8 GB free disk (dataset cache + outputs)
- Internet for first run (downloads `Helsinki-NLP/opus-100`)
- Optional: NVIDIA GPU or Apple Silicon (MPS) for faster runs

## 1. One-time setup

From the project root:

```bash
cd transformer_from_scratch_translation

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Confirm PyTorch sees your device:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('mps:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

## 2. Smoke test (recommended first)

Uses a small subset (200 sentences), tiny model, 1 epoch — finishes in minutes on CPU.

```bash
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

python scripts/train.py --config configs/local_smoke_test.yaml
```

**What to expect**

- First run downloads the dataset and builds tokenizers (can take a few minutes).
- Progress bar: `Epoch 00` with loss in the postfix.
- Validation samples printed every 10 steps.

## 3. Verify monitoring artifacts

In a **second terminal** while training (or after it finishes):

```bash
cd transformer_from_scratch_translation
OUT=outputs/local_smoke

# Heartbeat (epoch, step, loss, ETA)
cat "$OUT/status.json"

# Training log
tail -20 "$OUT/train.log"

# Saved config snapshot
cat "$OUT/config.json"

# Checkpoints
ls -la "$OUT/weights/"

# Tokenizers
ls -la "$OUT/tokenizers/"
```

`status.json` should update every 5 steps with `"state": "training"` and increasing `"global_step"`.

## 4. TensorBoard (local)

Use the **project venv** (system TensorBoard + setuptools 82 often crashes with `pkg_resources` errors).

```bash
source .venv/bin/activate
pip install "setuptools>=65,<81" "tensorboard>=2.16"
tensorboard --logdir outputs/local_smoke/runs --port 6006
```

Or without activating:

```bash
.venv/bin/tensorboard --logdir outputs/local_smoke/runs --port 6006
```

Open http://localhost:6006 — you should see `train/loss`, `train/epoch_loss`, etc.

**If you still see `pkg_resources.iter_entry_points`:** you are on the wrong Python. Run `which tensorboard` — it must point inside `.venv/`.

## 5. Optional: analyze sequence length (full dataset)

Before a full training run:

```bash
python scripts/analyze_seq_len.py --config configs/en_hi_translation.yaml
```

Use the printed **recommended seq_len** in `configs/en_hi_translation.yaml` if needed.

## 6. Full local training (GPU recommended)

Uses the full dataset and default model. On a 16 GB GPU, keep `batch_size: 8` in the config.

```bash
python scripts/train.py --config configs/en_hi_translation.yaml
```

Reduce memory if needed:

```bash
python scripts/train.py --config configs/en_hi_translation.yaml --batch-size 4
```

Outputs go to `outputs/en_hi/` (same monitoring paths as smoke test).

## 7. Test inference (after smoke or full run)

```bash
# Smoke checkpoint (epoch 0 -> smoke_00.pt)
python scripts/infer.py \
  --config configs/local_smoke_test.yaml \
  --checkpoint 0 \
  --text "Hello, how are you?"

# Full model (epoch 0 -> tmodel_00.pt)
python scripts/infer.py \
  --config configs/en_hi_translation.yaml \
  --checkpoint 0 \
  --text "The weather is nice today."
```

## 8. Test webhook alerts (optional)

Use a test webhook (e.g. Slack incoming webhook or [webhook.site](https://webhook.site)):

```bash
export ALERT_WEBHOOK_URL="https://your-webhook-url"
python scripts/train.py --config configs/local_smoke_test.yaml
```

With `alert_on_*` enabled in config, you will get POSTs on start/finish/epoch.

## Troubleshooting

| Issue | Fix |
|--------|-----|
| `ModuleNotFoundError: transformer_lib` | `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"` |
| CUDA OOM | `--batch-size 2` or use `local_smoke_test.yaml` |
| macOS dataloader hangs | `num_workers: 0` (already set in smoke config) |
| Sentence too long | Increase `model.seq_len` or set `truncate_long: true` |
| Slow on CPU | Use `configs/local_smoke_test.yaml` only for verification |

## Checklist

- [ ] `pip install -r requirements.txt` succeeds
- [ ] Smoke training completes without error
- [ ] `outputs/local_smoke/status.json` updates
- [ ] `outputs/local_smoke/train.log` has epoch lines
- [ ] `outputs/local_smoke/weights/smoke_00.pt` exists
- [ ] TensorBoard shows scalars at localhost:6006
- [ ] `scripts/infer.py` prints a Hindi-ish translation

Once all boxes are checked, use `deploy/run_train.sh` on your remote GPU VM.
