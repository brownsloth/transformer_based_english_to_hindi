# RunPod GPU training (W&B + Slack + TensorBoard)

Use this after your local smoke test passes.

## What you set up (your checklist)

### 1. Weights & Biases

1. Create account at [wandb.ai](https://wandb.ai).
2. Settings → **API keys** → copy key.
3. Optional: create project `transformer-en-hi` (or it is created on first run).

### 2. Slack alerts

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → From scratch.
2. Name it e.g. `Transformer Training`, pick your workspace.
3. **Incoming Webhooks** → turn **On** → **Add New Webhook to Workspace** → pick channel (e.g. `#ml-training`).
4. Copy webhook URL (`https://hooks.slack.com/services/...`).
5. In `configs/runpod_en_hi.yaml`, set `monitoring.webhook_type: slack` (already default).

You do **not** need a full Slack bot for simple pings — incoming webhooks are enough.

### 3. RunPod

1. Add billing (~$10+ to start).
2. **Deploy** a GPU pod:
   - GPU: **RTX 4090** (24GB) or A5000 if 4090 unavailable
   - Template: **PyTorch 2.x** (Ubuntu + CUDA)
   - Disk: **≥ 50 GB** container + optional **Volume** mounted at `/workspace` for persistent outputs
3. Note **SSH** command from the pod page.
4. **Stop the pod** when done — billing runs until you terminate.

### 4. Secrets on the pod

Create `.env` on the server (never commit it):

```bash
WANDB_API_KEY=...
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
```

---

## Deploy code to RunPod

**Option A — Git (recommended)**

```bash
ssh root@<pod-ip> -p <port>
cd /workspace
git clone <your-repo-url> transformer_translation
cd transformer_translation
```

**Option B — rsync from laptop**

```bash
rsync -avz -e "ssh -p <port>" \
  --exclude .venv --exclude outputs \
  ./ root@<pod-ip>:/workspace/transformer_translation/
```

---

## Install and run (tmux)

```bash
cd /workspace/transformer_translation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt wandb

cp .env.example .env   # edit with real keys
nano .env

tmux new -s train
source .venv/bin/activate
export PYTHONPATH="/workspace/transformer_translation/src:$PYTHONPATH"
bash deploy/runpod_start.sh
```

Detach: **Ctrl+B**, then **D**. Training keeps running.

Reattach:

```bash
tmux attach -t train
```

---

## Monitor while away

| Tool | Where |
|------|--------|
| **W&B** | [wandb.ai](https://wandb.ai) → project `transformer-en-hi` — live loss charts, sample translations table |
| **Slack** | Channel ping on start / each epoch / finish / failure |
| **TensorBoard** | SSH port-forward (below) or download `runs/` |
| **status.json** | `cat /workspace/outputs/en_hi/status.json` over SSH |

### TensorBoard on RunPod (optional)

On the pod:

```bash
tensorboard --logdir /workspace/outputs/en_hi/runs --host 0.0.0.0 --port 6006
```

On your laptop:

```bash
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>
```

Open http://localhost:6006

### OOM on 4090

```bash
python scripts/train.py --config configs/runpod_en_hi.yaml --batch-size 8
```

---

## When training finishes

1. Slack message: **Training finished**.
2. Save artifacts to your machine:

```bash
rsync -avz -e "ssh -p <port>" \
  root@<pod-ip>:/workspace/outputs/en_hi/ \
  ./outputs/en_hi_from_runpod/
```

Important files:

- `weights/tmodel_*.pt`
- `tokenizers/`
- `config.json`
- `runs/` (TensorBoard)

3. **Terminate** the RunPod pod to stop charges.

---

## Config reference

| File | Purpose |
|------|---------|
| `configs/runpod_en_hi.yaml` | GPU defaults, W&B on, Slack type |
| `configs/en_hi_translation.yaml` | Local / generic full training |
| `.env.example` | Secret template |

Environment variables (override YAML):

- `WANDB_API_KEY` — required for W&B
- `ALERT_WEBHOOK_URL` — Slack webhook
- `WANDB_PROJECT` — optional project name

---

## What the codebase logs

- **TensorBoard**: `train/loss`, `train/epoch_loss`, `train/lr` (unchanged)
- **W&B**: same scalars + `val/translations` table each validation
- **Slack**: start, epoch end (avg loss), finish, errors
- **status.json**: step, ETA, loss for quick SSH checks
