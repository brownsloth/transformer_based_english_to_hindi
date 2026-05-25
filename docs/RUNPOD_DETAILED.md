# RunPod — detailed setup (EN→HI Transformer)

Prerequisites: local smoke test passed, secrets in project `.env` (gitignored).

---

## Part 1 — Accounts (one-time)

### Slack

1. Open https://api.slack.com/apps
2. **Create New App** → **From scratch**
3. App name: `Transformer Training`, workspace: yours
4. Left menu → **Incoming Webhooks** → toggle **On**
5. **Add New Webhook to Workspace** → pick channel (e.g. `#general` or create `#ml-training`)
6. Copy webhook URL → already saved in your `.env` as `ALERT_WEBHOOK_URL`

Test from your laptop:

```bash
cd transformer_from_scratch_translation
source .env 2>/dev/null || set -a && source .env && set +a
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"RunPod test ping from laptop"}' \
  "$ALERT_WEBHOOK_URL"
```

You should see the message in Slack.

### Weights & Biases

1. Log in at https://wandb.ai (user: **tarun-ssharma**)
2. API key is in your `.env` as `WANDB_API_KEY`
3. Project `transformer-en-hi` is created automatically on first run

Test locally:

```bash
source .venv/bin/activate
pip install wandb
export $(grep -v '^#' .env | xargs)
wandb login --relogin
# paste API key when prompted, or it uses WANDB_API_KEY if set
```

---

## Part 2 — RunPod pod

### 2.1 Billing

1. https://www.runpod.io → sign up / log in
2. **Billing** → add payment method + **$10–20** credit

### 2.2 Deploy GPU

1. **Pods** → **Deploy** (+ GPU Pod)
2. **GPU**: search **RTX 4090** (24 GB VRAM). If unavailable, **RTX 3090** or **A5000**.
3. **Pod template**: pick **RunPod PyTorch 2.x** or **Ubuntu 22.04 + CUDA 12** (anything with Python 3.10+ and NVIDIA drivers).
4. **Container disk**: **50 GB** minimum (dataset cache + checkpoints).
5. **Volume disk** (recommended): **50–100 GB**, mount path **`/workspace`**
   - Persists outputs if the container restarts (still copy weights home when done).
6. **Start SSH** / **Start Jupyter** — enable **SSH** (required).
7. Click **Deploy** — wait until status is **Running**.

### 2.3 Connect via SSH

On the pod page, copy the SSH command, e.g.:

```bash
ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

First connection: type `yes` for host key.

Verify GPU:

```bash
nvidia-smi
python3 --version
```

You should see the GPU name and driver version.

---

## Part 3 — Put code on the pod

### Option A — Git (best if repo is on GitHub)

On the pod:

```bash
cd /workspace
git clone https://github.com/YOUR_USER/transformer_from_scratch_translation.git
cd transformer_from_scratch_translation
```

### Option B — rsync from your Mac (no GitHub needed)

On your **Mac** (new terminal), from the project parent folder:

```bash
rsync -avz --progress -e "ssh -p YOUR_PORT" \
  --exclude '.venv' \
  --exclude 'outputs' \
  --exclude '__pycache__' \
  --exclude '.git' \
  transformer_from_scratch_translation/ \
  root@YOUR_POD_IP:/workspace/transformer_from_scratch_translation/
```

Replace `YOUR_PORT` and `YOUR_POD_IP` from the RunPod SSH box.

---

## Part 4 — Secrets on the pod

On the pod:

```bash
cd /workspace/transformer_from_scratch_translation
nano .env
```

Paste the same contents as your local `.env` (W&B key, Slack URL, entity). Save: Ctrl+O, Enter, Ctrl+X.

```bash
chmod 600 .env
```

---

## Part 5 — Install dependencies

```bash
cd /workspace/transformer_from_scratch_translation

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt wandb

export $(grep -v '^#' .env | xargs)
wandb login --relogin
# paste API key if asked
```

Optional: analyze seq length (downloads dataset once):

```bash
export PYTHONPATH="/workspace/transformer_from_scratch_translation/src:$PYTHONPATH"
python scripts/analyze_seq_len.py --config configs/runpod_en_hi.yaml
```

---

## Part 6 — Start training in tmux

```bash
cd /workspace/transformer_from_scratch_translation
source .venv/bin/activate

tmux new -s train
```

Inside tmux:

```bash
cd /workspace/transformer_from_scratch_translation
source .venv/bin/activate
bash deploy/runpod_start.sh
```

You should see:

- `Device: cuda`
- `W&B enabled: True`
- `Slack/webhook alerts: True`
- tqdm progress for epoch 00

**Detach** (training keeps running): press **Ctrl+B**, then **D**.

**Reattach later:**

```bash
tmux attach -t train
```

**Kill session** (only if you want to stop training):

```bash
tmux kill-session -t train
```

---

## Part 7 — Monitor while away

| Where | What |
|--------|------|
| https://wandb.ai/tarun-ssharma | Project **transformer-en-hi** — live loss charts, translation samples |
| Slack channel | Pings: started, each epoch, finished / failed |
| SSH + status | `cat /workspace/outputs/en_hi/status.json` |
| SSH + log | `tail -f /workspace/outputs/en_hi/train.log` |

### TensorBoard (optional)

On pod (another tmux window or SSH session):

```bash
source .venv/bin/activate
tensorboard --logdir /workspace/outputs/en_hi/runs --host 0.0.0.0 --port 6006
```

On Mac:

```bash
ssh -L 6006:localhost:6006 root@POD_IP -p PORT
```

Browser: http://localhost:6006

---

## Part 8 — If CUDA runs out of memory

Stop training (Ctrl+C in tmux), then:

```bash
python scripts/train.py --config configs/runpod_en_hi.yaml --batch-size 8
```

Or edit `configs/runpod_en_hi.yaml` → `data.batch_size: 8`.

---

## Part 9 — When training finishes

### 9.1 Slack

You get **Training finished** with W&B link and output path.

### 9.2 Download artifacts to Mac

On your **Mac**:

```bash
rsync -avz --progress -e "ssh -p YOUR_PORT" \
  root@YOUR_POD_IP:/workspace/outputs/en_hi/ \
  ~/transformer_outputs/en_hi/
```

Keep:

- `weights/tmodel_*.pt`
- `tokenizers/`
- `config.json`
- `runs/` (TensorBoard)

### 9.3 Stop billing

1. RunPod console → your pod → **Stop** or **Terminate**
2. **Terminate** ends charges; **Stop** may still reserve GPU depending on plan

Do not leave a 4090 running overnight unless you intend to pay for it.

---

## Part 10 — Resume after crash

If the pod died but volume kept files:

```bash
python scripts/train.py --config configs/runpod_en_hi.yaml --preload 05
```

Uses checkpoint `tmodel_05.pt` (epoch 5).

---

## Quick reference

| Item | Value |
|------|--------|
| Config | `configs/runpod_en_hi.yaml` |
| Outputs | `/workspace/outputs/en_hi` |
| W&B entity | `tarun-ssharma` |
| W&B project | `transformer-en-hi` |
| Start command | `bash deploy/runpod_start.sh` |
| tmux session | `train` |
