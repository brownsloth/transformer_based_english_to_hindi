# Hindi Jinnie — serving

Simple web UI + FastAPI backend for your trained EN→HI transformer.

```
Browser (Netlify)  ──POST /translate──▶  FastAPI (GPU/CPU server)
  serving/web/                            serving/api/
```

Netlify cannot run PyTorch. **Host the UI on Netlify, the model on RunPod (or similar).**

---

## 1. Model artifacts (Hugging Face)

Production weights live on Hugging Face — no need to commit `.pt` files:

**https://huggingface.co/1starun8-research/en-hi-translation**

Layout matches `serving/artifacts/`:

```
student/dict_14_fp16.pt
teacher/tmodel_10_dynamic_int8_linear.pt
tokenizers/tokenizer_en.json
tokenizers/tokenizer_hi.json
```

### Download locally

```bash
pip install huggingface_hub
python serving/download_artifacts.py
```

Or from experiment export (offline):

```bash
bash serving/package_artifacts.sh experiment_20260526_152743
```

### Refresh Hugging Face after re-quantizing

```bash
bash serving/package_artifacts.sh experiment_20260526_152743
hf upload 1starun8-research/en-hi-translation serving/artifacts \
  --repo-type model \
  --include "student/*" --include "teacher/*" --include "tokenizers/*"
```

---

## 2. Modes

| UI / API `mode` | Model | Typical CPU latency |
|-----------------|-------|---------------------|
| `fast` | LSTM student dict_14 fp16 | ~5–50 ms |
| `accurate` | Transformer teacher int8 linear | ~300–800 ms |

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"good morning","mode":"accurate"}'
```

---

## 3. Test locally

```bash
# From repo root — downloads from HF if artifacts missing
bash serving/run_local.sh
```

Or manually:

```bash
python serving/download_artifacts.py   # if needed
export PYTHONPATH="$(pwd)/src:$(pwd)"
export SERVE_DEVICE=cpu
cd serving/api && uvicorn app:app --reload --port 8000
```

Open http://localhost:8888 — choose **Fast** or **Accurate** in the dropdown.

Health check: http://localhost:8000/health

---

## 4. Deploy API on Railway

**Netlify = UI only.** Railway runs the FastAPI + PyTorch container.

### Steps

1. Push this repo to GitHub (no weights in git — Dockerfile pulls from HF at build time).

2. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub** → select repo.

3. **Settings → Build**
   - Builder: **Dockerfile**
   - Dockerfile path: `serving/Dockerfile`
   - Root directory: `/` (repo root, not `serving/`)

4. **Settings → Variables**
   ```
   SERVE_DEVICE=cpu
   CORS_ORIGINS=https://projects.tarun-ssharma.com
   HF_ARTIFACTS_REPO=1starun8-research/en-hi-translation
   ```
   (`HF_ARTIFACTS_REPO` is optional — same as Dockerfile default.)

5. **Networking → Generate domain** → e.g. `https://en-hi-translation.up.railway.app`

6. Test:
   ```bash
   curl https://YOUR-RAILWAY-URL/health
   curl -X POST https://YOUR-RAILWAY-URL/translate \
     -H "Content-Type: application/json" \
     -d '{"text":"good morning","mode":"accurate"}'
   ```

Build takes a few minutes (PyTorch + ~242 MB HF download). Railway sets `PORT` automatically.

---

## 5. Deploy UI on Netlify

See [netlify/GATSBY_INTEGRATION.md](netlify/GATSBY_INTEGRATION.md).

Quick version:

```bash
cp -r serving/web/* /path/to/gatsby-site/static/hindi-jinnie/
```

Edit `static/hindi-jinnie/config.js`:

```javascript
window.HINDI_JINNIE_API = "https://YOUR-RAILWAY-URL";
```

Push Gatsby repo → Netlify deploys → visit:

**https://projects.tarun-ssharma.com/hindi-jinnie/**

---

## 6. Docker (local test)

```bash
docker build -f serving/Dockerfile -t hindi-jinnie-api .
docker run -p 8000:8000 -e SERVE_DEVICE=cpu hindi-jinnie-api
```

No local `serving/artifacts/` needed — the image downloads from Hugging Face during build.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVE_CONFIG` | `serving/config/serve.yaml` | Model paths + mode config |
| `SERVE_ARTIFACTS_DIR` | `serving/artifacts` | Weights + tokenizers root |
| `HF_ARTIFACTS_REPO` | `1starun8-research/en-hi-translation` | Hugging Face model repo |
| `HF_TOKEN` | — | Optional; for private HF repos |
| `SERVE_DEVICE` | cuda if available | Use `cpu` on Railway |
| `CORS_ORIGINS` | projects.tarun-ssharma.com | Comma-separated |
| `PORT` | `8000` | Set by Railway at runtime |

---

## File layout

```
serving/
  README.md                 ← this file
  Dockerfile
  download_artifacts.py     ← pull weights from Hugging Face
  package_artifacts.sh      ← copy from experiment export (offline)
  run_local.sh
  config/serve.yaml
  artifacts/                ← gitignored weights (local/deploy)
  api/
    app.py                  ← FastAPI
    translator.py
    requirements.txt
  web/
    index.html              ← Hindi Jinnie UI
    styles.css
    app.js
    config.js               ← API URL for frontend
  netlify/
    GATSBY_INTEGRATION.md
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CORS error in browser | Set `CORS_ORIGINS` on API to your Netlify URL |
| `Checkpoint not found` | Run `python serving/download_artifacts.py` or check HF repo |
| Slow first request | Model loads at startup; first call after boot is normal |
| Hindi shows as boxes locally | Browser/font issue; should work in modern Chrome |
