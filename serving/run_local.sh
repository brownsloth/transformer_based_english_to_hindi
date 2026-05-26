#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SERVE_CONFIG="$ROOT/serving/config/serve.yaml"
export SERVE_DEVICE="${SERVE_DEVICE:-cpu}"
export CORS_ORIGINS="http://localhost:8888,http://127.0.0.1:8888"

if [[ ! -f "$ROOT/serving/artifacts/student/dict_14_fp16.pt" ]]; then
  echo "Downloading artifacts from Hugging Face..."
  python3 "$ROOT/serving/download_artifacts.py"
fi

# Terminal 1: API
echo "Starting API on :8000 (fast + accurate modes, device=$SERVE_DEVICE)"
cd serving/api
pip install -q -r requirements.txt 2>/dev/null || true
uvicorn app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cd "$ROOT/serving/web"
python3 -m http.server 8888 &
WEB_PID=$!

echo "Web UI: http://localhost:8888"
echo "API:    http://localhost:8000/health"
echo "Ctrl+C to stop"

trap "kill $API_PID $WEB_PID 2>/dev/null" EXIT
wait
