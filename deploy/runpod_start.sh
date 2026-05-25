#!/usr/bin/env bash
# Start full GPU training on RunPod inside tmux-friendly shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/runpod_en_hi.yaml}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
elif [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -m pip install -q -r requirements.txt wandb 2>/dev/null || pip install -r requirements.txt wandb

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  wandb login --relogin <<< "${WANDB_API_KEY}" 2>/dev/null || true
fi

echo "Config: $CONFIG"
echo "Output: $(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['paths']['output_dir'])")"
echo "W&B: ${WANDB_PROJECT:-from yaml}"
echo "Slack: ${ALERT_WEBHOOK_URL:+enabled}${ALERT_WEBHOOK_URL:-disabled}"

exec python scripts/train.py --config "$CONFIG" "$@"
