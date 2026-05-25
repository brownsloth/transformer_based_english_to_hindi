#!/usr/bin/env bash
# Bootstrap a fresh GPU VM for training.
set -euo pipefail

PROJECT_DIR="${1:-$HOME/transformer_from_scratch_translation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Setting up project at: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if ! command -v python3 &>/dev/null; then
  echo "python3 not found. Install Python $PYTHON_VERSION first."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo ""
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/analyze_seq_len.py"
echo "  ALERT_WEBHOOK_URL='https://...' python scripts/train.py"
echo ""
echo "Monitor remotely:"
echo "  tail -f outputs/en_hi/train.log"
echo "  cat outputs/en_hi/status.json"
echo "  tensorboard --logdir outputs/en_hi/runs --bind_all --port 6006"
