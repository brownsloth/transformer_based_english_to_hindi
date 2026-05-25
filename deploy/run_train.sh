#!/usr/bin/env bash
# Run training in background with nohup (survives SSH disconnect).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/en_hi_translation.yaml}"
LOG_DIR="${LOG_DIR:-outputs/en_hi}"
PID_FILE="$LOG_DIR/train.pid"
NOHUP_LOG="$LOG_DIR/nohup_train.log"

source .venv/bin/activate 2>/dev/null || true
mkdir -p "$LOG_DIR"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

nohup python scripts/train.py --config "$CONFIG" "$@" >> "$NOHUP_LOG" 2>&1 &
echo $! > "$PID_FILE"

echo "Training started (PID $(cat "$PID_FILE"))"
echo "  nohup log: $NOHUP_LOG"
echo "  train log: $LOG_DIR/train.log"
echo "  status:    $LOG_DIR/status.json"
echo "  stop:      kill \$(cat $PID_FILE)"
