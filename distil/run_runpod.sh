#!/usr/bin/env bash
# Run LSTM knowledge distillation on RunPod (uses remaining GPU credits).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

TEACHER_ARTIFACTS="${TEACHER_ARTIFACTS:-/workspace/outputs/en_hi}"
TEACHER_CKPT="${TEACHER_CKPT:-8}"
CONFIG="${CONFIG:-distil/configs/lstm_kd.yaml}"

if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

pip install -q sacrebleu 2>/dev/null || true

echo "Teacher: $TEACHER_ARTIFACTS (epoch $TEACHER_CKPT)"
echo "Config:  $CONFIG"

python distil/train.py \
  --config "$CONFIG" \
  --teacher-artifacts "$TEACHER_ARTIFACTS" \
  --teacher-checkpoint "$TEACHER_CKPT" \
  "$@"
