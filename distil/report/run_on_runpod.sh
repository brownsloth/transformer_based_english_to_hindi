#!/usr/bin/env bash
# Run experiment export on RunPod after distillation training completes.
#
# Default: lightweight (phrase translations only, ~5-15 min GPU).
# Full BLEU export: FULL=1 bash distil/report/run_on_runpod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"

TEACHER_ARTIFACTS="${TEACHER_ARTIFACTS:-/workspace/outputs/en_hi}"
TEACHER_CKPT="${TEACHER_CKPT:-10}"
DICT_MAX_SRC_WORDS="${DICT_MAX_SRC_WORDS:-3}"

pip install -q sacrebleu matplotlib 2>/dev/null || true

MODE_ARGS=(--lightweight --tarball)
if [[ "${FULL:-0}" == "1" ]]; then
  MODE_ARGS=(--bleu-samples 200 --val-examples 50 --phrase-mode all --tarball)
fi
if [[ "${QUANTIZE:-0}" == "1" ]]; then
  MODE_ARGS+=(--quantize)
fi

EXTRA=()
if [[ "${INCLUDE_ALL_TEACHER:-0}" == "1" ]]; then
  EXTRA+=(--include-all-teacher-weights)
fi

echo "=== Distillation experiment export ==="
echo "Teacher: $TEACHER_ARTIFACTS (checkpoint $TEACHER_CKPT)"
if [[ "${FULL:-0}" == "1" ]]; then
  echo "Mode:    FULL (corpus BLEU 200 + all-epoch phrases + 50 val examples)"
else
  echo "Mode:    LIGHTWEIGHT (phrase benchmark only, all student epochs)"
fi
echo ""

python distil/report/compile_experiment.py \
  --teacher-artifacts "$TEACHER_ARTIFACTS" \
  --teacher-checkpoint "$TEACHER_CKPT" \
  --dict-max-src-words "$DICT_MAX_SRC_WORDS" \
  "${MODE_ARGS[@]}" \
  "${EXTRA[@]}" \
  "$@"

BUNDLE="$ROOT/distil/exports/experiment_bundle.tar.gz"
if [[ -f "$BUNDLE" ]]; then
  echo ""
  echo "Bundle ready: $BUNDLE"
  ls -lh "$BUNDLE"
  echo ""
  echo "Download from your Mac:"
  echo "  scp -P <PORT> root@<POD_IP>:$BUNDLE ~/Downloads/"
fi
