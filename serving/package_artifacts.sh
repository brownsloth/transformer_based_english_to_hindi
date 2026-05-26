#!/usr/bin/env bash
# Copy quantized student + teacher weights and tokenizers into serving/artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT="${1:-$ROOT/experiment_20260526_152743}"
DST="$ROOT/serving/artifacts"

echo "Experiment bundle: $EXPERIMENT"
echo "Dest:              $DST"

STUDENT_FP16="$EXPERIMENT/quantized/models/dict_14_fp16.pt"
TEACHER_INT8="$EXPERIMENT/quantized/models/teacher/tmodel_10_dynamic_int8_linear.pt"
TOK_DIR="$EXPERIMENT/artifacts/tokenizers"

if [[ ! -f "$STUDENT_FP16" ]]; then
  echo "Missing $STUDENT_FP16"
  echo "Run quantize on RunPod or point to experiment export with quantized/models/"
  exit 1
fi
if [[ ! -f "$TEACHER_INT8" ]]; then
  echo "Missing $TEACHER_INT8"
  exit 1
fi
if [[ ! -d "$TOK_DIR" ]]; then
  echo "Missing tokenizers at $TOK_DIR"
  exit 1
fi

mkdir -p "$DST/student" "$DST/teacher" "$DST/tokenizers"

cp "$STUDENT_FP16" "$DST/student/dict_14_fp16.pt"
cp "$TEACHER_INT8" "$DST/teacher/tmodel_10_dynamic_int8_linear.pt"
cp "$TOK_DIR/"*.json "$DST/tokenizers/"

echo "Done. Artifacts:"
ls -lh "$DST/student/" "$DST/teacher/" "$DST/tokenizers/"
