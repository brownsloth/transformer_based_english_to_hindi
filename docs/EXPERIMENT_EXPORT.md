# Experiment export — RunPod → local machine

After **both** distillation runs finish (general `lstm_kd` + dictionary `lstm_dict`), compile a full report on the GPU pod while CUDA is still available.

## On RunPod (one command)

```bash
cd /workspace/transformer_from_scratch_translation
git pull   # get distil/report/ scripts

bash distil/report/run_on_runpod.sh
```

**Default = lightweight:** phrase translations on ~40 benchmark phrases × teacher + **all** student checkpoints. **No corpus BLEU.** ~**5–15 min**, ~**$0.10–0.25** GPU.

### Modes

```bash
# Default — recommended when credits are low
bash distil/report/run_on_runpod.sh

# Full export (corpus BLEU 200 + phrases + 50 val examples) ~30–90 min
FULL=1 bash distil/report/run_on_runpod.sh

# Include all teacher checkpoints (larger tarball)
INCLUDE_ALL_TEACHER=1 bash distil/report/run_on_runpod.sh
```

Environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `TEACHER_ARTIFACTS` | `/workspace/outputs/en_hi` | Teacher tokenizers + weights |
| `TEACHER_CKPT` | `10` | Teacher epoch for phrase eval |
| `DICT_MAX_SRC_WORDS` | `3` | Match dict training filter |
| `FULL` | `0` | Set `1` for corpus BLEU export |
| `QUANTIZE` | `0` | Set `1` to include teacher+student quant comparison in bundle |

### Quantization (standalone or in export)

Teacher (~50M transformer) — runs on **CPU**:

```bash
python distil/quantize_teacher.py \
  --teacher-artifacts /workspace/outputs/en_hi \
  --checkpoint 10 \
  --eval-mode phrases \
  --methods fp32_baseline fp16 dynamic_int8_linear \
  --save-models
```

Student (dict checkpoint):

```bash
python distil/quantize_student.py \
  --config distil/configs/lstm_kd_dict.yaml \
  --checkpoint latest \
  --teacher-artifacts /workspace/outputs/en_hi \
  --dict-max-src-words 3 \
  --num-samples 50 \
  --methods fp32_baseline fp16 dynamic_int8_linear \
  --save-models
```

Or include in export bundle:

```bash
QUANTIZE=1 bash distil/report/run_on_runpod.sh
```

Outputs: `quantized/quant_compare_teacher_10.csv`, `quant_compare_dict_08.csv`, saved models under `quantized/models/`.

### Lightweight vs full

| | Lightweight (default) | Full (`FULL=1`) |
|---|----------------------|-----------------|
| Corpus BLEU | skipped | 200 val samples × all epochs |
| Phrase benchmark | ~40 phrases × all ckpts | same |
| Val examples | skipped | 50 side-by-side |
| Time | ~5–15 min | ~30–90 min |

## What gets created

```
distil/exports/experiment_YYYYMMDD_HHMMSS/
  REPORT.md                    ← blog-ready summary
  summary.json
  metadata/
    system.json                ← GPU, PyTorch, platform
    nvidia_smi.txt
    pip_freeze.txt
  metrics/                     (full mode only)
    bleu_teacher.json
    bleu_general.csv
    bleu_dict.csv
  training/
    lstm_kd_loss.csv + .png
    lstm_dict_loss.csv + .png
  inferences/
    phrases_benchmark.json     ← teacher + all student ckpts × ~40 phrases
    phrases_comparison.md      ← readable table (primary metric in lightweight mode)
    val_examples_general.json  ← full mode only
  artifacts/
    tokenizers/
    teacher_weights/tmodel_10.pt
    student_weights/lstm_kd/*.pt
    student_weights/lstm_dict/*.pt
    logs/
    configs/

distil/exports/experiment_bundle.tar.gz   ← download this
```

## Download to your Mac

From RunPod pod connect panel, note **SSH IP** and **port**.

```bash
scp -P <PORT> root@<POD_IP>:/workspace/transformer_from_scratch_translation/distil/exports/experiment_bundle.tar.gz ~/Downloads/
```

Extract into your repo or anywhere:

```bash
mkdir -p ~/Downloads/en-hi-experiment
tar -xzf ~/Downloads/experiment_bundle.tar.gz -C ~/Downloads/en-hi-experiment
open ~/Downloads/en-hi-experiment/experiment_*/REPORT.md
```

## Also sync code (git)

Artifacts tarball does **not** replace git — push from pod or pull on Mac:

```bash
# on pod (if you committed locally during work)
git add distil/ && git commit -m "Distillation experiment" && git push

# on Mac
git pull
```

## Local inference after download

Copy tokenizers + weights from the bundle:

```bash
EXP=~/Downloads/en-hi-experiment/experiment_20260526_120000
cp -r "$EXP/artifacts/tokenizers" ./local_artifacts/tokenizers
cp "$EXP/artifacts/student_weights/lstm_dict/dict_09.pt" ./local_artifacts/

python distil/infer.py \
  --config distil/configs/lstm_kd_dict.yaml \
  --checkpoint /path/to/dict_09.pt \
  ...
```

Easier: point `--teacher-artifacts` at a dir containing `tokenizers/` copied from the bundle.

## Blog checklist

From the export you can directly cite:

- [ ] GPU model + VRAM (`metadata/system.json`)
- [ ] Teacher vs student param counts (`summary.json`)
- [ ] Loss curves (`training/*.png`)
- [ ] BLEU tables (`metrics/*.csv`)
- [ ] Phrase qualitative table (`inferences/phrases_comparison.md`)
- [ ] Random val examples (`inferences/val_examples_general.json`)
- [ ] What worked / what didn’t (`REPORT.md` next steps)
