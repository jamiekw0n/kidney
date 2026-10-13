#!/usr/bin/env bash
#SBATCH --job-name=uni2-extract-three
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/extract-three-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/extract-three-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_PREPARED="${UNI2_PREPARED_ROOT:-$BASE/uni2_filter_checks/job_47524714}"
UNI2_OUTPUT="$BASE/uni2_feature_checks/job_${SLURM_JOB_ID}"
UNI2_RECORD="$BASE/uni2_slurm_logs/extract_job_${SLURM_JOB_ID}"
mkdir -p "$UNI2_RECORD"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="$BASE/cache/huggingface"
export TMPDIR="$BASE/cache/tmp/extract_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'uni2_status=$?; printf "exit_code=%s\noutput=%s\n" "$uni2_status" "$UNI2_OUTPUT" > "$UNI2_RECORD/status.txt"; exit "$uni2_status"' EXIT
cp "$0" "$UNI2_RECORD/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RECORD/slurm_job.txt"
printf 'Prepared input: %s\nFeature output: %s\n' "$UNI2_PREPARED" "$UNI2_OUTPUT"
# Require all intended samples, not merely whichever manifests exist when submitted.
"$UNI2_PYTHON" - "$UNI2_PREPARED" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
status=json.loads((root/'cohort_status.json').read_text())
if not status.get('finished') or status.get('failed_or_excluded'):
    raise RuntimeError('Filtering must finish successfully before extraction')
for sample in ('M5172118','M5173023','M5173144'):
    meta=json.loads((root/sample/'manifest.json').read_text())
    if not meta.get('complete') or meta.get('diagnostic_only'):
        raise RuntimeError(f'Incomplete or diagnostic-only sample: {sample}')
print('All three filtering outputs complete and non-diagnostic',flush=True)
PY
"$UNI2_PYTHON" "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_PREPARED/M5172118" "$UNI2_PREPARED/M5173023" "$UNI2_PREPARED/M5173144"
"$UNI2_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:",torch.cuda.get_device_name(0),flush=True)'
"$UNI2_PYTHON" -u "$UNI2_CODE/cohort_extract.py" \
  --prepared-root "$UNI2_PREPARED" --out "$UNI2_OUTPUT" \
  --checkpoint "$BASE/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin" \
  --revision d517a8dd47902dd7c308b3c36f63bce47e7b9a43 \
  --he-cache-gb 2 --batch-size 16 --device cuda
"$UNI2_PYTHON" "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_OUTPUT/M5172118" "$UNI2_OUTPUT/M5173023" "$UNI2_OUTPUT/M5173144"
echo 'Frozen feature extraction and cache validation complete; no head training performed.'
