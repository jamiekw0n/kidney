#!/usr/bin/env bash
#SBATCH --job-name=uni2-evaluate-head
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/evaluate-head-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/evaluate-head-%j.err
# Usage: sbatch evaluate_head.sbatch /actual/training/best.pt [val|test]
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
UNI2_CHECKPOINT="${1:?Pass the actual best.pt path as the first script argument}"
UNI2_ROLE="${2:-val}"
UNI2_CACHE_ROOT="${UNI2_EVAL_CACHE_ROOT:-$BASE/uni2_feature_checks/job_47565765}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_OUT="$BASE/uni2_evaluations/job_${SLURM_JOB_ID}"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/evaluation_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR"
"$UNI2_PYTHON" -m py_compile "$UNI2_CODE/evaluate_head.py" "$UNI2_CODE/train.py"
"$UNI2_PYTHON" -c 'import matplotlib; print("Matplotlib:",matplotlib.__version__)'
printf 'Checkpoint: %s\nRole: %s\nEvaluation output: %s\n' "$UNI2_CHECKPOINT" "$UNI2_ROLE" "$UNI2_OUT"
# Match supplied caches to the checkpoint split; support both two- and three-patient pilots.
mapfile -t UNI2_CACHES < <("$UNI2_PYTHON" - "$UNI2_CHECKPOINT" "$UNI2_CACHE_ROOT" <<'PY'
import json,sys
from pathlib import Path
import torch
checkpoint=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
patients=set().union(*map(set,checkpoint['split'].values()))
for manifest in sorted(Path(sys.argv[2]).glob('*/manifest.json')):
    meta=json.loads(manifest.read_text())
    if meta['patient_id'] in patients: print(manifest.parent)
PY
)
if [[ ${#UNI2_CACHES[@]} -eq 0 ]]; then echo 'No matching caches found'; exit 1; fi
"$UNI2_PYTHON" -u "$UNI2_CODE/evaluate_head.py" \
  --checkpoint "$UNI2_CHECKPOINT" --caches "${UNI2_CACHES[@]}" \
  --role "$UNI2_ROLE" --out "$UNI2_OUT" --device cpu \
  --markers CD10 VIM NES UMOD DAPI
cp "$0" "$UNI2_OUT/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_OUT/environment.txt"
echo 'Evaluation completed. Inspect marker_summary.csv and sample index.html galleries.'
