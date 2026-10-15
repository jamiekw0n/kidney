#!/usr/bin/env bash
#SBATCH --job-name=uni2-cv39-mlp
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/cv39-mlp-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/cv39-mlp-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
UNI2_CACHE="$BASE/uni2_cohort_cache/cohort39_v1"
UNI2_PLAN="$BASE/uni2_cv_plan39_v1"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_OUT="$BASE/uni2_cv_runs/cohort39_mlp_job_${SLURM_JOB_ID}"
UNI2_RECORD="$BASE/uni2_slurm_logs/cv_mlp_job_${SLURM_JOB_ID}"
mkdir -p "$UNI2_RECORD/source"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Execute one fixed source copy for the whole run, including fold subprocesses.
cp "$UNI2_CODE/"*.py "$UNI2_RECORD/source/"
cp "$0" "$UNI2_RECORD/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RECORD/slurm_job.txt"
trap 'uni2_status=$?; printf "exit_code=%s\noutput=%s\n" "$uni2_status" "$UNI2_OUT" > "$UNI2_RECORD/status.txt"; exit "$uni2_status"' EXIT
"$UNI2_PYTHON" -m compileall -q "$UNI2_RECORD/source"
PYTHONPATH="$UNI2_RECORD/source" "$UNI2_PYTHON" - "$UNI2_CACHE" "$UNI2_PLAN" "$UNI2_RECORD" <<'PY'
import json,sys
from pathlib import Path
import torch
from cross_validate import make_folds
from data import read_meta,partitions
root,plan,record=map(Path,sys.argv[1:])
roots=sorted(p.parent for p in root.glob('*/manifest.json'))
metas=[read_meta(r) for r in roots]
patients={m['patient_id'] for m in metas}
if len(roots)!=39 or len(patients)!=39:
    raise RuntimeError('Expected 39 samples and 39 distinct patient IDs under current mapping')
inventory=json.loads((plan/'inventory.json').read_text())
if {r['sample_id'] for r in inventory}!={m['sample_id'] for m in metas} or {r['patient_id'] for r in inventory}!=patients:
    raise RuntimeError('Current sample/patient inventory differs from reviewed plan')
# Also reject a changed sample-to-patient assignment with otherwise equal ID sets.
if {r['sample_id']:r['patient_id'] for r in inventory}!={m['sample_id']:m['patient_id'] for m in metas}:
    raise RuntimeError('Sample-to-patient mapping changed')
folds=json.loads((plan/'folds.json').read_text())
if folds!=make_folds(patients,5,42,0.2):
    raise RuntimeError('Saved folds do not match intended five-fold seed-42 design')
for fold in folds:partitions(roots,fold)
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
reference=Path('/scratch/groups/emmalu/hande_annotation/uni2_cv_runs/cohort39_linear_job_47804039')
if folds!=json.loads((reference/'folds.json').read_text()):
    raise RuntimeError('MLP patient folds differ from completed linear comparison')
(record/'reviewed_folds.json').write_text(json.dumps(folds,indent=2))
print('Preflight passed: 39 patients; saved fold assignments match',flush=True)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
for index,fold in enumerate(folds):
    print(f'Fold {index}: train={len(fold["train"])} inner_val={len(fold["val"])} outer_test={len(fold["test"])}',flush=True)
PY
printf 'CV output: %s\nFixed source: %s\n' "$UNI2_OUT" "$UNI2_RECORD/source"
"$UNI2_PYTHON" -u "$UNI2_RECORD/source/cross_validate.py" \
  --cache-root "$UNI2_CACHE" --out "$UNI2_OUT" \
  --folds 5 --val-fraction 0.2 --hidden 256 --seed 42 \
  --epochs 100 --patience 10 --batch-size 256 --device cuda \
  --bootstrap-replicates 2000 --bootstrap-seed 42
printf 'Cross-validation and primary OOF evaluation finished: %s\n' "$UNI2_OUT"
