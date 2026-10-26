#!/usr/bin/env bash
#SBATCH --job-name=uni2-expr-cv
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-cv-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-cv-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
PY="$BASE/envs/kidney-uni2/bin/python"
CODE="$BASE/kidney_uni2/expression_aware"
: "${SLURM_JOB_ID:?Submit through sbatch}"
OUT="$BASE/uni2_expression_runs/job_${SLURM_JOB_ID}"
RECORD="$BASE/uni2_slurm_logs/expression_job_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$CODE/"*.py "$CODE/"*.md "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PY" -m pip freeze > "$RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$RECORD/slurm_job.txt"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" -m unittest discover -s "$RECORD/source" -p 'test_*.py' -v
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
printf 'Output: %s\n' "$OUT"
"$PY" -u "$RECORD/source/cross_validate.py" \
 --cache-root "$BASE/uni2_cohort_cache/cohort39_v1" \
 --reference-folds "$BASE/uni2_cv_runs/cohort39_mlp_job_47815087/folds.json" \
 --out "$OUT" --folds 5 --hidden 256 --seed 42 --epochs 100 --patience 10 \
 --batch-size 256 --device cuda --expression-quantile 0.9 --max-expression-weight 5 "$@"
printf 'Complete: %s\n' "$OUT"
