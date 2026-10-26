#!/usr/bin/env bash
#SBATCH --job-name=uni2-quant-eval
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/quant-eval-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/quant-eval-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
RECORD="$BASE/uni2_slurm_logs/quant_eval_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$BASE/kidney_uni2/cohort_evaluation/"*.py "$BASE/kidney_uni2/cohort_evaluation/"*.md "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
PY="$BASE/envs/kidney-uni2/bin/python"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
OUT="$BASE/uni2_quantitative_evaluation/job_${SLURM_JOB_ID}"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" -m pip freeze > "$RECORD/environment.txt"
"$PY" -m unittest discover -s "$RECORD/source" -v > "$RECORD/tests.log" 2>&1
"$PY" -u "$RECORD/source/evaluate.py" --base-dir "$BASE" --out "$OUT" "$@"
printf 'Report: %s/REPORT.md\n' "$OUT"
