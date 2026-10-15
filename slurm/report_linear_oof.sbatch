#!/usr/bin/env bash
#SBATCH --job-name=uni2-oof-report
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/oof-report-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/oof-report-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_PYTHON="$BASE/envs/kidney-uni2/bin/python"
: "${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/oof_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$UNI2_PYTHON" -u "$BASE/kidney_uni2/report_oof.py" \
  --cv-root "$BASE/uni2_cv_runs/cohort39_linear_job_47804039" \
  --out "$BASE/uni2_oof_reports/linear_job_${SLURM_JOB_ID}" \
  --markers VIM CD10 PCNA CAV CD34 CD31 PODXL --maps-per-fold 1
