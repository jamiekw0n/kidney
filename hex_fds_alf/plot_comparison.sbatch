#!/usr/bin/env bash
#SBATCH --job-name=uni2-hex-plots
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/hex-plots-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/hex-plots-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
RECORD="$BASE/uni2_slurm_logs/hex_plots_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$BASE/kidney_uni2/hex_fds_alf/"*.py "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="$BASE/cache/matplotlib/hex_plots_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
OUT="$BASE/uni2_oof_comparisons/hex_job_${SLURM_JOB_ID}"
PY="$BASE/envs/kidney-uni2/bin/python"
"$PY" -m pip freeze > "$RECORD/environment.txt"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" -u "$RECORD/source/compare_predictions.py" \
 --baseline-root "$BASE/uni2_hex_runs/control_job_48680643" \
 --candidate-root "$BASE/uni2_hex_runs/fds_alf_job_48680648" \
 --baseline-label 'Matched control' --candidate-label 'FDS + ALF' \
 --out "$OUT" --intensity-max 255 --positive-threshold 0 \
 --markers CAV CD8 CD138 COLIV POSTN VIM "$@"
printf 'Gallery: %s/index.html\n' "$OUT"
