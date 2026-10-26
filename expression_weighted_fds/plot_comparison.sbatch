#!/usr/bin/env bash
#SBATCH --job-name=uni2-weighted-fds-plots
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/weighted-fds-plots-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/weighted-fds-plots-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit using sbatch}"
PY="$BASE/envs/kidney-uni2/bin/python"
RECORD="$BASE/uni2_slurm_logs/weighted_fds_plots_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$BASE/kidney_uni2/expression_weighted_fds/"*.py "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="$BASE/cache/matplotlib/weighted_fds_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
OUT="$BASE/uni2_oof_comparisons/weighted_fds_job_${SLURM_JOB_ID}"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
# Resolve without empty arrays: Bash 3.2/4.3 can reject them under set -u.
if (($#)) && [[ "$1" != --* ]]; then
 : "${2:?Provide both weighted and weighted_fds run paths}"
 "$PY" "$RECORD/source/resolve_runs.py" --run-root "$BASE/uni2_weighted_fds_runs" \
   --baseline-root "$1" --candidate-root "$2" > "$RECORD/selected_runs.txt"
 shift 2
else
 "$PY" "$RECORD/source/resolve_runs.py" --run-root "$BASE/uni2_weighted_fds_runs" > "$RECORD/selected_runs.txt"
fi
# read is available in older Bash releases; mapfile is not required.
{
 IFS= read -r CONTROL
 IFS= read -r CANDIDATE
} < "$RECORD/selected_runs.txt"
"$PY" -m pip freeze > "$RECORD/environment.txt"
printf 'Weighted: %s\nWeighted + FDS: %s\n' "$CONTROL" "$CANDIDATE"
"$PY" "$RECORD/source/verify_pair.py" --baseline-root "$CONTROL" --candidate-root "$CANDIDATE" > "$RECORD/pair_check.json"
"$PY" -u "$RECORD/source/compare_predictions.py" \
 --baseline-root "$CONTROL" --candidate-root "$CANDIDATE" --out "$OUT" \
 --intensity-max 255 --positive-threshold 0 --markers CAV CD8 CD138 COLIV POSTN VIM "$@"
printf 'Gallery: %s/index.html\n' "$OUT"
