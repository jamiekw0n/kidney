#!/usr/bin/env bash
#SBATCH --job-name=uni2-slide-p99-cv
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/slide-p99-cv-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/slide-p99-cv-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${1:?Provide normalization plan directory after successful collect}"
RUN="$1"; shift
MODE="${1:-weighted}"
if (($#)); then shift; fi
case "$MODE" in weighted|weighted_fds) ;; *) echo 'Mode must be weighted or weighted_fds' >&2; exit 2;; esac
CODE="$BASE/kidney_uni2/slide_normalization"
PY="$BASE/envs/kidney-uni2/bin/python"
OUT="$BASE/uni2_slide_p99_runs/${MODE}_job_${SLURM_JOB_ID}"
RECORD="$BASE/uni2_slurm_logs/slide_p99_cv_${MODE}_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$CODE/"*.py "$CODE/"*.md "$CODE/"*.txt "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PY" -m pip freeze > "$RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$RECORD/slurm_job.txt"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" "$RECORD/source/cohort.py" collect --run "$RUN" > "$RECORD/collection.json"
"$PY" -m compileall -q "$RECORD/source"
"$PY" -m unittest discover -s "$RECORD/source" -v > "$RECORD/tests.log" 2>&1
"$PY" "$RECORD/source/doctor.py" --device cuda > "$RECORD/doctor.json"
printf 'Mode: %s\nOutput: %s\nSource snapshot: %s\n' "$MODE" "$OUT" "$RECORD/source"
"$PY" -u "$RECORD/source/cross_validate.py" \
 --cache-root "$RUN/cache" \
 --reference-folds "$RUN/folds.json" \
 --out "$OUT" --mode "$MODE" --device cuda "$@"
printf 'Complete: %s\n' "$OUT"
