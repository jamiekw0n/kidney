#!/usr/bin/env bash
#SBATCH --job-name=uni2-expr-dense
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-dense-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-dense-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${1:?Provide completed expression training run directory}"
RUN="$1"
shift
RECORD="$BASE/uni2_slurm_logs/expression_dense_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$BASE/kidney_uni2/expression_aware/"*.py "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$BASE/cache/matplotlib/expression_dense_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
OUT="$BASE/uni2_dense_rois/f03600_expression_job_${SLURM_JOB_ID}"
PY="$BASE/envs/kidney-uni2/bin/python"
"$PY" -m pip freeze > "$RECORD/environment.txt"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" -u "$RECORD/source/dense_from_review.py" \
 --review-root "$BASE/uni2_dense_rois/f03600_cav_review_v1" \
 --weighted-root "$RUN" --out "$OUT" "$@"
printf 'Gallery: %s/index.html\n' "$OUT"
