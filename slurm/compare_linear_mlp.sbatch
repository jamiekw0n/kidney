#!/usr/bin/env bash
#SBATCH --job-name=uni2-compare-heads
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/compare-heads-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/compare-heads-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
# Arguments are required: declared source intensity ceiling, raw GT threshold.
: "${1:?Provide intensity ceiling, e.g. 255 after verifying uint8}"
: "${2:?Provide raw GT threshold, e.g. 0 selects nonzero means, not biological positives}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/compare_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$BASE/envs/kidney-uni2/bin/python" -u "$BASE/kidney_uni2/compare_oof.py" \
 --linear-root "$BASE/uni2_cv_runs/cohort39_linear_job_47804039" \
 --mlp-root "$BASE/uni2_cv_runs/cohort39_mlp_job_47815087" \
 --out "$BASE/uni2_oof_comparisons/job_${SLURM_JOB_ID}" \
 --intensity-max "$1" --positive-threshold "$2"
