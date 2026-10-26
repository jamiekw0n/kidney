#!/usr/bin/env bash
#SBATCH --job-name=uni2-fixed-scale
#SBATCH --partition=owners
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/fixed-scale-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/fixed-scale-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/fixed_scale_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PY="$BASE/envs/kidney-uni2/bin/python"
OUT="$BASE/uni2_dense_fixed_scale/job_${SLURM_JOB_ID}"
# Extra arguments can select markers or a globally shared alternative vmax.
"$PY" "$BASE/kidney_uni2/dense_fixed_scale/render.py" \
  --scan-root "$BASE/uni2_dense_rois" --out "$OUT" "$@"
