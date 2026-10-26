#!/usr/bin/env bash
#SBATCH --job-name=uni2-annotation-xy
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/annotation-xy-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/annotation-xy-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/annotation_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$BASE/envs/kidney-uni2/bin/python" -u "$BASE/kidney_uni2/annotation_coordinates.py" \
 --base-dir "$BASE" --cache-root "$BASE/uni2_cohort_cache/cohort39_v1" \
 --annotation-root "$BASE/PRMNT_SP_annotated_cells_39samples" \
 --sample "${1:-K2110291}" --out "$BASE/uni2_annotation_coordinate_checks/job_${SLURM_JOB_ID}"
