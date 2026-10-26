#!/usr/bin/env bash
#SBATCH --job-name=uni2-embedding-umap
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/embedding-umap-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/embedding-umap-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${1:?Provide verified units: native-pixels or micrometers}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/umap_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$BASE/envs/kidney-uni2/bin/python" -u "$BASE/kidney_uni2/embedding_umap.py" \
 --cache-root "$BASE/uni2_cohort_cache/cohort39_v1" \
 --annotation-root "$BASE/PRMNT_SP_annotated_cells_39samples" \
 --coordinate-units "$1" --out "$BASE/uni2_embedding_umap/job_${SLURM_JOB_ID}" "${@:2}"
