#!/usr/bin/env bash
#SBATCH --job-name=uni2-dense-f03600
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/dense-f03600-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/dense-f03600-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$BASE/cache/matplotlib/dense_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$BASE/envs/kidney-uni2/bin/python" -u "$BASE/kidney_uni2/dense_roi.py" \
 --base-dir "$BASE" --sample F03600 \
 --cache-root "$BASE/uni2_cohort_cache/cohort39_v1" \
 --linear-root "$BASE/uni2_cv_runs/cohort39_linear_job_47804039" \
 --mlp-root "$BASE/uni2_cv_runs/cohort39_mlp_job_47815087" \
 --checkpoint "$BASE/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin" \
 --roi 3800 27500 6200 31500 --markers COLIV POSTN CD138 \
 --batch-size 16 --device cuda --intensity-max 255 --assume-all-valid \
 --out "$BASE/uni2_dense_rois/f03600_job_${SLURM_JOB_ID}"
