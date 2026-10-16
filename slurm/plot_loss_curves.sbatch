#!/usr/bin/env bash
#SBATCH --job-name=uni2-loss-curves
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/loss-curves-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/loss-curves-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/loss_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
"$BASE/envs/kidney-uni2/bin/python" -u "$BASE/kidney_uni2/plot_loss_curves.py" \
 --cv-roots "$BASE/uni2_cv_runs/cohort39_linear_job_47804039" "$BASE/uni2_cv_runs/cohort39_mlp_job_47815087" \
 --labels Linear MLP --out "$BASE/uni2_loss_curves/job_${SLURM_JOB_ID}"
