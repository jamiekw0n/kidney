#!/usr/bin/env bash
#SBATCH --job-name=uni2-slide-p99
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/slide-p99-%A_%a.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/slide-p99-%A_%a.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array=...}"
: "${1:?Provide the NEW normalization plan directory}"
RUN="$1"
PY="$BASE/envs/kidney-uni2/bin/python"
RECORD="$BASE/uni2_slurm_logs/slide_p99_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RECORD/source"
cp "$BASE/kidney_uni2/slide_normalization/"*.py "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PY" -m pip freeze > "$RECORD/environment.txt"
trap 'status=$?; printf "exit_code=%s\nplan=%s\ntask=%s\n" "$status" "$RUN" "$SLURM_ARRAY_TASK_ID" > "$RECORD/status.txt"; exit "$status"' EXIT
"$PY" -u "$RECORD/source/cohort.py" task --run "$RUN" --task-id "$SLURM_ARRAY_TASK_ID"
