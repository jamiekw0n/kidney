#!/usr/bin/env bash
#SBATCH --job-name=uni2-feature-sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
set -euo pipefail
UNI2_PLAN="${1:?Pass feature array directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit as GPU array}"
UNI2_BASE=/scratch/groups/emmalu/hande_annotation
UNI2_ENV="$UNI2_BASE/envs/kidney-uni2"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="$UNI2_BASE/cache/huggingface"
export TMPDIR="$UNI2_BASE/cache/tmp/features_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
"$UNI2_ENV/bin/python" -m pip freeze > "$UNI2_PLAN/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_environment.txt"
printf 'Feature array=%s task=%s node=%s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)"
"$UNI2_ENV/bin/python" -u "$UNI2_PLAN/source_snapshot/feature_array.py" task \
  --run "$UNI2_PLAN" --task-id "$SLURM_ARRAY_TASK_ID"
