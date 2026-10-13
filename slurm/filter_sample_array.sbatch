#!/usr/bin/env bash
#SBATCH --job-name=uni2-filter-sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Array size, partition, memory, time and log paths supplied by filter_array.py.
set -euo pipefail
UNI2_PLAN="${1:?Pass array plan directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
UNI2_BASE=/scratch/groups/emmalu/hande_annotation
UNI2_ENV="$UNI2_BASE/envs/kidney-uni2"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$UNI2_BASE/cache/tmp/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
printf 'Array job=%s task=%s node=%s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)"
"$UNI2_ENV/bin/python" -m pip freeze > "$UNI2_PLAN/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_environment.txt"
"$UNI2_ENV/bin/python" -u "$UNI2_PLAN/source_snapshot/filter_array.py" task \
  --run "$UNI2_PLAN" --task-id "$SLURM_ARRAY_TASK_ID"
