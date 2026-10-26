#!/usr/bin/env bash
#SBATCH --job-name=uni2-dense-stage
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
PY="$BASE/envs/kidney-uni2/bin/python"
STAGE="$1"
PLAN="$2"
MODE="${3:-weighted}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$BASE/cache/matplotlib/dense_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
case "$STAGE" in
 prepare|extract|train-fold)
  CODE="$PLAN/source"
  ;;
 collect-cache|collect-cv)
  CODE="$PLAN/source"
  ;;
 infer|assemble)
  CODE="" # resolved below from the slide plan
  ;;
esac
if [[ "$STAGE" == infer || "$STAGE" == assemble ]]; then
 CODE=$("$PY" -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["run"])/"source")' "$PLAN/slide.json")
fi
printf 'stage=%s plan=%s node=%s job=%s start=%s\n' "$STAGE" "$PLAN" "${SLURMD_NODENAME:-unknown}" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
case "$STAGE" in
 prepare) "$PY" -u "$CODE/workflow.py" prepare --run "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID" ;;
 extract) "$PY" -u "$CODE/workflow.py" extract --run "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID" ;;
 train-fold) "$PY" -u "$CODE/workflow.py" train-fold --run "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID" --mode "$MODE" ;;
 collect-cache|collect-cv) "$PY" -u "$CODE/workflow.py" "$STAGE" --run "$PLAN" ;;
 infer) "$PY" -u "$CODE/whole_slide.py" infer --slide-plan "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID" ;;
 assemble) "$PY" -u "$CODE/whole_slide.py" assemble --slide-plan "$PLAN" --out "${PLAN}_figures_${SLURM_JOB_ID}" ;;
 *) echo 'Unknown stage' >&2; exit 2 ;;
esac
