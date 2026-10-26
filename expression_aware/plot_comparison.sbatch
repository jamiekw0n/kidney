#!/usr/bin/env bash
#SBATCH --job-name=uni2-expr-plots
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-plots-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/expression-plots-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${1:?Provide the completed expression run directory as first argument}"
RUN="$1"
shift
CODE="$BASE/kidney_uni2/expression_aware"
OUT="$BASE/uni2_oof_comparisons/expression_job_${SLURM_JOB_ID}"
RECORD="$BASE/uni2_slurm_logs/expression_plots_${SLURM_JOB_ID}"
mkdir -p "$RECORD/source"
cp "$CODE/"*.py "$RECORD/source/"
cp "$0" "$RECORD/submitted.sbatch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$BASE/cache/matplotlib/expression_plots_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR"
PY="$BASE/envs/kidney-uni2/bin/python"
"$PY" -m pip freeze > "$RECORD/environment.txt"
trap 'status=$?; printf "exit_code=%s\noutput=%s\n" "$status" "$OUT" > "$RECORD/status.txt"; exit "$status"' EXIT
# Completion gate before expensive plotting; script additionally verifies all folds,
# patient inventories, measured arrays, xy, and embedding/preprocessing provenance.
"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
for name in ('primary_summary.json','oof_expression_summary.json','folds.json','inventory.json'):
    json.loads((run/name).read_text())
print('Completed expression run:',run,flush=True)
PY
"$PY" -u "$RECORD/source/compare_predictions.py" \
 --baseline-root "$BASE/uni2_cv_runs/cohort39_mlp_job_47815087" \
 --weighted-root "$RUN" --out "$OUT" \
 --intensity-max 255 --positive-threshold 0 "$@"
printf 'Gallery: %s/index.html\n' "$OUT"
