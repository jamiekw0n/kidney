#!/usr/bin/env bash
#SBATCH --job-name=uni2-filter-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-check-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-check-%j.err

# CPU-only filtering. Add your cluster's required account/partition at submission.
# Create uni2_slurm_logs BEFORE sbatch: Slurm opens log files before this script runs.
set -euo pipefail

BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
UNI2_OUTPUT="$BASE/uni2_filter_checks/job_${SLURM_JOB_ID}"
UNI2_RECORD="$BASE/uni2_slurm_logs/job_${SLURM_JOB_ID}"
mkdir -p "$UNI2_RECORD"

# Calling the environment's Python directly does not require interactive conda init.
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CONDA_PKGS_DIRS="$BASE/cache/conda"
export PIP_CACHE_DIR="$BASE/cache/pip"
export HF_HOME="$BASE/cache/huggingface"
export TMPDIR="$BASE/cache/tmp/filter_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

trap 'uni2_status=$?; printf "exit_code=%s\nfinished_utc=%s\noutput=%s\n" "$uni2_status" "$(date -u +%FT%TZ)" "$UNI2_OUTPUT" > "$UNI2_RECORD/status.txt"; exit "$uni2_status"' EXIT

printf 'Job: %s\nNode: %s\nStart UTC: %s\nOutput: %s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" "$UNI2_OUTPUT"
[[ -x "$UNI2_PYTHON" ]] || { echo 'Environment Python missing'; exit 1; }
[[ -f "$UNI2_CODE/cohort_prepare.py" ]] || { echo 'Updated cohort_prepare.py missing'; exit 1; }
[[ -f "$BASE/uni2_panel.json" ]] || { echo 'Marker panel missing'; exit 1; }
[[ ! -e "$UNI2_OUTPUT" ]] || { echo 'Output exists; preserving previous results'; exit 1; }
cp "$0" "$UNI2_RECORD/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RECORD/slurm_job.txt"
"$UNI2_PYTHON" -c 'import sys,numpy,PIL,tifffile,zarr; print("Python:",sys.executable); print("Image-reading imports passed", flush=True)'

"$UNI2_PYTHON" -u "$UNI2_CODE/cohort_prepare.py" \
  --base-dir "$BASE" \
  --panel "$BASE/uni2_panel.json" \
  --out "$UNI2_OUTPUT" \
  --mpp 0.5073077046562237 \
  --samples-as-patients \
  --assume-all-valid \
  --samples M5172118 M5173023 M5173144 \
  --stride 128 \
  --max-per-sample 5000 \
  --min-context 0.5 \
  --min-center 0.5 \
  --seed 42

"$UNI2_PYTHON" -u "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_OUTPUT/M5172118" \
  "$UNI2_OUTPUT/M5173023" \
  "$UNI2_OUTPUT/M5173144"
printf 'Filtering and prepared-target validation completed. Review filter_review.html in each sample directory.\n'
