#!/usr/bin/env bash
#SBATCH --job-name=uni2-filter-cohort
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-cohort-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-cohort-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_OUT="$BASE/uni2_cohort_prepared/job_${SLURM_JOB_ID}"
UNI2_RECORD="$BASE/uni2_slurm_logs/cohort_job_${SLURM_JOB_ID}"
mkdir -p "$UNI2_RECORD"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$BASE/cache/tmp/cohort_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'uni2_status=$?; printf "exit_code=%s\noutput=%s\n" "$uni2_status" "$UNI2_OUT" > "$UNI2_RECORD/status.txt"; exit "$uni2_status"' EXIT
cp "$0" "$UNI2_RECORD/submitted_script.sbatch"
"$UNI2_PYTHON" -m py_compile "$UNI2_CODE/cohort_prepare.py" "$UNI2_CODE/raw_safety.py"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RECORD/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RECORD/slurm_job.txt"
printf 'Raw images read-only. Derived outputs: %s\n' "$UNI2_OUT"
"$UNI2_PYTHON" -u "$UNI2_CODE/cohort_prepare.py" \
  --base-dir "$BASE" --panel "$BASE/uni2_panel.json" --out "$UNI2_OUT" \
  --mpp 0.5073077046562237 --samples-as-patients --assume-all-valid \
  --stride 128 --max-per-sample 5000 --min-context 0.5 --min-center 0.5 \
  --he-cache-gb 8 --shard-size 1000 --progress-every 100 --seed 42
