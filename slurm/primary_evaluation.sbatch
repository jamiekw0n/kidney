#!/usr/bin/env bash
#SBATCH --job-name=uni2-primary-eval
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/primary-eval-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/primary-eval-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_PYTHON="$BASE/envs/kidney-uni2/bin/python"
: "${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
"$UNI2_PYTHON" -m py_compile "$BASE/kidney_uni2/evaluate_head.py" "$BASE/kidney_uni2/primary_evaluation.py" "$BASE/kidney_uni2/metrics.py"
"$UNI2_PYTHON" -u "$BASE/kidney_uni2/evaluate_head.py" \
  --checkpoint "$BASE/uni2_train2_val1_tests/job_47664301/training/best.pt" \
  --caches "$BASE/uni2_feature_checks/job_47565765/M5172118" \
    "$BASE/uni2_feature_checks/job_47565765/M5173144" \
    "$BASE/uni2_feature_checks/job_47565765/M5173023" \
  --role val --device cpu --no-plots \
  --bootstrap-replicates 2000 --bootstrap-seed 42 \
  --out "$BASE/uni2_primary_evaluations/job_${SLURM_JOB_ID}"
cp "$0" "$BASE/uni2_primary_evaluations/job_${SLURM_JOB_ID}/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$BASE/uni2_primary_evaluations/job_${SLURM_JOB_ID}/environment.txt"
