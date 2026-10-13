#!/usr/bin/env bash
#SBATCH --job-name=uni2-train2-val1
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/train2-val1-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/train2-val1-%j.err
# Reuse all three completed feature caches; keep prior validation patient M5173023.
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
UNI2_CACHE="$BASE/uni2_feature_checks/job_47565765"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_RUN="$BASE/uni2_train2_val1_tests/job_${SLURM_JOB_ID}"
[[ ! -e "$UNI2_RUN" ]] || { echo 'Run exists'; exit 1; }
mkdir -p "$UNI2_RUN"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
trap 'uni2_status=$?; printf "exit_code=%s\nrun=%s\n" "$uni2_status" "$UNI2_RUN" > "$UNI2_RUN/status.txt"; exit "$uni2_status"' EXIT
cp "$0" "$UNI2_RUN/submitted_script.sbatch"
"$UNI2_PYTHON" -m py_compile "$UNI2_CODE/train.py" "$UNI2_CODE/predict.py" "$UNI2_CODE/metrics.py"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RUN/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RUN/slurm_job.txt"
printf '{"train":["M5172118","M5173144"],"val":["M5173023"],"test":[]}\n' > "$UNI2_RUN/split.json"
printf 'Reuse cache: %s\nRun output: %s\n' "$UNI2_CACHE" "$UNI2_RUN"
"$UNI2_PYTHON" "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_CACHE/M5172118" "$UNI2_CACHE/M5173144" "$UNI2_CACHE/M5173023" --split "$UNI2_RUN/split.json"
"$UNI2_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
"$UNI2_PYTHON" -u "$UNI2_CODE/train.py" \
  --caches "$UNI2_CACHE/M5172118" "$UNI2_CACHE/M5173144" "$UNI2_CACHE/M5173023" \
  --split "$UNI2_RUN/split.json" --out "$UNI2_RUN/training" \
  --hidden 0 --epochs 30 --patience 5 --batch-size 256 --seed 42 --device cuda \
  2>&1 | tee "$UNI2_RUN/training.log"
"$UNI2_PYTHON" -u "$UNI2_CODE/predict.py" \
  --cache "$UNI2_CACHE/M5173023" --heads "$UNI2_RUN/training/best.pt" \
  --out "$UNI2_RUN/validation_predictions" --device cuda
"$UNI2_PYTHON" - "$UNI2_RUN" "$UNI2_CACHE" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root,cache=map(Path,sys.argv[1:]); measured=cache/'M5173023'
meta=json.loads((measured/'manifest.json').read_text());rows=0
for name in meta['shards']:
    with np.load(measured/name,allow_pickle=False) as z, np.load(root/'validation_predictions'/name,allow_pickle=False) as p:
        np.testing.assert_array_equal(z['xy'],p['xy'])
        if p['prediction'].shape!=z['target'].shape or not np.isfinite(p['prediction']).all():
            raise RuntimeError('Invalid validation predictions')
        rows+=len(z['xy'])
report=json.loads((root/'training/validation_metrics.json').read_text())
wins=sum(r['patient_mean_mae']<report['training_mean_baseline'][m]['patient_mean_mae'] for m,r in report['model'].items())
summary={'pipeline_passed':True,'source_cache':str(cache),'validation_rows':rows,'markers':len(meta['panel']),
    'markers_lower_mae_than_baseline':wins,'training_patients':['M5172118','M5173144'],'validation_patients':['M5173023'],'evaluation':'two training patients, one validation patient; not independent testing or CV'}
(root/'pipeline_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
print(f'{"Marker":12} {"Model MAE":>12} {"Baseline MAE":>14} {"Pearson":>10}')
for marker,r in report['model'].items():
    baseline=report['training_mean_baseline'][marker]['patient_mean_mae']
    corr=r['patient_mean_pearson'];text='undefined' if corr is None else f'{corr:.3f}'
    print(f'{marker:12} {r["patient_mean_mae"]:12.3f} {baseline:14.3f} {text:>10}')
PY
