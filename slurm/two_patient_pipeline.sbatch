#!/usr/bin/env bash
#SBATCH --job-name=uni2-two-patient
#SBATCH --partition=emmalu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/groups/emmalu/hande_annotation
#SBATCH --output=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/two-patient-%j.out
#SBATCH --error=/scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/two-patient-%j.err
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
UNI2_CODE="$BASE/kidney_uni2"
UNI2_ENV="$BASE/envs/kidney-uni2"
UNI2_PYTHON="$UNI2_ENV/bin/python"
: "${SLURM_JOB_ID:?Submit with sbatch}"
UNI2_PREPARED="$BASE/uni2_filter_checks/job_47524714"
UNI2_RUN="$BASE/uni2_two_patient_tests/job_${SLURM_JOB_ID}"
[[ ! -e "$UNI2_RUN" ]] || { echo 'Run output exists; preserving it'; exit 1; }
mkdir -p "$UNI2_RUN"
export PATH="$UNI2_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="$BASE/cache/huggingface"
export TMPDIR="$BASE/cache/tmp/two_patient_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'uni2_status=$?; printf "exit_code=%s\nfinished_utc=%s\nrun=%s\n" "$uni2_status" "$(date -u +%FT%TZ)" "$UNI2_RUN" > "$UNI2_RUN/status.txt"; exit "$uni2_status"' EXIT
cp "$0" "$UNI2_RUN/submitted_script.sbatch"
"$UNI2_PYTHON" -m pip freeze > "$UNI2_RUN/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$UNI2_RUN/slurm_job.txt"
printf 'Run: %s\nPrepared inputs: %s\n' "$UNI2_RUN" "$UNI2_PREPARED"
"$UNI2_PYTHON" - "$UNI2_PREPARED" "$UNI2_RUN" "$UNI2_CODE" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
source,run,code=map(Path,sys.argv[1:])
records=[]
for name in ('M5172118','M5173023'):
    path=source/name/'manifest.json'
    meta=json.loads(path.read_text())
    if not meta.get('complete') or meta.get('diagnostic_only'):
        raise RuntimeError('Need completed full filtering sample: '+name)
    records.append({'sample_id':name,'patient_id':meta['patient_id'],'manifest':str(path),
                    'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':meta['retained']})
(run/'inputs.json').write_text(json.dumps({'scope':'two completed samples from a timed-out parent job; partial third sample excluded',
    'inputs':records,'evaluation':'patient holdout smoke test, not cohort cross-validation'},indent=2))
(run/'split.json').write_text(json.dumps({'train':['M5172118'],'val':['M5173023'],'test':[]},indent=2))
snapshot=run/'source_snapshot';snapshot.mkdir()
for path in code.iterdir():
    if path.suffix in ('.py','.md','.txt'): shutil.copy2(path,snapshot/path.name)
print('Both source manifests complete; third patient excluded',flush=True)
PY
"$UNI2_PYTHON" "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_PREPARED/M5172118" "$UNI2_PREPARED/M5173023"
"$UNI2_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:",torch.cuda.get_device_name(0),flush=True)'
for uni2_sample in M5172118 M5173023; do
  printf 'Stage: frozen extraction | sample=%s | UTC=%s\n' "$uni2_sample" "$(date -u +%FT%TZ)"
  "$UNI2_PYTHON" -u "$UNI2_CODE/extract.py" \
    --prepared "$UNI2_PREPARED/$uni2_sample" \
    --checkpoint "$BASE/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin" \
    --revision d517a8dd47902dd7c308b3c36f63bce47e7b9a43 \
    --out "$UNI2_RUN/cache/$uni2_sample" \
    --he-cache-gb 2 --batch-size 16 --device cuda \
    2>&1 | tee "$UNI2_RUN/${uni2_sample}_extraction.log"
done
"$UNI2_PYTHON" "$UNI2_CODE/validate.py" --datasets \
  "$UNI2_RUN/cache/M5172118" "$UNI2_RUN/cache/M5173023" --split "$UNI2_RUN/split.json"
echo 'Stage: train linear head; one training patient, separate validation patient'
"$UNI2_PYTHON" -u "$UNI2_CODE/train.py" \
  --caches "$UNI2_RUN/cache/M5172118" "$UNI2_RUN/cache/M5173023" \
  --split "$UNI2_RUN/split.json" --out "$UNI2_RUN/training" \
  --hidden 0 --epochs 30 --patience 5 --batch-size 256 --seed 42 --device cuda \
  2>&1 | tee "$UNI2_RUN/training.log"
echo 'Stage: export validation predictions (not independent test scoring)'
"$UNI2_PYTHON" -u "$UNI2_CODE/predict.py" \
  --cache "$UNI2_RUN/cache/M5173023" --heads "$UNI2_RUN/training/best.pt" \
  --out "$UNI2_RUN/validation_predictions" --batch-size 256 --device cuda
"$UNI2_PYTHON" - "$UNI2_RUN" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); report=json.loads((root/'training/validation_metrics.json').read_text())
wins=0
print(f'{"Marker":12} {"Model MAE":>12} {"Baseline MAE":>14} {"Pearson":>10}')
for marker,record in report['model'].items():
    mae=record['patient_mean_mae']; baseline=report['training_mean_baseline'][marker]['patient_mean_mae']
    r=record['patient_mean_pearson'];wins+=mae<baseline
    text='undefined' if r is None else f'{r:.3f}'
    print(f'{marker:12} {mae:12.3f} {baseline:14.3f} {text:>10}')
rows=0
cache=root/'cache/M5173023'; output=root/'validation_predictions'
meta=json.loads((cache/'manifest.json').read_text())
for name in meta['shards']:
    with np.load(cache/name,allow_pickle=False) as original, np.load(output/name,allow_pickle=False) as predicted:
        np.testing.assert_array_equal(original['xy'],predicted['xy'])
        if predicted['prediction'].shape!=(len(original['xy']),len(meta['panel'])) or not np.isfinite(predicted['prediction']).all():
            raise RuntimeError('Invalid prediction arrays')
        rows+=len(original['xy'])
summary={'pipeline_passed':True,'markers':len(report['model']),'markers_lower_mae_than_baseline':wins,
         'validation_prediction_rows':rows,'coordinate_check':'exact match','evaluation':'single patient validation; no independent test or cross-validation'}
(root/'pipeline_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
PY
