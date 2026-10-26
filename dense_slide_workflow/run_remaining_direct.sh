#!/usr/bin/env bash
# Run inside the existing two-GPU code-server allocation. No sbatch submission.
set -euo pipefail
BASE=/scratch/groups/emmalu/hande_annotation
PY="$BASE/envs/kidney-uni2/bin/python"
RUN="${1:?Provide dense experiment directory}"
CODE="$RUN/source"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
N=$("$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1]);c=json.loads((root/'experiment.json').read_text())
sys.path.insert(0,str(root/'source'))
from validate import validate
if torch.cuda.device_count()<2:raise RuntimeError('Need two visible allocated GPUs')
print('GPUs: '+', '.join(torch.cuda.get_device_name(i) for i in range(2)),file=sys.stderr)
validate(root/'cache'/c['tasks'][0]['sample_id'])
for t in c['tasks'][1:]:
    for stage in ('prepared','cache'):
        p=root/stage/t['sample_id']
        if p.exists():raise FileExistsError(f'Existing output: {p}; do not duplicate or overwrite tasks')
print(len(c['tasks']))
PY
)
LOG="$RUN/direct_remaining_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$LOG"
printf 'Log directory: %s\n' "$LOG"
worker() (
 set -euo pipefail
 GPU="$1"
 for ((TASK=GPU+1; TASK<N; TASK+=2)); do
  printf 'GPU %s: preparing task %s at %s\n' "$GPU" "$TASK" "$(date -u +%FT%TZ)"
  "$PY" -u "$CODE/workflow.py" prepare --run "$RUN" --task-id "$TASK" > "$LOG/prepare_${TASK}.log" 2>&1
  printf 'GPU %s: extracting task %s at %s\n' "$GPU" "$TASK" "$(date -u +%FT%TZ)"
  "$PY" -u "$CODE/workflow.py" extract --run "$RUN" --task-id "$TASK" --device "cuda:$GPU" > "$LOG/extract_${TASK}.log" 2>&1
  printf 'GPU %s: completed task %s\n' "$GPU" "$TASK"
 done
)
worker 0 > "$LOG/worker_0.log" 2>&1 &
P0=$!
worker 1 > "$LOG/worker_1.log" 2>&1 &
P1=$!
# Wait for both workers even if one fails; never report false success.
S0=0; wait "$P0" || S0=$?
S1=0; wait "$P1" || S1=$?
if (( S0 || S1 )); then
 printf 'FAILED: worker0=%s worker1=%s. Inspect %s; outputs preserved.\n' "$S0" "$S1" "$LOG" >&2
 exit 1
fi
"$PY" "$CODE/workflow.py" collect-cache --run "$RUN" > "$LOG/collection.log" 2>&1
printf 'SUCCESS: all patient caches validated. Report: %s/collection.log\n' "$LOG"
