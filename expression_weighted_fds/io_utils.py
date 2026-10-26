"""Read-only cache handling and resource guards; no raw-image access."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
from data import read_meta


SIGNATURE=('encoder','revision','checkpoint_sha256','transform','input_patch','he_mpp','target_block','selection_policy')


def independent_output(out,inputs):
    out=Path(out).resolve()
    for source in inputs:
        source=Path(source).resolve()
        if out==source or source in out.parents or out in source.parents:
            raise ValueError('Output overlaps input directory: '+str(source))
    if out.exists():raise FileExistsError(out)
    return out


def estimate(roots):
    # Every shard validated separately elsewhere. Numeric peak conservative for
    # concatenation, features+targets, scaled targets, CPU predictions, weights.
    rows=0;size=0
    for root in roots:
        meta=read_meta(root)
        for name in meta['shards']:
            with np.load(Path(root)/name,allow_pickle=False) as z:
                n=len(z['target']);rows+=n;size+=n*(1536+len(meta['panel']))*4
    return {'rows':rows,'numeric_bytes':size,'estimated_numeric_peak_bytes':size*3,
            'excludes':'Framework, CUDA context, optimizer, FDS buffers, Python and OS overhead'}


def resource_check(roots,out,max_gb):
    if not np.isfinite(max_gb) or max_gb<=0:raise ValueError('Invalid numeric memory limit')
    report=estimate(roots)
    if report['estimated_numeric_peak_bytes']>max_gb*1e9:raise MemoryError(report)
    ancestor=Path(out)
    while not ancestor.exists():ancestor=ancestor.parent
    report['disk_free_bytes']=shutil.disk_usage(ancestor).free
    report['slurm_job_id']=os.environ.get('SLURM_JOB_ID')
    report['slurm_mem_per_node_mb']=os.environ.get('SLURM_MEM_PER_NODE')
    return report


def load_group(roots):
    xs,ys,patients=[],[],[]
    for root in roots:
        meta=read_meta(root)
        for name in meta['shards']:
            with np.load(Path(root)/name,allow_pickle=False) as z:
                x=z['features'].astype(np.float32);y=z['target'].astype(np.float32)
            if x.shape!=(len(y),1536) or y.shape!=(len(x),len(meta['panel'])):
                raise ValueError('Invalid cache shapes')
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite cache arrays')
            xs.append(x);ys.append(y);patients.extend([meta['patient_id']]*len(y))
    if not xs or sum(map(len,xs))==0:raise ValueError('Empty partition')
    return np.concatenate(xs),np.concatenate(ys),np.asarray(patients)


def predict(model,x,device,batch_size):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size]).to(device)).cpu().numpy()
                               for i in range(0,len(x),batch_size)])
