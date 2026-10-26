"""Target-free inference on existing caches with a patient-held-out new-format head."""
import argparse
from pathlib import Path
import numpy as np
from data import read_meta,scale
from io_utils import SIGNATURE,independent_output,predict
from audit import snapshot,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--caches',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--role',choices=['val','test'],default='test')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args()
    if a.batch_size<1:raise ValueError('Positive batch size required')
    from model import load_checkpoint
    model,checkpoint=load_checkpoint(a.checkpoint,a.device)
    roots=[Path(r).resolve() for r in a.caches]
    out=independent_output(a.out,roots+[Path(a.checkpoint).resolve().parent])
    metas=[read_meta(r) for r in roots]
    if len({m['sample_id'] for m in metas})!=len(metas):raise ValueError('Duplicate samples')
    for m in metas:
        pid=m['patient_id']
        if pid not in checkpoint['split'][a.role] or pid in checkpoint['split']['train']:
            raise ValueError('Patient not held out in requested role')
        if m['panel']!=checkpoint['panel'] or any(m.get(k)!=checkpoint['cache_signature'][k] for k in SIGNATURE):
            raise ValueError('Cache/head provenance mismatch')
    out.mkdir(parents=True);snapshot(out,vars(a))
    for root,m in zip(roots,metas):
        dest=out/m['sample_id'];dest.mkdir()
        for name in m['shards']:
            with np.load(root/name,allow_pickle=False) as z:
                x=z['features'].astype(np.float32);xy=z['xy'].copy();target=z['target'].copy()
            if x.shape!=(len(xy),1536) or not np.isfinite(x).all():raise ValueError('Invalid features')
            raw=scale(predict(model,x,a.device,a.batch_size),checkpoint['scaler'],inverse=True)
            if not np.isfinite(raw).all():raise ValueError('Nonfinite predictions')
            np.savez(dest/name,xy=xy,target=target,prediction=raw)
        write_json(dest/'manifest.json',{**m,'complete':True,'checkpoint':str(Path(a.checkpoint).resolve()),
                    'model_format':checkpoint['format'],'evaluation_role':a.role,'prediction_units':'original measured intensity'})
    write_json(out/'manifest.json',{'complete':True,'samples':[m['sample_id'] for m in metas]})

if __name__=='__main__':main()
