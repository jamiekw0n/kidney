"""Recompute primary evaluation from completed CV prediction arrays without retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
from metrics import metrics
from primary_evaluation import save_primary
from audit import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=42)
    a=p.parse_args()
    if a.bootstrap_replicates<1 or a.bootstrap_seed<0:p.error('Invalid bootstrap settings')
    root=Path(a.cv_root);folds=json.loads((root/'folds.json').read_text())
    expected={};by_patient={};panel=None;samples=set()
    for i,split in enumerate(folds):
        if set(split['test'])&(set(split['train'])|set(split['val'])):raise ValueError('Fold patient leakage')
        for patient in split['test']:
            if patient in expected:raise ValueError('Patient outer-evaluated more than once')
            expected[patient]=i
    for i,split in enumerate(folds):
        manifests=sorted((root/f'fold_{i:02d}'/'outer_predictions').glob('*/manifest.json'))
        if not manifests:raise ValueError(f'No completed predictions for fold {i}')
        for manifest in manifests:
            meta=json.loads(manifest.read_text());patient=meta['patient_id']
            if not meta.get('complete') or meta.get('outer_fold')!=i or expected.get(patient)!=i:raise ValueError('Prediction role/fold mismatch')
            if meta['sample_id'] in samples:raise ValueError('Duplicate sample predictions')
            samples.add(meta['sample_id'])
            if panel is not None and meta['panel']!=panel:raise ValueError('Marker panel mismatch')
            panel=meta['panel'];ys,ps=by_patient.setdefault(patient,([],[]))
            for name in meta['shards']:
                with np.load(manifest.parent/name,allow_pickle=False) as z:
                    if z['target'].shape!=z['prediction'].shape or z['target'].shape[1]!=len(panel):raise ValueError('Invalid prediction shape')
                    ys.append(z['target'].copy());ps.append(z['prediction'].copy())
    if set(by_patient)!=set(expected):raise ValueError('Some outer patients lack predictions')
    # Check sample-level completeness, not merely presence of one sample per patient.
    inventory=json.loads((root/'inventory.json').read_text())
    if samples!={r['sample_id'] for r in inventory}:raise ValueError('Prediction samples differ from CV inventory')
    report={marker:{'patients':{}} for marker in panel}
    for patient,(ys,ps) in by_patient.items():
        y,pred=np.concatenate(ys),np.concatenate(ps)
        result=metrics(y,pred,np.full(len(y),patient),panel)
        for marker in panel:report[marker]['patients'][patient]=result[marker]['patients'][patient]
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    save_primary(out,report,'out_of_fold',a.bootstrap_replicates,a.bootstrap_seed)
    print(f'Primary OOF evaluation: {len(by_patient)} patients, {len(samples)} samples, {out}')

if __name__=='__main__':main()
