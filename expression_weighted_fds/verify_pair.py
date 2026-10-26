"""Refuse a claimed matched comparison when settings, thresholds or folds differ."""
import argparse,json
from pathlib import Path


def read(root,name):return json.loads((root/name).read_text())


def verify(baseline,candidate):
    roots=[Path(baseline),Path(candidate)]
    for root,mode in zip(roots,['weighted','weighted_fds']):
        m=read(root,'manifest.json')
        if not m.get('complete') or m.get('mode')!=mode:raise ValueError('Expected completed '+mode+' run: '+str(root))
    configs=[read(r,'config.json') for r in roots]
    for c in configs:c.pop('mode')
    if configs[0]!=configs[1]:raise ValueError('Architecture/FDS configuration differs beyond mode')
    folds=[read(r,'folds.json') for r in roots]
    if folds[0]!=folds[1]:raise ValueError('Patient folds differ')
    inventories=[{v['sample_id']:(v['patient_id'],v['rows']) for v in read(r,'inventory.json')} for r in roots]
    if inventories[0]!=inventories[1]:raise ValueError('Cache inventory differs')
    for i in range(len(folds[0])):
        key=Path(f'fold_{i:02d}')/'training'
        for filename in ['expression_state.json','target_scaler.json']:
            if read(roots[0],key/filename)!=read(roots[1],key/filename):
                raise ValueError(f'Fold {i}: unequal {filename}')
        args=[read(r,key/'invocation.json')['args'] for r in roots]
        for a in args:
            for k in ['mode','out','split']:a.pop(k,None)
        if args[0]!=args[1]:raise ValueError(f'Fold {i}: other training arguments differ')
    return {'matched':True,'folds':len(folds[0]),'samples':len(inventories[0]),
            'baseline':str(roots[0]),'candidate':str(roots[1]),'treatment':'training-time FDS only'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',required=True);p.add_argument('--candidate-root',required=True)
    a=p.parse_args();print(json.dumps(verify(a.baseline_root,a.candidate_root),indent=2))

if __name__=='__main__':main()
