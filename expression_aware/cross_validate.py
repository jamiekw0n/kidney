"""Patient-level outer CV with separate inner holdout for early stopping; fixed head settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from data import read_meta,partitions,scale
from audit import snapshot,event,write_json
from metrics import metrics


def make_folds(patients,folds,seed,val_fraction):
    patients=sorted(set(patients))
    if folds<2 or folds>len(patients) or len(patients)<4 or not 0<val_fraction<1:
        raise ValueError('Need >=4 patients, 2..N folds, and validation fraction in (0,1)')
    rng=np.random.default_rng(seed); order=rng.permutation(patients)
    result=[]
    for i,test in enumerate(np.array_split(order,folds)):
        remaining=np.array([p for p in order if p not in test])
        if len(remaining)<2: raise ValueError('Not enough inner training/validation patients')
        nval=min(len(remaining)-1,max(1,int(np.ceil(len(remaining)*val_fraction))))
        shuffled=np.random.default_rng(seed+i+1).permutation(remaining)
        result.append({'train':sorted(shuffled[nval:].tolist()),'val':sorted(shuffled[:nval].tolist()),'test':sorted(test.tolist())})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-folds',required=True,help='Existing reviewed folds.json; reused exactly')
    p.add_argument('--thresholds')
    p.add_argument('--expression-quantile',type=float,default=0.9)
    p.add_argument('--max-expression-weight',type=float,default=5.0)
    p.add_argument('--unweighted',action='store_true')
    p.add_argument('--cache-root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--folds',type=int,default=5); p.add_argument('--val-fraction',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--hidden',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--intensity-max',type=float,help='Fixed target range scaling; forwarded to train.py')
    p.add_argument('--log1p',action='store_true'); p.add_argument('--device',default='cuda')
    p.add_argument('--bootstrap-replicates',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=42)
    p.add_argument('--plan-only',action='store_true',help='Write audited folds without training; rerun to a NEW directory for execution')
    p.add_argument('--max-numeric-gb',type=float,default=8,help='Refuse estimated numeric peak above this limit; excludes framework/model overhead')
    a=p.parse_args()
    if a.bootstrap_replicates<1 or a.bootstrap_seed<0 or min(a.epochs,a.patience,a.batch_size)<1 or a.hidden<0 or a.lr<=0 or a.max_numeric_gb<=0 or a.seed<0:
        p.error('Invalid training parameters')
    from validate import validate
    roots=sorted(path.parent for path in Path(a.cache_root).glob('*/manifest.json'))
    if not roots: p.error('No complete caches found')
    inventory=[]; numeric=0; meta=None
    for root in roots:
        report=validate(root); m=read_meta(root)
        if m.get('diagnostic_only'): p.error('Diagnostic-only caches are not cohort evaluation data')
        if m.get('encoder')!='UNI2-h': p.error('Expected UNI2-h feature caches')
        if meta is None: meta=m
        n=report['rows']; numeric+=n*(1536+len(m['panel']))*4
        inventory.append({'path':str(root.resolve()),'sample_id':m['sample_id'],'patient_id':m['patient_id'],'rows':n})
    if numeric*3>a.max_numeric_gb*1e9:
        p.error(f'Numeric memory estimate {numeric*3/1e9:.2f} GB exceeds limit; reduce selection cap or use an out-of-core loader')
    folds=json.loads(Path(a.reference_folds).read_text())
    if len(folds)!=a.folds: raise ValueError('Reference fold count mismatch')
    patients={r['patient_id'] for r in inventory}
    test_ids=[patient for split in folds for patient in split['test']]
    if len(test_ids)!=len(set(test_ids)) or set(test_ids)!=patients:
        raise ValueError('Every patient must occur in exactly one outer test fold')
    for split in folds: partitions(roots,split)
    out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); snapshot(out,vars(a))
    write_json(out/'inventory.json',inventory); write_json(out/'folds.json',folds)
    for reportfile in ('extraction_report.json','preparation_cohort_report.json'):
        source=Path(a.cache_root)/reportfile
        if source.exists(): write_json(out/reportfile,json.loads(source.read_text()))
    event(out,'fold_plan',patients=len(set(r['patient_id'] for r in inventory)),folds=len(folds),numeric_peak_estimate=numeric*3,
          parent_evaluation_device='cpu',training_device=a.device,protocol='Outer patient evaluation; inner patient holdout for early stopping. Hyperparameters fixed before outer evaluation.')
    if a.plan_only:
        print(f'Fold plan saved to {out}; no training performed.'); return
    # Parent never acquires a CUDA context: only each isolated trainer uses GPU.
    # CPU scoring is cheap for cached features and avoids exclusive-process conflicts.
    import torch
    from train import make_head,load_group,predict
    patient_results={}; patient_baselines={}
    expression_results={}
    from expression import subgroup_metrics, summarize_subgroups
    for i,split in enumerate(folds):
        fold=out/f'fold_{i:02d}'; fold.mkdir(); write_json(fold/'split.json',split)
        command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--caches',*[str(r) for r in roots],
            '--split',str(fold/'split.json'),'--out',str(fold/'training'),'--hidden',str(a.hidden),
            '--epochs',str(a.epochs),'--patience',str(a.patience),'--batch-size',str(a.batch_size),
            '--lr',str(a.lr),'--seed',str(a.seed+i),'--device',a.device]
        command.extend(['--expression-quantile',str(a.expression_quantile),'--max-expression-weight',str(a.max_expression_weight)])
        if a.thresholds: command.extend(['--thresholds',str(Path(a.thresholds).resolve())])
        if a.unweighted: command.append('--unweighted')
        if a.log1p: command.append('--log1p')
        if a.intensity_max is not None: command.extend(['--intensity-max',str(a.intensity_max)])
        event(out,'fold_training_start',fold=i,command=command,split=split)
        with (fold/'training.log').open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            event(out,'failed',fold=i,returncode=code); raise RuntimeError(f'Fold {i} failed; inspect {fold}/training.log')
        checkpoint=torch.load(fold/'training'/'best.pt',map_location='cpu',weights_only=True)
        model=make_head(torch,len(meta['panel']),checkpoint['hidden']).to('cpu')
        model.load_state_dict(checkpoint['state_dict'])
        groups=partitions(roots,split)
        # Combine every sample belonging to the same test patient before scoring.
        _,train_y,train_patients=load_group(groups['train'])
        baseline=np.mean([train_y[train_patients==patient].astype(np.float64).mean(0) for patient in np.unique(train_patients)],axis=0)
        del _,train_y,train_patients
        predictions=fold/'outer_predictions'; predictions.mkdir()
        for patient in split['test']:
            patient_roots=[root for root in groups['test'] if read_meta(root)['patient_id']==patient]
            x,y,patient_ids=load_group(patient_roots)
            raw=scale(predict(torch,model,x,'cpu',a.batch_size),checkpoint['scaler'],inverse=True)
            expression_results[patient]=subgroup_metrics(y,raw,patient_ids,meta['panel'],checkpoint['expression_state'])
            evaluated=metrics(y,raw,patient_ids,meta['panel'])
            evaluated_baseline=metrics(y,np.tile(baseline,(len(y),1)),patient_ids,meta['panel'])
            patient_results[patient]=evaluated; patient_baselines[patient]=evaluated_baseline
            offset=0
            for root in patient_roots:
                sample=predictions/root.name; sample.mkdir()
                m=read_meta(root)
                for name in m['shards']:
                    with np.load(root/name,allow_pickle=False) as z:
                        n=len(z['xy']); xy=z['xy'].copy(); target=z['target'].copy()
                    np.savez(sample/name,xy=xy,target=target,prediction=raw[offset:offset+n]); offset+=n
                write_json(sample/'manifest.json',{**m,'complete':True,'outer_fold':i,'prediction_units':'original measured intensity',
                    'checkpoint':str((fold/'training'/'best.pt').resolve())})
            write_json(fold/(patient+'_metrics.json'),{'model':evaluated,'training_mean_baseline':evaluated_baseline})
        event(out,'fold_complete',fold=i,best_epoch=checkpoint['epoch'],test_patients=split['test'])
        del model,checkpoint,x,y,raw
        write_json(out/'oof_patient_metrics.json',{'model':patient_results,'training_mean_baseline':patient_baselines})
    write_json(out/'oof_expression_patients.json',expression_results)
    write_json(out/'oof_expression_summary.json',summarize_subgroups(expression_results,meta['panel']))
    summary={}
    for marker in meta['panel']:
        summary[marker]={}
        for name,records in [('model',patient_results),('training_mean_baseline',patient_baselines)]:
            summary[marker][name]={}
            for metric in ('mae','rmse','pearson'):
                values=[r[marker]['patient_mean_'+metric] for r in records.values()]
                finite=[v for v in values if v is not None]
                summary[marker][name][metric]={'patient_mean':float(np.mean(finite)) if finite else None,
                    'patient_std':float(np.std(finite,ddof=1)) if len(finite)>1 else None,'defined_patients':len(finite)}
    from primary_evaluation import save_primary
    merged={marker:{'patients':{patient:record[marker]['patients'][patient] for patient,record in patient_results.items()}} for marker in meta['panel']}
    save_primary(out,merged,'out_of_fold',a.bootstrap_replicates,a.bootstrap_seed)
    write_json(out/'oof_summary.json',summary); event(out,'complete',evaluated_patients=len(patient_results))
    print(f'Cross-validation complete: {out}/oof_summary.json')

if __name__=='__main__': main()
