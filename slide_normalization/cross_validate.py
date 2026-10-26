"""Reviewed patient folds; isolated trainers; CPU-only outer scoring; immutable outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train import add_training_arguments,check_arguments
from config import config_from_args
from data import read_meta,partitions,scale
from io_utils import load_group,predict,independent_output,resource_check
from audit import snapshot,event,write_json
from metrics import metrics
from expression import subgroup_metrics,summarize_subgroups
from primary_evaluation import save_primary


def reviewed_folds(path,roots):
    folds=json.loads(Path(path).read_text())
    if not isinstance(folds,list) or len(folds)<2:raise ValueError('Need at least two outer folds')
    patients={read_meta(root)['patient_id'] for root in roots}
    seen=[]
    for split in folds:
        partitions(roots,split)
        if not split['test']:raise ValueError('Empty outer test partition')
        seen.extend(split['test'])
    if set(seen)!=patients or len(seen)!=len(set(seen)):
        raise ValueError('Every patient must occur in exactly one outer test fold')
    ref=Path(path).with_name('inventory.json')
    if ref.exists():
        inventory=json.loads(ref.read_text())
        previous={r['sample_id']:r['patient_id'] for r in inventory}
        current={read_meta(r)['sample_id']:read_meta(r)['patient_id'] for r in roots}
        if len(previous)!=len(inventory) or previous!=current:
            raise ValueError('Sample-to-patient inventory differs from reviewed reference')
    return folds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',required=True);p.add_argument('--reference-folds',required=True)
    p.add_argument('--out',required=True);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--bootstrap-replicates',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=42)
    add_training_arguments(p);a=p.parse_args();c=config_from_args(a);check_arguments(a,c)
    if a.bootstrap_replicates<1 or a.bootstrap_seed<0:raise ValueError('Invalid bootstrap settings')
    roots=sorted(path.parent.resolve() for path in Path(a.cache_root).glob('*/manifest.json'))
    if not roots:raise ValueError('No complete cache manifests found')
    from validate import validate
    inventory=[]
    for root in roots:
        r=validate(root);m=read_meta(root)
        if m.get('diagnostic_only') or m.get('encoder')!='UNI2-h':raise ValueError('Need real UNI2-h caches')
        inventory.append({'path':str(root),'sample_id':m['sample_id'],'patient_id':m['patient_id'],'rows':r['rows']})
    folds=reviewed_folds(a.reference_folds,roots)
    raw_folders=[Path(m[key]).resolve().parent for m in (read_meta(r) for r in roots) for key in ('he','mif') if key in m]
    out=independent_output(a.out,[Path(a.cache_root),Path(a.reference_folds).parent,*raw_folders])
    preflight=resource_check(roots,out,a.max_numeric_gb)
    print(json.dumps({'preflight':preflight,'mode':c.mode,'folds':len(folds)}),flush=True)
    out.mkdir(parents=True);snapshot(out,vars(a))
    write_json(out/'inventory.json',inventory);write_json(out/'folds.json',folds)
    write_json(out/'config.json',c.to_dict());write_json(out/'preflight.json',preflight)
    event(out,'plan',folds=folds,mode=c.mode,parent_scoring_device='cpu',trainer_device=a.device)
    if a.plan_only:return
    from model import load_checkpoint
    patient_results={};expression_results={}
    panel=read_meta(roots[0])['panel']
    for i,split in enumerate(folds):
        fold=out/f'fold_{i:02d}';fold.mkdir();write_json(fold/'split.json',split)
        command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--caches',*[str(r) for r in roots],
                 '--split',str(fold/'split.json'),'--out',str(fold/'training')]
        forwarded={**c.to_dict(),**{key:getattr(a,key) for key in
            ('epochs','min_epochs','patience','batch_size','lr','device','max_numeric_gb','expression_quantile','max_expression_weight')}}
        forwarded.pop('input_dim');forwarded['seed']=a.seed+i
        if a.thresholds:forwarded['thresholds']=str(Path(a.thresholds).resolve())
        for key,value in forwarded.items():command.extend(['--'+key.replace('_','-'),str(value)])
        print(f'Fold {i}: train={len(split["train"])} val={len(split["val"])} test={len(split["test"])}',flush=True)
        event(out,'fold_start',fold=i,command=command)
        with (fold/'training.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            event(out,'failed',fold=i,returncode=result.returncode)
            raise RuntimeError(f'Fold {i} failed: {fold}/training.log')
        # This process NEVER initializes CUDA. Only the child trainer holds GPU.
        model,checkpoint=load_checkpoint(fold/'training/best.pt','cpu')
        groups=partitions(roots,split)
        predictions=fold/'outer_predictions';predictions.mkdir()
        for patient in split['test']:
            selected=[r for r in groups['test'] if read_meta(r)['patient_id']==patient]
            x,y,ids=load_group(selected)
            raw=scale(predict(model,x,'cpu',a.batch_size),checkpoint['scaler'],inverse=True)
            patient_results[patient]=metrics(y,raw,ids,panel)
            expression_results[patient]=subgroup_metrics(y,raw,ids,panel,checkpoint['expression_state'])
            offset=0
            for root in selected:
                m=read_meta(root);dest=predictions/m['sample_id'];dest.mkdir()
                for name in m['shards']:
                    with np.load(root/name,allow_pickle=False) as z:
                        xy=z['xy'].copy();target=z['target'].copy()
                    n=len(target)
                    np.savez(dest/name,xy=xy,target=target,prediction=raw[offset:offset+n]);offset+=n
                write_json(dest/'manifest.json',{**m,'complete':True,'outer_fold':i,
                    'prediction_units':'per-slide normalized pixel-clipped 8x8 mean (relative units)','model_format':'kidney_slide_p99_v1',
                    'checkpoint':str((fold/'training/best.pt').resolve()),'experiment_mode':c.mode})
            if offset!=len(raw):raise ValueError('Prediction offsets differ from cache rows')
            del x,y,ids,raw
        event(out,'fold_complete',fold=i,best_epoch=checkpoint['epoch'])
        write_json(out/'oof_patient_metrics.json',patient_results)
        write_json(out/'oof_expression_patients.json',expression_results)
        del model,checkpoint
    merged={m:{'patients':{pid:r[m]['patients'][pid] for pid,r in patient_results.items()}} for m in panel}
    save_primary(out,merged,'out_of_fold',a.bootstrap_replicates,a.bootstrap_seed)
    write_json(out/'oof_expression_summary.json',summarize_subgroups(expression_results,panel))
    write_json(out/'manifest.json',{'complete':True,'mode':c.mode,'panel':panel,'patients':len(patient_results),
                                  'folds':len(folds),'encoder_frozen':True})
    event(out,'complete',patients=len(patient_results));print(f'Completed: {out}',flush=True)

if __name__=='__main__':main()
