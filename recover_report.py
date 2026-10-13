"""Regenerate validation metrics and scale diagnostics from a saved head; no retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import read_meta,partitions,scale
from metrics import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--caches',nargs='+',required=True)
    p.add_argument('--out',required=True,help='New report directory, preserving the original run')
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args()
    if a.batch_size<1:
        p.error('batch-size must be positive')
    import torch
    from train import make_head,load_group,predict
    from validate import validate
    ckpt=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    for root in a.caches:
        validate(root)
        meta=read_meta(root)
        if meta['panel']!=ckpt['panel'] or any(meta[k]!=v for k,v in ckpt['cache_signature'].items()):
            raise ValueError('Cache and checkpoint provenance differ')
    groups=partitions(a.caches,ckpt['split'])
    out=Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    _,y,patients=load_group(groups['train'])
    vx,vy,vp=load_group(groups['val'])
    model=make_head(torch,len(ckpt['panel']),ckpt['hidden']).to(a.device)
    model.load_state_dict(ckpt['state_dict'])
    standardized=predict(torch,model,vx,a.device,a.batch_size)
    raw=scale(standardized,ckpt['scaler'],inverse=True)
    baseline=np.mean([y[patients==patient].astype(np.float64).mean(0) for patient in np.unique(patients)],axis=0)
    report={'model':metrics(vy,raw,vp,ckpt['panel']),
            'training_mean_baseline':metrics(vy,np.tile(baseline,(len(vy),1)),vp,ckpt['panel'])}
    svy=scale(vy,ckpt['scaler'])
    error=np.abs(standardized.astype(np.float64)-svy)
    huber=np.where(error<=1,0.5*error**2,error-0.5)
    transformed_train=np.log1p(y) if ckpt['scaler']['log1p'] else y
    transformed_val=np.log1p(vy) if ckpt['scaler']['log1p'] else vy
    diagnostic=[]
    for j,marker in enumerate(ckpt['panel']):
        diagnostic.append({'marker':marker,'training_raw_min':float(y[:,j].min()),
            'training_raw_max':float(y[:,j].max()),'validation_raw_min':float(vy[:,j].min()),
            'validation_raw_max':float(vy[:,j].max()),
            'training_transformed_std':float(transformed_train[:,j].astype(np.float64).std()),
            'scaler_std':ckpt['scaler']['std'][j],
            'validation_transformed_mean':float(transformed_val[:,j].astype(np.float64).mean()),
            'training_scaler_mean':ckpt['scaler']['mean'][j],
            'validation_patient_mean_huber':float(np.mean([huber[vp==patient,j].mean() for patient in np.unique(vp)]))})
    diagnostic.sort(key=lambda item:item['validation_patient_mean_huber'],reverse=True)
    out.mkdir(parents=True)
    (out/'validation_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'scale_diagnostics.json').write_text(json.dumps(diagnostic,indent=2,allow_nan=False))
    (out/'recovery.json').write_text(json.dumps({'checkpoint':str(Path(a.checkpoint).resolve()),
        'best_epoch':ckpt['epoch'],'saved_val_huber':ckpt['val_huber'],'args':vars(a)},indent=2))
    print(json.dumps({'best_epoch':ckpt['epoch'],'reports':str(out),'largest_loss_markers':diagnostic[:5]},indent=2))

if __name__=='__main__':
    main()
