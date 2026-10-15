"""Evaluate saved regression heads and compare sparse measured/predicted marker maps."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from data import read_meta,partitions,scale
from metrics import metrics
from audit import snapshot,event,write_json


def check_role(patient,split,role):
    if role not in ('val','test') or patient not in split[role]:
        raise ValueError(f'Patient {patient} is not in checkpoint {role} partition')


def sparse_grid(xy,values,shape,stride):
    """One visible square per sampled location; unsampled cells stay masked."""
    xy=np.asarray(xy)
    if stride<8 or xy.ndim!=2 or xy.shape[1]!=2 or len(xy)!=len(values):
        raise ValueError('Invalid sparse map inputs')
    h,w=shape
    ix,iy=xy[:,0]//stride,xy[:,1]//stride
    if (xy<0).any() or (xy[:,0]>=w).any() or (xy[:,1]>=h).any():
        raise ValueError('Coordinates outside image')
    cells=np.column_stack((ix,iy))
    if len(np.unique(cells,axis=0))!=len(xy):
        raise ValueError('Multiple targets in one display cell; use correct sampling stride')
    grid=np.full(((h+stride-1)//stride,(w+stride-1)//stride),np.nan)
    grid[iy,ix]=values
    return np.ma.masked_invalid(grid)


def render_sample(out,meta,xy,y,pred,markers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stride=(meta.get('selection_policy') or {}).get('stride',8)
    cards=[]; display=[]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e5e7eb')
    for index,marker in enumerate(markers):
        j=meta['panel'].index(marker)
        combined=np.concatenate((y[:,j],pred[:,j]))
        low,high=np.percentile(combined,[1,99])
        if high<=low: high=low+1
        a=sparse_grid(xy,y[:,j],meta['shape'],stride)
        b=sparse_grid(xy,pred[:,j],meta['shape'],stride)
        figure,axes=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
        extent=(0,meta['shape'][1],meta['shape'][0],0)
        for ax,grid,title in zip(axes[:2],(a,b),('Measured mIF','Predicted marker')):
            im=ax.imshow(grid,cmap=cmap,vmin=low,vmax=high,extent=extent,interpolation='nearest')
            ax.set_title(title);ax.set_xlabel('x (native pixels)');ax.set_ylabel('y (native pixels)')
        figure.colorbar(im,ax=list(axes[:2]),label='Exported intensity; shared display scale',shrink=0.8)
        error=np.abs(pred[:,j]-y[:,j])
        error_grid=sparse_grid(xy,error,meta['shape'],stride)
        error_high=max(float(np.percentile(error,99)),1e-6)
        error_cmap=plt.get_cmap('magma').copy();error_cmap.set_bad('#e5e7eb')
        error_image=axes[2].imshow(error_grid,cmap=error_cmap,vmin=0,vmax=error_high,extent=extent,interpolation='nearest')
        axes[2].set_title('Absolute prediction error');axes[2].set_xlabel('x (native pixels)');axes[2].set_ylabel('y (native pixels)')
        figure.colorbar(error_image,ax=axes[2],label='Absolute error',shrink=0.8)
        axes[3].scatter(y[:,j],pred[:,j],s=3,alpha=0.3,rasterized=True)
        bounds=[float(combined.min()),float(combined.max())]
        if bounds[0]==bounds[1]: bounds[1]+=1
        axes[3].plot(bounds,bounds,'--',color='black',linewidth=1)
        axes[3].set_xlabel('Measured intensity');axes[3].set_ylabel('Predicted intensity');axes[3].set_title('Matched locations; dashed identity line')
        figure.suptitle(f'{meta["sample_id"]} | {marker} | sampled locations, not dense maps')
        filename=f'marker_{index:02d}.png';figure.savefig(out/filename,dpi=160);plt.close(figure)
        cards.append(f'<h3>{html.escape(marker)}</h3><img style="max-width:100%" src="{filename}">')
        display.append({'marker':marker,'low':float(low),'high':float(high),'error_high':error_high,'stride':stride})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Marker prediction review</title><h1>'+html.escape(meta['sample_id'])+'</h1><p>Role: '+html.escape(meta.get('evaluation_role','unspecified'))+'; validation used for checkpoint selection is not independent testing.</p><p>Gray cells are unsampled. Each square represents one sampled central 8x8 target; it does not establish expression throughout the larger stride cell. Maps share intensity limits within each marker. Clipping affects display only. Scatter plots use all raw predictions.</p>'+''.join(cards))
    write_json(out/'display_settings.json',display)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--caches',nargs='+',required=True,help='All checkpoint split caches, including training for baseline calculation')
    p.add_argument('--role',choices=['val','test'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['CD10','VIM','NES','UMOD','DAPI'])
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--device',default='cpu')
    p.add_argument('--bootstrap-replicates',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=42)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    if a.bootstrap_replicates<1 or a.bootstrap_seed<0 or a.batch_size<1 or len(set(a.markers))!=len(a.markers): p.error('Invalid batch size or duplicate display markers')
    import torch
    from train import make_head,load_group,predict
    from validate import validate
    ckpt=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    groups=partitions(a.caches,ckpt['split'])
    if not groups[a.role]: p.error(f'Checkpoint has no {a.role} patients; cannot score an empty partition')
    for root in a.caches:
        validate(root);m=read_meta(root)
        if m['panel']!=ckpt['panel'] or any(m[k]!=v for k,v in ckpt['cache_signature'].items()):
            raise ValueError('Checkpoint/cache provenance differs')
        if m.get('diagnostic_only'): raise ValueError('Diagnostic data cannot be evaluated as a cohort')
    if set(a.markers)-set(ckpt['panel']): p.error('Requested markers absent from panel')
    if not a.no_plots:
        import matplotlib  # fail before inference/output if plotting dependency is missing
    out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    event(out,'evaluation_start',role=a.role,checkpoint=str(Path(a.checkpoint).resolve()),
          caveat='Used for early stopping; not independent testing' if a.role=='val' else 'Held-out checkpoint test partition')
    # Equal patient raw means; compute independently of optional log1p scaler.
    train_x,train_y,train_ids=load_group(groups['train'])
    baseline=np.mean([train_y[train_ids==patient].astype(np.float64).mean(0) for patient in np.unique(train_ids)],axis=0)
    del train_x,train_y,train_ids
    model=make_head(torch,len(ckpt['panel']),ckpt['hidden']).to(a.device)
    model.load_state_dict(ckpt['state_dict'])
    ys=[];ps=[];ids=[]
    for root in groups[a.role]:
        meta=read_meta(root);check_role(meta['patient_id'],ckpt['split'],a.role)
        sample=out/meta['sample_id'];sample.mkdir();xs=[];targets=[];coordinates=[]
        for name in meta['shards']:
            with np.load(root/name,allow_pickle=False) as z:
                x=z['features'].copy(); y=z['target'].copy();xy=z['xy'].copy()
            raw=scale(predict(torch,model,x,a.device,a.batch_size),ckpt['scaler'],inverse=True)
            if not np.isfinite(raw).all(): raise ValueError('Nonfinite prediction')
            np.savez(sample/name,xy=xy,target=y,prediction=raw)
            targets.append(y);xs.append(raw);coordinates.append(xy)
        y,pred,xy=np.concatenate(targets),np.concatenate(xs),np.concatenate(coordinates)
        ys.append(y);ps.append(pred);ids.extend([meta['patient_id']]*len(y))
        if not a.no_plots: render_sample(sample,{**meta,'evaluation_role':a.role},xy,y,pred,a.markers)
        write_json(sample/'manifest.json',{**meta,'role':a.role,'checkpoint':str(Path(a.checkpoint).resolve()),
            'arrays':['xy','target','prediction'],'complete':True})
        event(out,'sample_complete',sample_id=meta['sample_id'],rows=len(y))
    y,pred,patients=np.concatenate(ys),np.concatenate(ps),np.asarray(ids)
    report={'model':metrics(y,pred,patients,ckpt['panel']),
        'training_mean_baseline':metrics(y,np.tile(baseline,(len(y),1)),patients,ckpt['panel'])}
    write_json(out/'metrics.json',report)
    from primary_evaluation import save_primary
    save_primary(out,report['model'],a.role,a.bootstrap_replicates,a.bootstrap_seed)
    with (out/'marker_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['marker','patient_mean_model_mae','patient_mean_baseline_mae','mae_reduction_percent','patient_mean_pearson'])
        for marker in ckpt['panel']:
            model_mae=report['model'][marker]['patient_mean_mae'];base=report['training_mean_baseline'][marker]['patient_mean_mae']
            writer.writerow([marker,model_mae,base,100*(base-model_mae)/base if base else None,report['model'][marker]['patient_mean_pearson']])
    summary={'role':a.role,'best_epoch':ckpt['epoch'],'patients':len(np.unique(patients)),'rows':len(y),'markers':len(ckpt['panel']),
        'markers_lower_mae_than_baseline':sum(report['model'][m]['patient_mean_mae']<report['training_mean_baseline'][m]['patient_mean_mae'] for m in ckpt['panel']),
        'independent_testing':a.role=='test','uncertainty':'not estimated'}
    write_json(out/'summary.json',summary);event(out,'evaluation_complete',**summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
