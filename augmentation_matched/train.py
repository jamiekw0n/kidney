"""Train a cached-feature expression-weighted MLP with optional FDS; no encoder or raw image is loaded."""
import argparse
import json
import random
import hashlib
from pathlib import Path
import numpy as np
from config import add_config_arguments,config_from_args
from data import partitions,read_meta,fit_scaler,scale
from io_utils import load_group,predict,independent_output,resource_check,SIGNATURE
from audit import snapshot,event,write_json
from metrics import metrics
from expression import fit_expression,element_weights,subgroup_metrics


def add_training_arguments(p):
    add_config_arguments(p)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--min-epochs',type=int,default=51,help='Checkpoint selection and patience begin at this 1-based epoch')
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-3,help='Same constant learning rate as earlier expression-aware MLP')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-numeric-gb',type=float,default=8)
    p.add_argument('--expression-quantile',type=float,default=.9)
    p.add_argument('--max-expression-weight',type=float,default=5.)
    p.add_argument('--thresholds',help='Optional assay-defined raw thresholds for both weighting and subgroup evaluation')


def check_arguments(a,c):
    if c.intensity_max!=1.:raise ValueError('Normalized targets require intensity_max=1 for binning')
    if min(a.epochs,a.min_epochs,a.patience,a.batch_size)<1 or a.min_epochs>a.epochs:
        raise ValueError('Invalid training epoch/batch settings')
    if not np.isfinite(a.lr) or a.lr<=0 or a.seed<0:
        raise ValueError('Invalid optimizer/seed settings')
    if a.min_epochs<=c.fds_off:
        raise ValueError('Selection must start after FDS switches off: min_epochs > fds_off')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--caches',nargs='+',required=True);p.add_argument('--split',required=True)
    p.add_argument('--out',required=True);add_training_arguments(p)
    p.add_argument('--augmentation-plan',required=True);p.add_argument('--variant',choices=['control','geometry','geometry_color'],required=True)
    p.set_defaults(mode='weighted')
    a=p.parse_args();c=config_from_args(a);check_arguments(a,c)
    if c.mode!='weighted':raise ValueError('FDS is fixed OFF in this augmentation-only comparison')
    split=json.loads(Path(a.split).read_text())
    plan_cfg=json.loads((Path(a.augmentation_plan)/'plan.json').read_text())
    if split not in plan_cfg['folds']:raise ValueError('Split differs from augmentation plan')
    fold_index=plan_cfg['folds'].index(split)
    if a.seed!=plan_cfg['seed']+fold_index:raise ValueError('Seed differs from matched fold plan')
    for key,value in plan_cfg['training'].items():
        if getattr(a,key)!=value:raise ValueError('Training argument differs from matched plan: '+key)
    roots=[Path(r).resolve() for r in a.caches]
    groups=partitions(roots,split)
    from validate import validate
    for root in roots:
        validate(root)
        m=read_meta(root)
        if m.get('target_units')!='slide_channel_percentile_pixel_clipped_0_1' or not m.get('normalization_recipe'):raise ValueError('This trainer requires newly normalized caches')
        if m.get('diagnostic_only') or m.get('encoder')!='UNI2-h':raise ValueError('Need real UNI2-h cohort caches')
    meta=read_meta(groups['train'][0])
    raw_folders=[Path(m[key]).resolve().parent for m in (read_meta(r) for r in roots) for key in ('he','mif') if key in m]
    out=independent_output(a.out,roots+raw_folders)
    preflight=resource_check(groups['train']+groups['val'],out,a.max_numeric_gb)
    import torch
    from model import Head
    from loss import bin_targets,weighted_huber
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; request GPU or explicitly choose CPU')
        torch.cuda.manual_seed_all(a.seed)
        preflight['gpu']={'name':torch.cuda.get_device_name(a.device),
                          'free_bytes':torch.cuda.mem_get_info(a.device)[0]}
    print(json.dumps({'preflight':preflight}),flush=True)
    x,y,patients=load_group(groups['train']);vx,vy,vp=load_group(groups['val'])
    scaler=fit_scaler(y,patients)  # Training-only z score of the ALREADY PER-SLIDE NORMALIZED block targets.
    sy,svy=scale(y,scaler),scale(vy,scaler)
    expression=fit_expression(y,patients,meta['panel'],a.thresholds,a.expression_quantile,a.max_expression_weight)
    ew=element_weights(y,expression)
    ny=bin_targets(y,c.intensity_max)
    bin_targets(vy,c.intensity_max)  # Validate declared range; never fit to val targets.
    model=Head(len(meta['panel']),c).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    from feature_views import FeatureViews
    view_store=FeatureViews(a.augmentation_plan,groups['train'])
    if view_store.rows!=len(x):raise ValueError('Augmentation row alignment differs')
    dataset=torch.utils.data.TensorDataset(torch.arange(len(x)))
    generator=torch.Generator().manual_seed(a.seed)
    counts={pid:int((patients==pid).sum()) for pid in np.unique(patients)}
    weights=torch.tensor([1/counts[pid] for pid in patients],dtype=torch.double)
    sampler=torch.utils.data.WeightedRandomSampler(weights,len(x),replacement=True,generator=generator)
    loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,sampler=sampler)
    out.mkdir(parents=True);snapshot(out,vars(a))
    write_json(out/'config.json',c.to_dict());write_json(out/'target_scaler.json',scaler)
    write_json(out/'expression_state.json',expression);write_json(out/'preflight.json',preflight)
    # Actual unique training rows/patients supporting bins, unlike repeated sampler draws.
    occupancy=[]
    bin_ids=np.minimum((ny*c.bins).astype(int),c.bins-1)
    for j,marker in enumerate(meta['panel']):
        occupancy.append({'marker':marker,'rows_per_bin':np.bincount(bin_ids[:,j],minlength=c.bins).tolist(),
            'patients_per_bin':[int(sum(np.any(bin_ids[patients==pid,j]==b) for pid in counts)) for b in range(c.bins)]})
    write_json(out/'training_bin_support.json',occupancy)
    event(out,'start',config=c.to_dict(),split=split,train_rows=len(x),val_rows=len(vx),
          train_patient_rows=counts,optimizer='AdamW',weight_decay=1e-4,sampling='equal expected patient draws',torch_version=str(torch.__version__),
          selection='equal-patient unweighted standardized Huber, after minimum epoch',encoder_frozen=True)
    init_hash=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        init_hash.update(key.encode());init_hash.update(value.detach().cpu().numpy().tobytes())
    write_json(out/'initial_state.json',{'sha256':init_hash.hexdigest()})
    history=[];best=float('inf');stale=0
    for epoch in range(a.epochs):
        model.train()
        collect=c.use_fds and epoch<c.fds_off
        moments=model.fds.new_moments() if collect else None
        total=0.;n=0;lr=optimizer.param_groups[0]['lr']
        view_rng=np.random.default_rng(a.seed+100000+epoch);draw_hash=hashlib.sha256();view_hash=hashlib.sha256();augmented=0
        for (row_ids,) in loader:
            ix=row_ids.numpy();draw_hash.update(ix.tobytes())
            chosen=view_rng.random(len(ix))<.5
            if a.variant=='control':chosen[:]=False
            view_hash.update(chosen.tobytes())
            augmented+=int(chosen.sum())
            bx=view_store.replace(x[ix].copy(),ix,chosen,a.variant)
            bx,by,bw,bn=[torch.from_numpy(v).to(a.device) for v in (bx,sy[ix],ew[ix],ny[ix])]
            optimizer.zero_grad(set_to_none=True)
            pred,h=model(bx,bn,epoch,return_features=True)
            loss=weighted_huber(pred,by,bw)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite gradient')
            optimizer.step()
            if collect:model.fds.accumulate(moments,h,bn)
            total+=float(loss.detach())*len(bx);n+=len(bx)
        if collect:model.fds.finish_epoch(moments,epoch)
        valpred=predict(model,vx,a.device,a.batch_size)
        if not np.isfinite(valpred).all():raise ValueError('Nonfinite validation predictions')
        err=np.abs(valpred.astype(np.float64)-svy)
        losses=np.where(err<=1,.5*err**2,err-.5)
        val=float(np.mean([losses[vp==pid].mean() for pid in np.unique(vp)]))
        if not np.isfinite(val):raise ValueError('Nonfinite validation Huber')
        record={'epoch':epoch+1,'zero_based_epoch':epoch,'train_objective':total/n,
                'train_objective_type':'expression_weighted_standardized_Huber_delta1',
                'val_huber':val,'lr':lr,'fds_applied':c.use_fds and c.fds_start<=epoch<c.fds_off,
                'checkpoint_eligible':epoch+1>=a.min_epochs}
        record.update(sampler_draw_sha256=draw_hash.hexdigest(),view_draw_sha256=view_hash.hexdigest(),augmented_draws=augmented,identity_draws=n-augmented,variant=a.variant,steps=len(loader))
        record['fds']=model.fds.diagnostics()
        history.append(record);write_json(out/'history.json',history)
        event(out,'epoch',**record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('adaptive_loss','fds')}),flush=True)
        if epoch+1<a.min_epochs:continue
        if val<best:
            best=val;stale=0
            torch.save({'format':'kidney_aug_views_v1','variant':a.variant,'augmentation_plan':str(Path(a.augmentation_plan).resolve()),'augmentation_plan_fingerprint':view_store.fingerprint,'state_dict':model.state_dict(),
                'config':c.to_dict(),'panel':meta['panel'],'scaler':scaler,'expression_state':expression,
                'split':split,'epoch':epoch+1,'val_huber':val,'args':vars(a),
                'selection_policy':meta.get('selection_policy'),
                'cache_signature':{k:meta.get(k) for k in SIGNATURE},
                'optimizer_state':optimizer.state_dict(),
                'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':generator.get_state()},out/'best.pt')
        else:
            stale+=1
            if stale>=a.patience:break
    from model import load_checkpoint
    model,checkpoint=load_checkpoint(out/'best.pt',a.device)
    raw=scale(predict(model,vx,a.device,a.batch_size),scaler,inverse=True)
    write_json(out/'validation_metrics.json',{'model':metrics(vy,raw,vp,meta['panel'])})
    write_json(out/'validation_expression_metrics.json',subgroup_metrics(vy,raw,vp,meta['panel'],expression))
    write_json(out/'run.json',{'complete':True,'best_epoch':checkpoint['epoch'],'best_val_huber':checkpoint['val_huber'],
                             'mode':c.mode,'variant':a.variant,'train_rows':len(x),'val_rows':len(vx),'test_arrays_used_for_fit':False})
    event(out,'complete',best_epoch=checkpoint['epoch'],best_val_huber=checkpoint['val_huber'])

if __name__=='__main__':main()
