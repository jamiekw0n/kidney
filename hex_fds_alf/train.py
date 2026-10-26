"""Train a cached-feature HEX-inspired head; no encoder or raw image is loaded."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from config import add_config_arguments,config_from_args
from data import partitions,read_meta,fit_scaler,scale
from io_utils import load_group,predict,independent_output,resource_check,SIGNATURE
from audit import snapshot,event,write_json
from metrics import metrics
from expression import fit_expression,subgroup_metrics


def add_training_arguments(p):
    add_config_arguments(p)
    p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--min-epochs',type=int,default=51,help='Checkpoint selection and patience begin at this 1-based epoch')
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-3,help='Cached-head adaptation; HEX end-to-end script uses 1e-5')
    p.add_argument('--lr-gamma',type=float,default=.95)
    p.add_argument('--sampling',choices=['patient','uniform'],default='patient')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-numeric-gb',type=float,default=8)
    p.add_argument('--thresholds',help='Optional assay-defined raw thresholds for evaluation ONLY')


def check_arguments(a,c):
    if min(a.epochs,a.min_epochs,a.patience,a.batch_size)<1 or a.min_epochs>a.epochs:
        raise ValueError('Invalid training epoch/batch settings')
    if not np.isfinite(a.lr) or a.lr<=0 or not 0<a.lr_gamma<=1 or a.seed<0:
        raise ValueError('Invalid optimizer/seed settings')
    if c.use_fds and a.min_epochs<=c.fds_off:
        raise ValueError('Selection must start after FDS switches off: min_epochs > fds_off')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--caches',nargs='+',required=True);p.add_argument('--split',required=True)
    p.add_argument('--out',required=True);add_training_arguments(p)
    a=p.parse_args();c=config_from_args(a);check_arguments(a,c)
    split=json.loads(Path(a.split).read_text())
    roots=[Path(r).resolve() for r in a.caches]
    groups=partitions(roots,split)
    from validate import validate
    for root in roots:
        validate(root)
        m=read_meta(root)
        if m.get('diagnostic_only') or m.get('encoder')!='UNI2-h':raise ValueError('Need real UNI2-h cohort caches')
    meta=read_meta(groups['train'][0])
    out=independent_output(a.out,roots)
    preflight=resource_check(groups['train']+groups['val'],out,a.max_numeric_gb)
    import torch
    from model import Head
    from adaptive_loss import AdaptiveLoss
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; request GPU or explicitly choose CPU')
        torch.cuda.manual_seed_all(a.seed)
        preflight['gpu']={'name':torch.cuda.get_device_name(a.device),
                          'free_bytes':torch.cuda.mem_get_info(a.device)[0]}
    print(json.dumps({'preflight':preflight}),flush=True)
    x,y,patients=load_group(groups['train']);vx,vy,vp=load_group(groups['val'])
    scaler=fit_scaler(y,patients,intensity_max=c.intensity_max)
    sy,svy=scale(y,scaler),scale(vy,scaler)
    # These thresholds label evaluation strata only; they do NOT enter loss or FDS.
    expression=fit_expression(y,patients,meta['panel'],a.thresholds)
    model=Head(len(meta['panel']),c).to(a.device)
    criterion=AdaptiveLoss(len(meta['panel']),c).to(a.device) if c.use_alf else None
    params=[{'params':model.parameters(),'name':'head'}]
    if criterion is not None:params.append({'params':criterion.parameters(),'name':'adaptive_loss'})
    optimizer=torch.optim.Adam(params,lr=a.lr,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=a.lr_gamma)
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(sy))
    generator=torch.Generator().manual_seed(a.seed)
    counts={pid:int((patients==pid).sum()) for pid in np.unique(patients)}
    if a.sampling=='patient':
        weights=torch.tensor([1/counts[pid] for pid in patients],dtype=torch.double)
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(x),replacement=True,generator=generator)
        loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,sampler=sampler)
    else:
        loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,shuffle=True,generator=generator)
    out.mkdir(parents=True);snapshot(out,vars(a))
    write_json(out/'config.json',c.to_dict());write_json(out/'target_scaler.json',scaler)
    write_json(out/'expression_state.json',expression);write_json(out/'preflight.json',preflight)
    # Actual unique training rows/patients supporting bins, unlike repeated sampler draws.
    occupancy=[]
    bin_ids=np.minimum((sy*c.bins).astype(int),c.bins-1)
    for j,marker in enumerate(meta['panel']):
        occupancy.append({'marker':marker,'rows_per_bin':np.bincount(bin_ids[:,j],minlength=c.bins).tolist(),
            'patients_per_bin':[int(sum(np.any(bin_ids[patients==pid,j]==b) for pid in counts)) for b in range(c.bins)]})
    write_json(out/'training_bin_support.json',occupancy)
    event(out,'start',config=c.to_dict(),split=split,train_rows=len(x),val_rows=len(vx),
          train_patient_rows=counts,optimizer='Adam',sampling=a.sampling,torch_version=str(torch.__version__),
          selection='equal-patient fixed-range MSE, after minimum epoch',encoder_frozen=True)
    history=[];best=float('inf');stale=0
    for epoch in range(a.epochs):
        model.train()
        collect=c.use_fds and epoch<c.fds_off
        moments=model.fds.new_moments() if collect else None
        total=0.;n=0;lr=optimizer.param_groups[0]['lr']
        for bx,by in loader:
            bx,by=bx.to(a.device),by.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            pred,h=model(bx,by,epoch,return_features=True)
            loss=criterion(pred-by).mean() if criterion is not None else torch.nn.functional.huber_loss(pred,by,delta=1.)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite gradient')
            optimizer.step()
            if collect:model.fds.accumulate(moments,h,by)
            total+=float(loss.detach())*len(bx);n+=len(bx)
        if collect:model.fds.finish_epoch(moments,epoch)
        valpred=predict(model,vx,a.device,a.batch_size)
        if not np.isfinite(valpred).all():raise ValueError('Nonfinite validation predictions')
        sq=(valpred.astype(np.float64)-svy)**2
        val=float(np.mean([sq[vp==pid].mean() for pid in np.unique(vp)]))
        if not np.isfinite(val):raise ValueError('Nonfinite validation MSE')
        record={'epoch':epoch+1,'zero_based_epoch':epoch,'train_objective':total/n,
                'train_objective_type':'adaptive_NLL' if c.use_alf else 'Huber_delta1',
                'val_mse':val,'lr':lr,'fds_applied':c.use_fds and c.fds_start<=epoch<c.fds_off,
                'checkpoint_eligible':epoch+1>=a.min_epochs}
        if criterion is not None:record['adaptive_loss']=criterion.diagnostics()
        record['fds']=model.fds.diagnostics()
        history.append(record);write_json(out/'history.json',history)
        event(out,'epoch',**record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('adaptive_loss','fds')}),flush=True)
        scheduler.step()
        if epoch+1<a.min_epochs:continue
        if val<best:
            best=val;stale=0
            torch.save({'format':'kidney_hex_fds_alf_v1','state_dict':model.state_dict(),
                'config':c.to_dict(),'panel':meta['panel'],'scaler':scaler,'expression_state':expression,
                'split':split,'epoch':epoch+1,'val_mse':val,'args':vars(a),
                'selection_policy':meta.get('selection_policy'),
                'cache_signature':{k:meta.get(k) for k in SIGNATURE},
                'criterion_state':criterion.state_dict() if criterion is not None else None,
                'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),
                'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':generator.get_state()},out/'best.pt')
        else:
            stale+=1
            if stale>=a.patience:break
    from model import load_checkpoint
    model,checkpoint=load_checkpoint(out/'best.pt',a.device)
    raw=scale(predict(model,vx,a.device,a.batch_size),scaler,inverse=True)
    write_json(out/'validation_metrics.json',{'model':metrics(vy,raw,vp,meta['panel'])})
    write_json(out/'validation_expression_metrics.json',subgroup_metrics(vy,raw,vp,meta['panel'],expression))
    write_json(out/'run.json',{'complete':True,'best_epoch':checkpoint['epoch'],'best_val_mse':checkpoint['val_mse'],
                             'mode':c.mode,'train_rows':len(x),'val_rows':len(vx),'test_arrays_used_for_fit':False})
    event(out,'complete',best_epoch=checkpoint['epoch'],best_val_mse=checkpoint['val_mse'])

if __name__=='__main__':main()
