"""Immutable augmentation plan, two-view extraction, original-view OOF evaluation and matched comparison."""
import argparse,csv,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from audit import snapshot,write_json
from data import read_meta,partitions,scale
from io_utils import independent_output,load_group,predict,SIGNATURE
from encoder_runtime import sha,load_encoder
from augment import RECIPE,parameters,views
from validate import validate

VARIANTS=('control','geometry','geometry_color')

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def software_versions():
    result={}
    for name in ('torch','torchvision','timm','numpy','Pillow'):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]='unavailable'
    return result

def config(plan):
    c=json.loads((Path(plan)/'plan.json').read_text());expected=c.pop('fingerprint')
    if fingerprint(c)!=expected:raise ValueError('Plan fingerprint mismatch')
    c['fingerprint']=expected;return c

def check_original(task):
    root=Path(task['cache']);meta=read_meta(root)
    if sha(root/'manifest.json')!=task['manifest_sha256']:raise ValueError('Original cache manifest changed')
    for r in task['shards']:
        if sha(root/r['name'])!=r['source_sha256']:raise ValueError('Original feature shard changed')
    he=Path(task['he_array']);stat=he.stat()
    if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=task['he_stat']:raise ValueError('Derived H&E array changed')
    return meta

def plan(a):
    if not 1<=a.min_epochs<=a.epochs or a.min_epochs<=50 or a.patience<1:raise ValueError('Need epochs>=min_epochs>50, positive patience')
    base=Path(a.source_run).resolve();old=json.loads((base/'experiment.json').read_text());folds=json.loads((base/'folds.json').read_text())
    roots=[base/'cache'/t['sample_id'] for t in old['tasks']]
    for f in folds:partitions(roots,f)
    patients={read_meta(r)['patient_id'] for r in roots};test=[pid for f in folds for pid in f['test']]
    if set(test)!=patients or len(test)!=len(set(test)):raise ValueError('Each patient must be outer test exactly once')
    tasks=[];raw=[]
    for i,root in enumerate(roots):
        report=validate(root);m=read_meta(root)
        if not m['sample_id'] or Path(m['sample_id']).name!=m['sample_id'] or m['sample_id'] in ('.','..'):raise ValueError('Unsafe sample ID')
        if m['sample_id']!=old['tasks'][i]['sample_id'] or m['patient_id']!=old['tasks'][i]['patient_id']:raise ValueError('Source experiment inventory differs from cache')
        if m.get('target_units')!='slide_channel_percentile_pixel_clipped_0_1':raise ValueError('Expected normalized dense-training features')
        if m.get('encoder')!='UNI2-h' or m['input_patch']!=128 or m['target_block']!=8:raise ValueError('Expected128-context8-target UNI2-h')
        if m['transform']!='PIL bilinear square resize 224; ToTensor; ImageNet normalization':raise ValueError('Unsupported identity transform')
        he=base/'prepared'/m['sample_id']/'he.npy';h=np.load(he,mmap_mode='r',allow_pickle=False)
        if list(h.shape)!=[*m['shape'],3] or h.dtype!=np.uint8:raise ValueError('Invalid derived H&E array')
        stat=he.stat()
        if m.get('he_array_stat')!={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}:raise ValueError('Derived H&E differs from original embedding provenance')
        shards=[]
        for name in m['shards']:
            with np.load(root/name,allow_pickle=False) as z:
                xy=z['xy'];n=len(xy)
                if (xy[:,0]<64).any() or (xy[:,1]<64).any() or (xy[:,0]+64>m['shape'][1]).any() or (xy[:,1]+64>m['shape'][0]).any():raise ValueError('Context outside H&E')
                xy_hash=hashlib.sha256(xy.tobytes()).hexdigest()
            shards.append({'name':name,'rows':n,'source_sha256':sha(root/name),'xy_sha256':xy_hash})
        tasks.append({'task_id':i,'sample_id':m['sample_id'],'patient_id':m['patient_id'],'cache':str(root.resolve()),
            'manifest_sha256':sha(root/'manifest.json'),'he_array':str(he.resolve()),'he_stat':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},'shards':shards,'rows':report['rows']})
        raw.extend(Path(m[k]).parent for k in ('he','mif') if k in m)
    ck=Path(a.checkpoint).resolve()
    checkpoint_hash=sha(ck)
    if any(read_meta(r)['checkpoint_sha256']!=checkpoint_hash for r in roots):raise ValueError('UNI2 weights differ')
    out=independent_output(a.out,[base,*raw]);out.mkdir(parents=True);snapshot(out,vars(a))
    c={'source_run':str(base),'tasks':tasks,'folds':folds,'checkpoint':str(ck),'checkpoint_sha256':checkpoint_hash,
       'recipe':RECIPE,'seed':42,'variants':VARIANTS,'training':{'epochs':a.epochs,'min_epochs':a.min_epochs,'patience':a.patience,'batch_size':256,'lr':.001},
       'identity_views':'existing caches referenced read-only; never copied/removed','new_views':['geometry','geometry_color'],'software_versions':software_versions(),'source_hashes':{p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.md','.sh')}}
    if not 1<=a.min_epochs<=a.epochs or a.min_epochs<=50 or a.patience<1:raise ValueError('Need epochs>=min_epochs>50, positive patience')
    c['fingerprint']=fingerprint(c);write_json(out/'plan.json',c);write_json(out/'folds.json',folds)
    (out/'views').mkdir();(out/'runs').mkdir();code=out/'source';code.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.sh'):(code/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'plan':str(out),'patients':len(patients),'locations':sum(t['rows'] for t in tasks),
        'new_feature_bytes':sum(t['rows'] for t in tasks)*1536*4*2,'original_features_unchanged':True},indent=2))


def verify_record(base,record,source):
    if any(record.get(k)!=source[k] for k in ('name','rows','source_sha256','xy_sha256')):raise ValueError('View/source shard identity differs')
    for key in ('geometry','geometry_color','parameters'):
        if sha(base/record[key])!=record[key+'_sha256']:raise ValueError('Committed cache checksum differs')
    for key in ('geometry','geometry_color'):
        a=np.load(base/record[key],mmap_mode='r',allow_pickle=False)
        if a.shape!=(source['rows'],1536) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid committed features')


def verify_views(plan,task):
    c=config(plan);base=Path(plan)/'views'/task['sample_id'];m=json.loads((base/'manifest.json').read_text())
    if not m.get('complete') or m['plan_fingerprint']!=c['fingerprint']:raise ValueError('Incomplete/different augmentation recipe')
    check_original(task)
    if len(m['shards'])!=len(task['shards']):raise ValueError('Shard count mismatch')
    for r,source in zip(m['shards'],task['shards']):
        if r['name']!=source['name'] or r['rows']!=source['rows'] or r['source_sha256']!=source['source_sha256']:raise ValueError('View/source alignment mismatch')
        for k in ('geometry','geometry_color','parameters'):
            if sha(base/r[k])!=r[k+'_sha256']:raise ValueError('Augmentation cache checksum differs')
        for k in ('geometry','geometry_color'):
            v=np.load(base/r[k],mmap_mode='r',allow_pickle=False)
            if v.shape!=(source['rows'],1536) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('Invalid view features')
    return m


def extract(plan,ids,device,batch_size,resume=False):
    import torch
    if batch_size<2:raise ValueError('Batch size>=2 required (two views/location)')
    plan=Path(plan);c=config(plan)
    if software_versions()!=c['software_versions']:raise ValueError('Software versions changed since planning; use the recorded environment')
    selected=[c['tasks'][i] for i in ids];todo=[]
    for t in selected:
        out=plan/'views'/t['sample_id']
        if out.exists():
            if not resume:raise FileExistsError(out)
            if (out/'manifest.json').exists():
                verify_views(plan,t);print('Verified completed cache; skipping '+t['sample_id'],flush=True)
            else:
                partial=json.loads((out/'partial_plan.json').read_text())
                if partial!={'plan_fingerprint':c['fingerprint'],'sample_id':t['sample_id']}:raise ValueError('Partial cache belongs to another recipe')
                todo.append(t)
        else:todo.append(t)
    if not todo:return
    encoder,transform=load_encoder(c['checkpoint'],c['checkpoint_sha256'],device)
    for task in todo:
        m=check_original(task);he=np.load(task['he_array'],mmap_mode='r',allow_pickle=False)
        out=plan/'views'/task['sample_id']
        if not out.exists():
            out.mkdir();write_json(out/'partial_plan.json',{'plan_fingerprint':c['fingerprint'],'sample_id':task['sample_id']})
        records=[];started=time.monotonic();processed=0;encoded=0
        for i,record in enumerate(task['shards']):
            committed=out/f'shard_record_{i:05d}.json'
            if committed.exists():
                saved=json.loads(committed.read_text());verify_record(out,saved,record)
                records.append(saved);processed+=saved['rows'];print('Reused committed shard '+str(i),flush=True);continue
            with np.load(Path(task['cache'])/record['name'],allow_pickle=False) as z:xy=z['xy'].copy()
            if hashlib.sha256(xy.tobytes()).hexdigest()!=record['xy_sha256']:raise ValueError('Coordinate hash mismatch')
            if i==0:
                from PIL import Image
                count=min(3,len(xy));images=[]
                for x,y in xy[:count]:
                    x,y=int(x),int(y);images.append(transform(Image.fromarray(he[y-64:y+64,x-64:x+64])))
                with torch.inference_mode():identity=encoder(torch.stack(images).to(device)).cpu().float().numpy()
                with np.load(Path(task['cache'])/record['name'],allow_pickle=False) as z:reference=z['features'][:count]
                compatible=np.allclose(identity,reference,rtol=1e-3,atol=1e-4)
                write_json(out/'identity_reproduction.json',{'matched':bool(compatible),'locations':count,'max_absolute_difference':float(np.max(np.abs(identity-reference))),'rtol':1e-3,'atol':1e-4,'software_versions':software_versions()})
                if not compatible:raise ValueError('Recomputed identity features differ from cached originals; inspect identity_reproduction.json before augmentation')
            n=len(xy);g=np.empty((n,1536),np.float32);gc=np.empty_like(g)
            params=np.asarray([parameters(task['sample_id'],x,y,c['seed']) for x,y in xy])
            if i==0 and not (out/'preview.png').exists():preview(he,xy[:3],params[:3],out/'preview.png')
            per_batch=max(1,batch_size//2)
            def prepare_batch(start,end):
                images=[]
                for j in range(start,end):
                    x,y=map(int,xy[j]);pair=views(he[y-64:y+64,x-64:x+64],params[j]);images.extend(transform(v) for v in pair)
                batch=torch.stack(images)
                return batch.pin_memory() if str(device).startswith('cuda') else batch
            starts=list(range(0,n,per_batch))
            with ThreadPoolExecutor(max_workers=1) as prefetch:
                pending=prefetch.submit(prepare_batch,starts[0],min(n,starts[0]+per_batch))
                for step,start in enumerate(starts):
                    end=min(n,start+per_batch);cpu_batch=pending.result()
                    if step+1<len(starts):
                        following=starts[step+1];pending=prefetch.submit(prepare_batch,following,min(n,following+per_batch))
                    batch=cpu_batch.to(device,non_blocking=str(device).startswith('cuda'))
                    with torch.inference_mode():features=encoder(batch)
                    if features.shape!=(2*(end-start),1536) or not torch.isfinite(features).all() or (features.norm(dim=1)<=0).any():raise ValueError('Unexpected UNI2 output')
                    encoded+=len(batch)
                    f=features.cpu().float().numpy();g[start:end]=f[0::2];gc[start:end]=f[1::2]
                    if start==0 or end==n or step%32==0:print(json.dumps({'sample':task['sample_id'],'shard':i,'locations_done':end,'locations_total':n,'total_images_done':2*(processed+end),'images_per_second':encoded/max(time.monotonic()-started,1e-9)}),flush=True)
            stem=f'view_{i:05d}_{uuid.uuid4().hex[:12]}';names={'geometry':stem+'_geometry.npy','geometry_color':stem+'_geometry_color.npy','parameters':stem+'_parameters.npz'}
            with (out/names['geometry']).open('xb') as f:np.save(f,g)
            with (out/names['geometry_color']).open('xb') as f:np.save(f,gc)
            with (out/names['parameters']).open('xb') as f:np.savez(f,xy=xy,parameters=params)
            saved={**record,**names,**{k+'_sha256':sha(out/v) for k,v in names.items()}}
            write_json(committed,saved);records.append(saved);processed+=n
            print(json.dumps({'sample':task['sample_id'],'shards_done':i+1,'shards_total':len(task['shards']),'rows':n,'seconds':time.monotonic()-started}),flush=True)
        check_original(task)
        write_json(out/'manifest.json',{'complete':True,'plan_fingerprint':c['fingerprint'],'sample_id':task['sample_id'],'recipe':c['recipe'],'shards':records,'identity_cache':task['cache']})
        print('Complete augmented views: '+task['sample_id'],flush=True)


def train_job(plan,variant,fold,device):
    c=config(plan);plan=Path(plan)
    if variant not in VARIANTS or not 0<=fold<len(c['folds']):raise ValueError('Invalid job')
    check_verification(plan)
    dest=plan/'runs'/variant/f'fold_{fold:02d}';dest.mkdir(parents=True,exist_ok=False);write_json(dest/'split.json',c['folds'][fold])
    command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--caches',*[t['cache'] for t in c['tasks']],
       '--split',str(dest/'split.json'),'--out',str(dest/'training'),'--augmentation-plan',str(plan),'--variant',variant,
       '--mode','weighted','--seed',str(c['seed']+fold),'--device',device,'--max-numeric-gb','24']
    for k,v in c['training'].items():command+=['--'+k.replace('_','-'),str(v)]
    print('Trainer log: '+str(dest/'training.log'),flush=True)
    with (dest/'training.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    evaluate(plan,variant,fold)


def evaluate(plan,variant,fold):
    from model import load_checkpoint
    from metrics import metrics
    from expression import subgroup_metrics
    plan=Path(plan);c=config(plan);dest=plan/'runs'/variant/f'fold_{fold:02d}'
    model,ck=load_checkpoint(dest/'training/best.pt','cpu');split=c['folds'][fold]
    if ck['split']!=split or ck['variant']!=variant or ck['augmentation_plan_fingerprint']!=c['fingerprint']:raise ValueError('Checkpoint role mismatch')
    roots=[Path(t['cache']) for t in c['tasks']];groups=partitions(roots,split);panel=ck['panel'];patient_results={};expression_results={}
    out=dest/'outer_predictions';out.mkdir()
    for patient in split['test']:
        selected=[r for r in groups['test'] if read_meta(r)['patient_id']==patient]
        x,y,ids=load_group(selected);pred=scale(predict(model,x,'cpu',256),ck['scaler'],inverse=True)
        patient_results[patient]=metrics(y,pred,ids,panel);expression_results[patient]=subgroup_metrics(y,pred,ids,panel,ck['expression_state'])
        offset=0
        for root in selected:
            meta=read_meta(root);d=out/meta['sample_id'];d.mkdir()
            for name in meta['shards']:
                with np.load(root/name,allow_pickle=False) as z:xy=z['xy'];target=z['target']
                n=len(target);np.savez(d/name,xy=xy,target=target,prediction=pred[offset:offset+n]);offset+=n
            write_json(d/'manifest.json',{**meta,'complete':True,'outer_fold':fold,'variant':variant,'inference_view':'original only','checkpoint':str(dest/'training/best.pt')})
    write_json(dest/'patient_metrics.json',patient_results);write_json(dest/'expression_patients.json',expression_results)
    write_json(dest/'manifest.json',{'complete':True,'variant':variant,'fold':fold,'plan_fingerprint':c['fingerprint'],'patients':list(patient_results)})


def aggregate(plan):
    from primary_evaluation import save_primary
    from expression import summarize_subgroups
    plan=Path(plan);c=config(plan);panel=read_meta(c['tasks'][0]['cache'])['panel'];tables={};all_patients={}
    for variant in VARIANTS:
        patients={};expr={}
        for i,split in enumerate(c['folds']):
            d=plan/'runs'/variant/f'fold_{i:02d}';m=json.loads((d/'manifest.json').read_text())
            if not m.get('complete') or m.get('plan_fingerprint')!=c['fingerprint'] or m.get('variant')!=variant or m.get('fold')!=i or set(m['patients'])!=set(split['test']):raise ValueError('Incomplete fold')
            r=json.loads((d/'patient_metrics.json').read_text())
            if set(r)!=set(split['test']) or set(r)&set(patients):raise ValueError('Wrong/duplicate outer patient')
            patients.update(r);expr.update(json.loads((d/'expression_patients.json').read_text()))
        out=plan/'runs'/variant/'aggregate';out.mkdir()
        merged={m:{'patients':{pid:r[m]['patients'][pid] for pid,r in patients.items()}} for m in panel}
        save_primary(out,merged,'out_of_fold',2000,42);tables[variant]=json.loads((out/'primary_summary.json').read_text())['metrics']
        all_patients[variant]=patients
        write_json(out/'expression_summary.json',summarize_subgroups(expr,panel))
        write_json(out/'manifest.json',{'complete':True,'variant':variant,'patients':len(patients),'original_view_test':True})
    # Audit equivalence of initialization, unique-target transforms and draw sequence.
    audit=[]
    for i in range(len(c['folds'])):
        states=[];histories=[]
        for variant in VARIANTS:
            d=plan/'runs'/variant/f'fold_{i:02d}'/'training'
            states.append([json.loads((d/n).read_text()) for n in ('initial_state.json','target_scaler.json','expression_state.json','config.json')])
            histories.append(json.loads((d/'history.json').read_text()))
        if not states[0]==states[1]==states[2]:raise ValueError('Matched initialization/scaler/weights differ')
        overlap=min(map(len,histories))
        for epoch in range(overlap):
            if len({h[epoch]['sampler_draw_sha256'] for h in histories})!=1:raise ValueError('Location sampler draws differ')
            if len({h[epoch]['steps'] for h in histories})!=1:raise ValueError('Per-epoch training budget differs')
            if histories[1][epoch]['augmented_draws']!=histories[2][epoch]['augmented_draws'] or histories[1][epoch]['view_draw_sha256']!=histories[2][epoch]['view_draw_sha256']:raise ValueError('Augmented-view draws differ')
        audit.append({'fold':i,'matched_initialization_scaler_weights':True,'matched_sampler_epochs_checked':overlap,'actual_epochs':list(map(len,histories))})
    comparison=plan/'comparison';comparison.mkdir()
    with (comparison/'all_marker_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['marker','variant','mae','pearson','spearman','bias'])
        for marker in panel:
            for v in VARIANTS:
                w.writerow([marker,v,*[tables[v][marker][k]['patient_mean'] for k in ('mae','pearson','spearman','mean_bias')]])
    paired=[]
    for baseline,candidate in [('control','geometry'),('control','geometry_color'),('geometry','geometry_color')]:
        ids=sorted(all_patients[baseline])
        for marker in panel:
            for metric in ('mae','pearson','spearman','mean_bias'):
                differences=[]
                for pid in ids:
                    b=all_patients[baseline][pid][marker]['patients'][pid][metric];a=all_patients[candidate][pid][marker]['patients'][pid][metric]
                    if b is not None and a is not None:differences.append(a-b)
                delta,lo,hi=paired_interval(differences)
                paired.append({'baseline':baseline,'candidate':candidate,'marker':marker,'metric':metric,'patient_mean_delta':delta,'ci95_low':lo,'ci95_high':hi,'defined_pairs':len(differences)})
    with (comparison/'paired_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    write_json(comparison/'matched_audit.json',audit);write_json(plan/'complete.json',{'complete':True,'variants':VARIANTS})
    print(f'ALL COMPLETE: {comparison}/all_marker_metrics.csv',flush=True)


def paired_interval(differences):
    values=np.asarray(differences,float)
    if not len(values):return None,None,None
    if len(values)==1:return float(values[0]),None,None
    draws=np.random.default_rng(42).integers(0,len(values),size=(2000,len(values)))
    lo,hi=np.percentile(values[draws].mean(1),[2.5,97.5])
    return float(values.mean()),float(lo),float(hi)


def preview(he,xy,params,path):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(672,len(xy)*248),'white');draw=ImageDraw.Draw(canvas)
    for row,((x,y),param) in enumerate(zip(xy,params)):
        x,y=int(x),int(y);rgb=he[y-64:y+64,x-64:x+64];pair=views(rgb,param)
        for col,(im,title) in enumerate(zip((Image.fromarray(rgb),*pair),('Identity','Geometry','Geometry + color'))):
            canvas.paste(im.resize((224,224),Image.Resampling.BILINEAR),(col*224,row*248+20))
            draw.text((col*224+4,row*248+3),title,fill='black')
    canvas.save(path)


def file_stat(path):
    s=Path(path).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def verify_all(plan):
    plan=Path(plan);c=config(plan);stats={}
    for task in c['tasks']:
        m=verify_views(plan,task);base=plan/'views'/task['sample_id'];original=Path(task['cache'])
        files=[original/'manifest.json',Path(task['he_array']),base/'manifest.json']
        for record in m['shards']:
            files.append(original/record['name'])
            files.extend(base/record[k] for k in ('geometry','geometry_color','parameters'))
        stats.update({str(p.resolve()):file_stat(p) for p in files})
    write_json(plan/'verification.json',{'complete':True,'plan_fingerprint':c['fingerprint'],'file_stats':stats})
    print('All original and augmented caches verified; files unchanged.',flush=True)


def check_verification(plan):
    plan=Path(plan);c=config(plan);v=json.loads((plan/'verification.json').read_text())
    if not v.get('complete') or v['plan_fingerprint']!=c['fingerprint']:raise ValueError('Run verify before training')
    for path,stat in v['file_stats'].items():
        if file_stat(path)!=stat:raise ValueError('Verified cache changed: '+path)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--source-run',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--out',required=True)
    q.add_argument('--epochs',type=int,default=100);q.add_argument('--min-epochs',type=int,default=51);q.add_argument('--patience',type=int,default=10)
    q=sub.add_parser('extract');q.add_argument('--plan',required=True);q.add_argument('--task-ids',nargs='+',type=int,required=True);q.add_argument('--device',default='cuda:0');q.add_argument('--batch-size',type=int,default=16);q.add_argument('--resume-complete',action='store_true')
    q=sub.add_parser('train');q.add_argument('--plan',required=True);q.add_argument('--variant',choices=VARIANTS,required=True);q.add_argument('--fold',type=int,required=True);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('collect');q.add_argument('--plan',required=True)
    q=sub.add_parser('verify');q.add_argument('--plan',required=True)
    a=p.parse_args()
    if a.command=='plan':plan(a)
    elif a.command=='extract':
        c=config(a.plan)
        if len(set(a.task_ids))!=len(a.task_ids) or any(not 0<=i<len(c['tasks']) for i in a.task_ids):raise ValueError('Invalid task IDs')
        extract(a.plan,a.task_ids,a.device,a.batch_size,a.resume_complete)
    elif a.command=='train':train_job(a.plan,a.variant,a.fold,a.device)
    elif a.command=='verify':verify_all(a.plan)
    else:aggregate(a.plan)

if __name__=='__main__':main()
