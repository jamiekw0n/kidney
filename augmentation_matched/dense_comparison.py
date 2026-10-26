"""Read-only original-view dense inference through three matched outer-test heads.
Reference is an existing whole_slide.py assembled *_figures directory.
New outputs only; two persistent GPU workers; immutable per-band feature caches.
"""
import argparse,json,sys,subprocess,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pipeline import config,VARIANTS
from encoder_runtime import sha,load_encoder,encode
from data import read_meta,scale
from io_utils import SIGNATURE,independent_output
from dense_spatial import inference_mask

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def verify_reference_encoder(reference,expected):
    """Older assembled manifests omit encoder hash; their bands retain it."""
    value=reference.get('encoder_sha256')
    if value is not None:
        if value!=expected:raise ValueError('Reference encoder hash differs from augmentation encoder')
        return
    source=Path(reference['source_slide_plan'])
    slide=read(source/'slide.json')
    for key in ('sample_id','fold','bands','markers','grid_shape'):
        if slide[key]!=reference[key]:raise ValueError('Reference slide plan metadata differs: '+key)
    for i in range(reference['bands']):
        band=read(source/'bands'/f'band_{i:05d}'/'manifest.json')
        if not band.get('complete') or band['band']!=i or band['sample_id']!=reference['sample_id']:
            raise ValueError('Reference band incomplete or wrong identity')
        if band.get('encoder_sha256')!=expected:raise ValueError(f'Reference band {i} encoder hash differs')
        if band.get('normalization_sha256')!=reference['normalization_sha256'] or band['markers']!=reference['markers']:
            raise ValueError('Reference band normalization/panel differs')
        if band['slide_plan_sha256']!=sha(source/'slide.json'):raise ValueError('Reference band slide plan changed')

def inputs(a):
    c=config(a.plan);matches=[t for t in c['tasks'] if t['sample_id']==a.sample]
    if len(matches)!=1:raise ValueError('Unknown sample')
    t=matches[0];meta=read_meta(t['cache']);ref=Path(a.reference).resolve();r=read(ref/'manifest.json')
    fold=[i for i,f in enumerate(c['folds']) if t['patient_id'] in f['test']]
    if len(fold)!=1:raise ValueError('Patient must be held out exactly once')
    if not r.get('complete') or r['sample_id']!=a.sample or r['fold']!=fold[0] or Path(r['run']).resolve()!=Path(c['source_run']).resolve():raise ValueError('Reference sample/fold/source mismatch')
    verify_reference_encoder(r,c['checkpoint_sha256'])
    if meta.get('checkpoint_sha256')!=c['checkpoint_sha256']:raise ValueError('Source cache encoder differs')
    normalization=Path(c['source_run'])/'prepared'/a.sample/'normalization.json'
    if r.get('normalization_sha256')!=sha(normalization):raise ValueError('Reference normalization differs')
    expected=(meta['shape'][0]//8,meta['shape'][1]//8)
    for name,shape in [('measured',(*expected,len(r['markers']))),('valid',expected),('he',(*expected,3))]:
        z=np.load(ref/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if z.shape!=shape:raise ValueError('Reference geometry mismatch: '+name)
    stat=Path(t['he_array']).stat()
    if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=t['he_stat']:raise ValueError('Derived H&E changed')
    if sha(Path(t['cache'])/'manifest.json')!=t['manifest_sha256']:raise ValueError('Source cache metadata changed')
    return c,t,meta,r,fold[0]

def worker(a):
    import torch
    from model import load_checkpoint
    root=Path(a.out);p=read(root/'comparison_plan.json');c,t,meta,r,fold=inputs(a)
    he=np.load(t['he_array'],mmap_mode='r',allow_pickle=False)
    valid=np.load(Path(a.reference)/'valid.npy',mmap_mode='r',allow_pickle=False)
    heads={}
    for v in VARIANTS:
        path=Path(a.plan)/'runs'/v/f'fold_{fold:02d}'/'training/best.pt'
        head,ck=load_checkpoint(path,a.device)
        if sha(path)!=p['checkpoints'][v] or ck['variant']!=v or ck['augmentation_plan_fingerprint']!=c['fingerprint'] or ck['split']!=c['folds'][fold]:raise ValueError('Checkpoint mismatch')
        if ck['panel']!=meta['panel'] or any(ck['cache_signature'].get(k)!=meta.get(k) for k in SIGNATURE):raise ValueError('Preprocessing mismatch')
        heads[v]=(head,ck)
    encoder=None;transform=None;ny,nx=valid.shape
    for band in range(a.worker_id,p['bands'],a.workers):
        start=band*p['band_rows'];stop=min(ny,start+p['band_rows']);d=root/'bands'/f'{band:05d}'
        if (d/'complete.json').exists():
            record=read(d/'complete.json')
            if record['plan_sha256']!=sha(root/'comparison_plan.json') or any(sha(d/k)!=v for k,v in record['files'].items()):raise ValueError('Completed band corrupted')
            print(f'REUSE band {band}',flush=True);continue
        if d.exists():raise FileExistsError(f'Partial band preserved: {d}. Use a new output path for retry.')
        d.mkdir();xy,mask=inference_mask(he,start,stop)
        if not np.array_equal(mask.reshape(stop-start,nx),valid[start:stop]):raise ValueError('Reference tissue/support mask differs')
        ix=np.flatnonzero(mask);coords=xy[ix];np.save(d/'xy.npy',coords)
        features=np.lib.format.open_memmap(d/'features.npy',mode='w+',dtype='float32',shape=(len(ix),1536))
        if len(ix) and encoder is None:encoder,transform=load_encoder(c['checkpoint'],c['checkpoint_sha256'],a.device)
        started=time.monotonic()
        for b in range(0,len(ix),a.batch_size):
            features[b:b+a.batch_size]=encode(encoder,transform,he,coords[b:b+a.batch_size],a.device)
            if b==0 or b%(a.batch_size*100)==0:print(json.dumps({'band':band,'done':min(b+a.batch_size,len(ix)),'total':len(ix)}),flush=True)
        features.flush()
        for v,(head,ck) in heads.items():
            target=np.lib.format.open_memmap(d/(v+'.npy'),mode='w+',dtype='float32',shape=(stop-start,nx,len(meta['panel'])))
            target[:]=np.nan;flat=target.reshape(-1,len(meta['panel']))
            for b in range(0,len(ix),1024):
                with torch.inference_mode():pred=head(torch.from_numpy(np.array(features[b:b+1024],copy=True)).to(a.device)).cpu().numpy()
                values=scale(pred,ck['scaler'],inverse=True)
                if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
                flat[ix[b:b+1024]]=values
            target.flush()
        files={f.name:sha(f) for f in d.glob('*.npy')}
        save(d/'complete.json',{'plan_sha256':sha(root/'comparison_plan.json'),'files':files,'start':start,'stop':stop,'seconds':time.monotonic()-started})
        print(f'COMPLETE band {band}/{p["bands"]-1}',flush=True)

def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    c,t,meta,r,fold=inputs(a);root=Path(a.out);p=read(root/'comparison_plan.json');ref=Path(a.reference)
    measured=np.load(ref/'measured.npy',mmap_mode='r');rgb=np.load(ref/'he.npy',mmap_mode='r');valid=np.load(ref/'valid.npy',mmap_mode='r')
    ny,nx=valid.shape;step=max(1,int(np.ceil(max(ny,nx)/2000)));extent=(0,nx*8,ny*8,0)
    out=root/'figures';out.mkdir();cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede');cards=[]
    for marker in p['markers']:
        j=meta['panel'].index(marker);mj=r['markers'].index(marker);fig,axes=plt.subplots(1,5,figsize=(25,9),constrained_layout=True)
        axes[0].imshow(rgb[::step,::step],extent=extent);axes[0].set_title('H&E')
        im=axes[1].imshow(measured[::step,::step,mj],extent=extent,cmap=cmap,vmin=0,vmax=1);axes[1].set_title('Measured mIF — 8×8 means')
        for ax,v,title in zip(axes[2:],VARIANTS,['Matched control','Geometry','Geometry + color']):
            display=np.full((ny,nx),np.nan,np.float32)
            for band in range(p['bands']):
                d=root/'bands'/f'{band:05d}';record=read(d/'complete.json')
                if record['plan_sha256']!=sha(root/'comparison_plan.json'):raise ValueError('Band plan differs')
                values=np.load(d/(v+'.npy'),mmap_mode='r');display[record['start']:record['stop']]=values[:,:,j]
            im=ax.imshow(np.ma.masked_invalid(display[::step,::step]),extent=extent,cmap=cmap,vmin=0,vmax=1,interpolation='nearest');ax.set_title(title)
        for ax in axes:ax.set_xlabel('Native x');ax.set_ylabel('Native y')
        fig.colorbar(im,ax=list(axes[1:]),shrink=.65,extend='both',label='Per-slide/channel P99 normalized intensity')
        fig.suptitle(f'{a.sample} | {marker} | outer test fold {fold}; gray = no prediction')
        name=f'marker_{j:02d}.png';fig.savefig(out/name,dpi=150);plt.close(fig);cards.append(f'<h2>{escape(marker)}</h2><img width="100%" src="{name}">')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Matched augmentation: held-out dense comparison</h1><p>All heads use identical ORIGINAL H&E features, no test-time augmentation. Fixed display scale 0–1; predictions are not numerically clipped. Gray means unsupported. These dense figures are qualitative; use the OOF summaries for cohort metrics.</p>'+''.join(cards))
    save(root/'complete.json',{'complete':True,'gallery':str(out/'index.html'),'plan_sha256':sha(root/'comparison_plan.json')})
    print('ALL COMPLETE: '+str(out/'index.html'),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','reference','out','sample'):p.add_argument('--'+name,required=True)
    p.add_argument('--markers',nargs='+',default=['ALL']);p.add_argument('--workers',type=int,choices=[1,2],default=2);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--resume',action='store_true');p.add_argument('--worker-id',type=int);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.batch_size<1:raise ValueError('Positive batch size required')
    if a.worker_id is not None:worker(a);return
    c,t,meta,r,fold=inputs(a);root=Path(a.out).resolve()
    markers=r['markers'] if a.markers==['ALL'] else a.markers
    if len(markers)!=len(set(markers)) or set(markers)-set(r['markers']):raise ValueError('Markers must exist in reference measured maps')
    heads={v:sha(Path(a.plan)/'runs'/v/f'fold_{fold:02d}'/'training/best.pt') for v in VARIANTS}
    spec={'augmentation_fingerprint':c['fingerprint'],'sample':a.sample,'fold':fold,'reference':str(Path(a.reference).resolve()),'reference_files':{n:sha(Path(a.reference)/n) for n in ('manifest.json','measured.npy','valid.npy','he.npy')},'checkpoints':heads,'markers':markers,'band_rows':64,'bands':(meta['shape'][0]//8+63)//64,'inference':'original only, stride8, unchanged support'}
    if root.exists():
        if not a.resume or read(root/'comparison_plan.json')!=spec:raise FileExistsError('Existing output or mismatched resume plan')
        if (root/'complete.json').exists():print('Already complete: '+str(root/'figures/index.html'));return
    else:
        independent_output(root,[Path(a.plan),Path(c['source_run']),Path(a.reference),Path(meta['he']).parent]);root.mkdir(parents=True);(root/'bands').mkdir();save(root/'comparison_plan.json',spec)
    print('New dense features: up to '+str(int(np.load(Path(a.reference)/'valid.npy',mmap_mode='r').sum())*1536*4/1e9)+' GB; shared by all three heads',flush=True)
    def launch(i):
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--plan',a.plan,'--reference',a.reference,'--sample',a.sample,'--out',str(root),'--workers',str(a.workers),'--worker-id',str(i),'--device',f'cuda:{i}','--batch-size',str(a.batch_size)]
        log=root/f'worker_{i}_{time.time_ns()}.log'
        with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(launch,range(a.workers)))
    plot(a)
if __name__=='__main__':main()
