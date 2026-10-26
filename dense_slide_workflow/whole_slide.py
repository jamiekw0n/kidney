"""Independent stride-8 band inference and fixed-scale whole-slide assembly."""
import argparse,json,time
from pathlib import Path
import numpy as np
from audit import write_json,snapshot
from data import read_meta,scale
from io_utils import independent_output,SIGNATURE
from raw_safety import source_stats
from spatial import band_geometry,inference_mask
from normalization import normalize_pixels
from encoder_runtime import load_encoder,encode,sha
from workflow import run_config


def plan(a):
    run=Path(a.run).resolve();c=run_config(run);matches=[t for t in c['tasks'] if t['sample_id']==a.sample]
    if len(matches)!=1:raise ValueError('Sample absent or ambiguous')
    task=matches[0];meta=read_meta(task['normalized_cache'])
    if a.markers==['ALL']:a.markers=list(meta['panel'])
    folds=json.loads((run/'folds.json').read_text())
    indices=[i for i,f in enumerate(folds) if task['patient_id'] in f['test']]
    if len(indices)!=1:raise ValueError('Sample must be held out exactly once')
    i=indices[0]
    if task['patient_id'] in folds[i]['train']+folds[i]['val']:raise ValueError('Held-out leakage')
    if len(a.markers)!=len(set(a.markers)) or set(a.markers)-set(meta['panel']):raise ValueError('Unknown/duplicate marker')
    if a.band_pixels<8 or a.band_pixels%8:raise ValueError('Band height must be multiple of 8')
    ny,nx=meta['shape'][0]//8,meta['shape'][1]//8
    if not nx or not ny:raise ValueError('Slide too small')
    rows=a.band_pixels//8;bands=(ny+rows-1)//rows
    out=independent_output(a.out,[Path(task['normalized_cache']),*[Path(p).parent for p in task['raw_paths'].values()]])
    out.mkdir(parents=True);snapshot(out,vars(a));(out/'bands').mkdir()
    cfg={'run':str(run),'sample_id':a.sample,'patient_id':task['patient_id'],'markers':a.markers,'fold':i,
         'rows_per_band':rows,'bands':bands,'shape':meta['shape'],'grid_shape':[ny,nx],
         'checkpoint':str(run/'cv'/f'part_{i:02d}'/f'fold_{i:02d}'/'training'/'best.pt'),
         'support':'same H&E tissue rules as training; unsupported/border cells masked, not zero predictions',
         'stride':8,'target_block':8,'context':128,'color_limits':[0,1]}
    write_json(out/'slide.json',cfg)
    print(json.dumps({'slide_plan':str(out),'bands':bands,'total_grid_locations':ny*nx,
        'upper_bound_inference_seconds_at_50_crops_per_second':ny*nx/50,
        'note':'Illustrative throughput, NOT a measured ETA; tissue filtering reduces encoder work'},indent=2))


def infer(a):
    import torch
    from model import load_checkpoint
    from image_io import TiffRegionReader
    root=Path(a.slide_plan).resolve();s=json.loads((root/'slide.json').read_text());run=Path(s['run']);c=run_config(run)
    ny,nx,start,stop,total=band_geometry(*s['shape'],s['rows_per_band'],a.task_id)
    ref=run/'cache'/s['sample_id'];meta=read_meta(ref);prepared=run/'prepared'/s['sample_id']
    row={'he':meta['he'],'mif':meta['mif'],'folder':str(Path(meta['mif']).parent)};before=source_stats(row)
    if before!=json.loads((prepared/'raw_input_check.json').read_text())['after']:raise ValueError('Raw input changed since preparation')
    out=independent_output(root/'bands'/f'band_{a.task_id:05d}',[ref,prepared,*[Path(meta[k]).parent for k in ('he','mif')]])
    # Head is checked before loading the expensive encoder; only its outer-test fold may predict.
    head,ck=load_checkpoint(s['checkpoint'],a.device)
    if s['patient_id'] not in ck['split']['test'] or s['patient_id'] in ck['split']['train']+ck['split']['val']:raise ValueError('Patient leakage in checkpoint')
    if ck['panel']!=meta['panel'] or any(ck['cache_signature'].get(k)!=meta.get(k) for k in SIGNATURE):raise ValueError('Head/cache preprocessing mismatch')
    he_path=prepared/'he.npy';he_stat=he_path.stat()
    if {'size':he_stat.st_size,'mtime_ns':he_stat.st_mtime_ns}!=meta['he_array_stat']:raise ValueError('Derived H&E changed')
    he=np.load(he_path,mmap_mode='r',allow_pickle=False)
    if list(he.shape)!=[*s['shape'],3]:raise ValueError('Derived H&E geometry mismatch')
    xy,valid=inference_mask(he,start,stop);count=int(valid.sum())
    print(json.dumps({'stage':'band_start','band':a.task_id,'bands':total,'grid_locations':len(xy),'supported_locations':count,'row_start':start,'row_stop':stop}),flush=True)
    j=[meta['panel'].index(m) for m in s['markers']]
    report=json.loads((prepared/'normalization.json').read_text());norm_source=Path(meta['source_normalized_cache'])
    if sha(norm_source/'normalization.json')!=next(t['normalization_sha256'] for t in c['tasks'] if t['sample_id']==s['sample_id']):raise ValueError('Normalization reference changed')
    if report!=json.loads((norm_source/'normalization.json').read_text()):raise ValueError('Normalization report differs')
    div=np.asarray([v['effective_divisor'] for v in report['markers']])[j]
    from prepare import header
    mi=header(meta['mif'],report.get('series',0),report.get('level',0));labels=mi['markers']
    if len(labels)!=len(set(labels)) or set(labels)!=set(meta['panel']):raise ValueError('Measured channel panel differs')
    order=[labels.index(m) for m in s['markers']]
    reader=TiffRegionReader(meta['mif'],len(meta['panel']),report['source_axes'],report.get('series',0),report.get('level',0))
    try:
        if [reader.height,reader.width]!=s['shape']:raise ValueError('Measured image geometry mismatch')
        # Bounded blocks avoid an entire 42-channel band allocation.
        measured=np.empty((stop-start,nx,len(j)),np.float32)
        for x0 in range(0,nx*8,512):
            if x0==0 or x0%4096==0:print(json.dumps({'stage':'measured_mif','band':a.task_id,'x':x0,'width':nx*8}),flush=True)
            x1=min(nx*8,x0+512);raw=reader.read(start*8,stop*8,x0,x1)[:,:,order]
            v=normalize_pixels(raw,div).reshape(stop-start,8,(x1-x0)//8,8,len(j)).mean((1,3))
            measured[:,x0//8:x1//8]=v
    finally:reader.close()
    predictions=np.full((len(xy),len(j)),np.nan,np.float32);started=time.monotonic()
    if count:
        encoder,transform=load_encoder(c['checkpoint'],c['checkpoint_sha256'],a.device)
        indices=np.flatnonzero(valid)
        for b in range(0,len(indices),a.batch_size):
            ix=indices[b:b+a.batch_size];features=encode(encoder,transform,he,xy[ix],a.device)
            with torch.inference_mode():p=head(torch.from_numpy(features).to(a.device)).cpu().numpy()
            predictions[ix]=scale(p,ck['scaler'],inverse=True)[:,j]
            if b==0 or b%(a.batch_size*100)==0 or b+a.batch_size>=len(indices):
                elapsed=time.monotonic()-started;done=min(b+a.batch_size,len(indices))
                print(json.dumps({'band':a.task_id,'done':done,'total':count,'crops_per_second':done/max(elapsed,1e-9),
                    'estimated_remaining_seconds':(count-done)*elapsed/done}),flush=True)
    if not np.isfinite(predictions[valid]).all():raise ValueError('Nonfinite supported predictions')
    after=source_stats(row)
    if before!=after:raise ValueError('Raw inputs changed during inference')
    if he_path.stat().st_mtime_ns!=he_stat.st_mtime_ns or he_path.stat().st_size!=he_stat.st_size:raise ValueError('Derived H&E changed during inference')
    rgb=he[start*8:stop*8,:nx*8].reshape(stop-start,8,nx,8,3).mean((1,3)).astype(np.uint8)
    out.mkdir();np.savez(out/'maps.npz',measured=measured,prediction=predictions.reshape(stop-start,nx,len(j)),
        valid=valid.reshape(stop-start,nx),he=rgb)
    write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':True})
    write_json(out/'manifest.json',{'complete':True,'band':a.task_id,'row_start':start,'row_stop':stop,'grid_shape':[ny,nx],
        'sample_id':s['sample_id'],'markers':s['markers'],'supported_locations':count,'checkpoint_sha256':sha(s['checkpoint']),
        'normalization_sha256':sha(prepared/'normalization.json'),'encoder_sha256':c['checkpoint_sha256'],
        'slide_plan_sha256':sha(root/'slide.json'),'inference_seconds':time.monotonic()-started})
    print(f'Complete band: {out}',flush=True)


def assemble(a):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(a.slide_plan).resolve();s=json.loads((root/'slide.json').read_text());ny,nx=s['grid_shape'];m=len(s['markers'])
    out=independent_output(a.out,[root]);out.mkdir(parents=True)
    # Disk-backed grids limit RAM; all bands must complete before final manifest.
    measured=np.lib.format.open_memmap(out/'measured.npy',mode='w+',dtype='float32',shape=(ny,nx,m))
    pred=np.lib.format.open_memmap(out/'prediction.npy',mode='w+',dtype='float32',shape=(ny,nx,m))
    valid=np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype='bool',shape=(ny,nx))
    rgb=np.lib.format.open_memmap(out/'he.npy',mode='w+',dtype='uint8',shape=(ny,nx,3))
    head_hash=None;norm_hash=None
    for i in range(s['bands']):
        band=root/'bands'/f'band_{i:05d}';meta=json.loads((band/'manifest.json').read_text())
        _,_,start,stop,_=band_geometry(*s['shape'],s['rows_per_band'],i)
        if not meta.get('complete') or meta['band']!=i or meta['row_start']!=start or meta['row_stop']!=stop or meta['slide_plan_sha256']!=sha(root/'slide.json'):raise ValueError('Band plan/geometry mismatch')
        if meta['markers']!=s['markers']:raise ValueError('Band marker mismatch')
        if head_hash is not None and (meta['checkpoint_sha256']!=head_hash or meta['normalization_sha256']!=norm_hash):raise ValueError('Mixed checkpoints or normalizations')
        head_hash=meta['checkpoint_sha256'];norm_hash=meta['normalization_sha256']
        with np.load(band/'maps.npz',allow_pickle=False) as z:
            shape=(stop-start,nx,m)
            if z['measured'].shape!=shape or z['prediction'].shape!=shape or z['valid'].shape!=shape[:2] or z['he'].shape!=(*shape[:2],3):raise ValueError('Band array shapes differ')
            if not np.isfinite(z['measured']).all() or not np.isfinite(z['prediction'][z['valid']]).all():raise ValueError('Nonfinite supported maps')
            measured[start:stop]=z['measured'];pred[start:stop]=z['prediction'];valid[start:stop]=z['valid'];rgb[start:stop]=z['he']
    for v in (measured,pred,valid,rgb):v.flush()
    # Thumbnail limit changes display sampling only; exported grids retain every block.
    step=max(1,int(np.ceil(max(ny,nx)/a.max_display_side)))
    extent=(0,nx*8,ny*8,0);cards=[];stats=[]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede')
    for j,marker in enumerate(s['markers']):
        fig,axes=plt.subplots(1,3,figsize=(18,9),constrained_layout=True)
        axes[0].imshow(rgb[::step,::step],extent=extent);axes[0].set_title('Whole-slide H&E')
        axes[1].imshow(measured[::step,::step,j],extent=extent,cmap=cmap,vmin=0,vmax=1,interpolation='nearest');axes[1].set_title('Measured mIF — normalized 8×8 means')
        values=np.ma.masked_where(~valid[::step,::step],pred[::step,::step,j])
        im=axes[2].imshow(values,extent=extent,cmap=cmap,vmin=0,vmax=1,interpolation='nearest');axes[2].set_title('Held-out prediction — stride 8')
        for ax in axes:ax.set_xlabel('Native x');ax.set_ylabel('Native y')
        fig.colorbar(im,ax=list(axes[1:]),label='Per-slide/channel P99 normalized intensity',shrink=.6,extend='both')
        fig.suptitle(f'{s["sample_id"]} | {marker} | outer fold {s["fold"]}; gray = no prediction')
        name=f'marker_{j:02d}.png';fig.savefig(out/name,dpi=160);plt.close(fig)
        cards.append(f'<h2>{marker}</h2><img style="max-width:100%" src="{name}">')
        vals=np.asarray(pred[:,:,j][valid]);stats.append({'marker':marker,'predicted_locations':len(vals),
            'below_zero_fraction':float(np.mean(vals<0)) if len(vals) else None,'above_one_fraction':float(np.mean(vals>1)) if len(vals) else None})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Whole-slide held-out review</h1><p>Fixed color scale 0–1 across markers. mIF is pixel-normalized before 8×8 averaging. Gray predictions are unsupported, not zero. Bottom/right remainders &lt;8 pixels omitted. H&E is block-averaged for display. Full arrays remain at stride 8.</p>'+''.join(cards))
    write_json(out/'display_statistics.json',stats)
    write_json(out/'manifest.json',{'complete':True,**s,'source_slide_plan':str(root),'checkpoint_sha256':head_hash,
        'normalization_sha256':norm_hash,'display_subsample_step':step,'predicted_locations':int(valid.sum())})
    print(f'Gallery: {out}/index.html',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--run',required=True);q.add_argument('--sample',required=True);q.add_argument('--out',required=True)
    q.add_argument('--markers',nargs='+',default=['CAV','CD8','CD138','COLIV','POSTN','VIM']);q.add_argument('--band-pixels',type=int,default=512)
    q=sub.add_parser('infer');q.add_argument('--slide-plan',required=True);q.add_argument('--task-id',type=int,required=True);q.add_argument('--batch-size',type=int,default=16);q.add_argument('--device',default='cuda')
    q=sub.add_parser('assemble');q.add_argument('--slide-plan',required=True);q.add_argument('--out',required=True);q.add_argument('--max-display-side',type=int,default=2500)
    a=p.parse_args()
    if a.command=='plan':plan(a)
    elif a.command=='infer':
        if a.batch_size<1:raise ValueError('Positive batch size required')
        infer(a)
    else:
        if a.max_display_side<1:raise ValueError('Positive display size required')
        assemble(a)

if __name__=='__main__':main()
