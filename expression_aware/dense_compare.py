"""Stride-8 ROI inference with held-out original/weighted MLP heads and registered H&E/mIF review."""
import argparse
from contextlib import ExitStack
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
from audit import snapshot,write_json,event
from data import read_meta,scale
from prepare import discover,header,check_scale,block_targets
from raw_safety import guard_output,source_stats


def roi_grid(roi,shape):
    x0,y0,x1,y1=roi;h,w=shape
    if min(x0,y0)<60 or x1+60>w or y1+60>h or x1<=x0 or y1<=y0 or (x1-x0)%8 or (y1-y0)%8:
        raise ValueError('ROI must have positive dimensions divisible by 8 and a 60-pixel in-bounds context margin')
    xx,yy=np.meshgrid(np.arange(x0+4,x1,8),np.arange(y0+4,y1,8))
    return np.column_stack((xx.ravel(),yy.ravel())).astype(np.int64)


def fold_head(root,patient):
    folds=json.loads((root/'folds.json').read_text())
    indices=[i for i,f in enumerate(folds) if patient in f['test']]
    if len(indices)!=1:raise ValueError('Patient must have exactly one outer-test fold')
    i=indices[0];f=folds[i]
    if patient in f['train'] or patient in f['val']:raise ValueError('Outer patient leakage')
    return i,root/f'fold_{i:02d}'/'training'/'best.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',required=True);p.add_argument('--sample',required=True)
    p.add_argument('--cache-root',required=True);p.add_argument('--baseline-root',required=True);p.add_argument('--weighted-root',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--roi',nargs=4,type=int,required=True,metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--markers',nargs='+',default=['COLIV','POSTN','CD138'])
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda')
    p.add_argument('--intensity-max',type=float,default=255)
    p.add_argument('--assume-all-valid',action='store_true',help='Required explicit support assumption; no validity mask is applied')
    a=p.parse_args()
    if not a.assume_all_valid:p.error('Requires explicit --assume-all-valid for this initial ROI workflow')
    if a.batch_size<1 or not np.isfinite(a.intensity_max) or a.intensity_max<=0:p.error('Invalid batch size/intensity ceiling')
    meta=read_meta(Path(a.cache_root)/a.sample)
    if meta.get('encoder')!='UNI2-h' or meta['input_patch']!=128 or meta['target_block']!=8:raise ValueError('Incompatible cache')
    if meta.get('transform')!='PIL bilinear square resize 224; ToTensor; ImageNet normalization':raise ValueError('Unexpected encoder transform')
    if len(set(a.markers))!=len(a.markers) or set(a.markers)-set(meta['panel']):raise ValueError('Unknown/duplicate markers')
    xy=roi_grid(a.roi,meta['shape']);x0,y0,x1,y1=a.roi;ny,nx=(y1-y0)//8,(x1-x0)//8
    rows=[r for r in discover(a.base_dir) if r['sample_id']==a.sample]
    if len(rows)!=1 or rows[0]['error']:raise ValueError('Missing/ambiguous registered pair')
    row=rows[0]
    if Path(row['he']).resolve()!=Path(meta['he']).resolve():raise ValueError('Discovered H&E differs from feature cache')
    out=Path(a.out);guard_output(out,rows)
    if out.exists():raise FileExistsError(out)
    roots=[Path(a.baseline_root),Path(a.weighted_root)]
    if json.loads((roots[0]/'folds.json').read_text())!=json.loads((roots[1]/'folds.json').read_text()):raise ValueError('CV folds differ')
    heads=[fold_head(root,meta['patient_id']) for root in roots]
    out.mkdir(parents=True);snapshot(out,vars(a));before=source_stats(row);start=time.monotonic()
    event(out,'start',locations=len(xy),roi=a.roi,outer_fold=heads[0][0],sample=a.sample)
    print(f'Sample {a.sample}: {len(xy)} predictions ({ny}x{nx}); outer fold {heads[0][0]}',flush=True)
    hi,mi=header(row['he']),header(row['mif']);check_scale(hi,meta['he_mpp']);check_scale(mi,meta['he_mpp'])
    panel=meta['panel'];labels=mi['markers']
    if len(labels)!=len(panel) or len(set(labels))!=len(labels) or set(labels)!=set(panel):raise ValueError('mIF panel mismatch')
    order=[labels.index(k) for k in panel]
    from image_io import TiffRegionReader
    with ExitStack() as stack:
        he=TiffRegionReader(row['he'],3);stack.callback(he.close)
        axes='CYX' if len(mi['shape'])==3 and mi['axes'].endswith('YX') else None
        mif=TiffRegionReader(row['mif'],len(panel),axes);stack.callback(mif.close)
        if he.dtype!=np.uint8 or [he.height,he.width]!=meta['shape'] or (he.height,he.width)!=(mif.height,mif.width):raise ValueError('Registered geometry/dtype mismatch')
        print('Reading H&E ROI plus context margin once',flush=True)
        pixels=he.read(y0-60,y1+60,x0-60,x1+60).copy()
        print('Reading measured mIF ROI once; averaging 8x8 blocks',flush=True)
        raw=mif.read(y0,y1,x0,x1)
        native_maps={marker:raw[:,:,order[panel.index(marker)]].copy() for marker in a.markers}
        xy_check,target=block_targets(raw,np.ones(raw.shape[:2],bool),order,x0,y0)
        if not np.array_equal(xy_check,xy):raise ValueError('Target grid mismatch')
        del raw
    if (target<0).any() or (target>a.intensity_max).any():raise ValueError('Targets outside declared intensity ceiling')
    import torch,timm
    from PIL import Image
    from torchvision import transforms
    from train import make_head
    digest=hashlib.sha256()
    with open(a.checkpoint,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=meta['checkpoint_sha256']:raise ValueError('UNI2-h checkpoint SHA differs from cached encoder')
    print('Loading frozen UNI2-h and original/weighted held-out MLP heads',flush=True)
    encoder=timm.create_model('vit_giant_patch14_224',pretrained=False,img_size=224,patch_size=14,depth=24,num_heads=24,init_values=1e-5,embed_dim=1536,mlp_ratio=2.66667*2,num_classes=0,no_embed_class=True,mlp_layer=timm.layers.SwiGLUPacked,act_layer=torch.nn.SiLU,reg_tokens=8,dynamic_img_size=True)
    encoder.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
    encoder.requires_grad_(False).eval().to(a.device)
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    models=[]
    for label,(_,path) in zip(('original_mlp','weighted_mlp'),heads):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['panel']!=panel or any(meta[k]!=v for k,v in ckpt['cache_signature'].items()):raise ValueError('Head provenance mismatch')
        if meta['patient_id'] not in ckpt['split']['test'] or meta['patient_id'] in ckpt['split']['train']+ckpt['split']['val']:raise ValueError('Head is not held out for this patient')
        if ckpt['hidden']==0:raise ValueError('Expected MLP heads for both models')
        model=make_head(torch,len(panel),ckpt['hidden']);model.load_state_dict(ckpt['state_dict']);model.requires_grad_(False).eval().to(a.device)
        models.append((model,ckpt['scaler']))
    predictions=[np.empty_like(target),np.empty_like(target)];inference_start=time.monotonic()
    for offset in range(0,len(xy),a.batch_size):
        indices=np.arange(offset,min(offset+a.batch_size,len(xy)))
        # With a 60px margin, target block (iy,ix) uses pixels[8*iy:8*iy+128,...].
        batch=torch.stack([transform(Image.fromarray(pixels[(k//nx)*8:(k//nx)*8+128,(k%nx)*8:(k%nx)*8+128])) for k in indices]).to(a.device)
        with torch.inference_mode():
            features=encoder(batch)
            if features.shape!=(len(indices),1536) or not torch.isfinite(features).all():raise ValueError('Invalid encoder features')
            for dest,(model,state) in zip(predictions,models):dest[indices]=scale(model(features).cpu().numpy(),state,inverse=True)
        done=int(indices[-1])+1
        if offset==0 or (offset//a.batch_size)%50==0 or done==len(xy):
            elapsed=time.monotonic()-inference_start;rate=done/elapsed
            print(json.dumps({'stage':'inference','done':done,'total':len(xy),'locations_per_second':rate,'estimated_remaining_minutes':(len(xy)-done)/rate/60}),flush=True)
    if not all(np.isfinite(v).all() for v in predictions):raise ValueError('Nonfinite predictions')
    np.savez(out/'measured_mif_native_roi.npz',**native_maps)
    np.savez(out/'dense_predictions.npz',xy=xy,target=target,original_mlp=predictions[0],weighted_mlp=predictions[1])
    he_roi=pixels[60:60+y1-y0,60:60+x1-x0];Image.fromarray(he_roi).save(out/'he_roi.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from metrics import metrics
    reports={name:metrics(target/a.intensity_max,pred/a.intensity_max,np.full(len(xy),meta['patient_id']),panel) for name,pred in zip(('original_mlp','weighted_mlp'),predictions)}
    write_json(out/'roi_metrics_normalized.json',reports)
    cards=[];settings=[];extent=(x0,x1,y1,y0)
    for marker in a.markers:
        j=panel.index(marker);maps=[v[:,j].reshape(ny,nx)/a.intensity_max for v in (target,*predictions)]
        combined=np.concatenate([v.ravel() for v in maps]);low,high=map(float,np.percentile(combined,[1,99]))
        if high<=low:high=low+1e-6
        fig,axes=plt.subplots(1,5,figsize=(19,8),constrained_layout=True)
        axes[0].imshow(he_roi,extent=extent);axes[0].set_title('Registered H&E')
        axes[1].imshow(native_maps[marker]/a.intensity_max,extent=extent,vmin=low,vmax=high,cmap='viridis',interpolation='nearest');axes[1].set_title('Measured mIF (native pixels)')
        for ax,v,label in zip(axes[2:],maps,('Measured mIF (8x8 means)','Original MLP — held-out','Weighted MLP — held-out')):
            im=ax.imshow(v,extent=extent,vmin=low,vmax=high,cmap='viridis',interpolation='nearest');ax.set_title(label)
        for ax in axes:ax.set_xlabel('Native x');ax.set_ylabel('Native y')
        fig.colorbar(im,ax=list(axes[1:]),label=f'Intensity / {a.intensity_max:g}; shared display scale',shrink=.5)
        fig.suptitle(f'{a.sample} | {marker} | stride 8, contiguous 8x8 block predictions')
        name=f'marker_{panel.index(marker):02d}.png';fig.savefig(out/name,dpi=160);plt.close(fig)
        cards.append(f'<h2>{html.escape(marker)}</h2><img style="max-width:100%" src="{name}">')
        settings.append({'marker':marker,'low':low,'high':high})
    write_json(out/'display_settings.json',settings)
    after=source_stats(row);write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':before==after})
    if before!=after:raise ValueError('Source metadata changed; no complete manifest written')
    write_json(out/'manifest.json',{'complete':True,'sample_id':a.sample,'patient_id':meta['patient_id'],'panel':panel,'roi':a.roi,'grid_shape':[ny,nx],'stride':8,'target_block':8,'context':128,'locations':len(xy),'grid_origin':'ROI upper-left; no rounding to training grid','support':'Explicit all-valid assumption; no tissue filtering applied','head_labels':['original_mlp','weighted_mlp'],'heads':[str(v[1].resolve()) for v in heads],'checkpoint_sha256':digest.hexdigest(),'encoder_revision':meta['revision'],'elapsed_seconds':time.monotonic()-start})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Dense held-out ROI review</h1><p>Each map pixel is an 8x8 mean, not native-pixel or cell-level expression. Native measured mIF is shown alongside its block-averaged version on the prediction grid. All ROI blocks included, without tissue filtering. Shared percentile display clipping only; numeric predictions are unclipped. Context uses a 60-pixel surrounding margin.</p>'+''.join(cards))
    event(out,'complete',elapsed_seconds=time.monotonic()-start)
    print(f'Completed: {out}/index.html',flush=True)

if __name__=='__main__':main()
