"""Read existing dense ROI caches and write NEW figures with one fixed color scale."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_reference(root):
    meta=json.loads((root/'manifest.json').read_text())
    inv=json.loads((root/'invocation.json').read_text())['args']
    if not meta.get('complete') or meta.get('stride')!=8 or meta.get('target_block')!=8:
        raise ValueError(f'{root}: need complete stride-8, block-8 cache')
    if meta.get('target_units') not in (None, 'raw', 'raw_intensity'):
        raise ValueError('This renderer supports original raw-intensity dense caches only')
    scale=float(inv['intensity_max'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid recorded display divisor')
    return meta,scale


def clipping(values, low, high):
    return {'below_vmin_fraction':float(np.mean(values<low)),
            'above_vmax_fraction':float(np.mean(values>high)),
            'minimum':float(values.min()),'maximum':float(values.max())}


def render(root, out, markers, divisor, low, high, dpi):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    meta,recorded=load_reference(root)
    # Existing generators save raw arrays; recorded intensity_max is a DISPLAY divisor.
    panel=meta['panel']
    if len(panel)!=len(set(panel)):raise ValueError('Duplicate panel labels')
    if set(markers)-set(panel):raise ValueError(f'Unknown markers: {set(markers)-set(panel)}')
    x0,y0,x1,y1=meta['roi'];ny,nx=meta['grid_shape']
    if (ny*8,nx*8)!=(y1-y0,x1-x0):raise ValueError('ROI/grid mismatch')
    he=np.asarray(Image.open(root/'he_roi.png').convert('RGB'))
    if he.shape!=(ny*8,nx*8,3):raise ValueError('H&E shape mismatch')
    extent=(x0,x1,y1,y0);norm=Normalize(vmin=low,vmax=high,clip=False)
    stats=[];cards=[]
    with np.load(root/'dense_predictions.npz',allow_pickle=False) as z, np.load(root/'measured_mif_native_roi.npz',allow_pickle=False) as native:
        heads=meta.get('head_labels') or [k for k in z.files if k not in ('xy','target')]
        if not heads or any(k not in z.files for k in heads):raise ValueError('Missing prediction heads')
        xx,yy=np.meshgrid(np.arange(x0+4,x1,8),np.arange(y0+4,y1,8))
        if not np.array_equal(z['xy'],np.column_stack((xx.ravel(),yy.ravel()))):raise ValueError('Coordinate order mismatch')
        arrays={k:z[k] for k in ['target',*heads]}
        if any(v.shape!=(ny*nx,len(panel)) or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid cached arrays')
        out.mkdir(parents=True,exist_ok=False)
        for marker in markers:
            j=panel.index(marker)
            values=[arrays[k][:,j].reshape(ny,nx)/divisor for k in ['target',*heads]]
            titles=['Measured mIF (8×8 means)',*[k.replace('_',' ')+' — held-out head' for k in heads]]
            native_present=marker in native.files
            if native_present:
                pixels=native[marker]
                if pixels.shape!=(ny*8,nx*8) or not np.isfinite(pixels).all():raise ValueError('Invalid native pixels')
                if not np.allclose(pixels.reshape(ny,8,nx,8).mean((1,3),dtype=np.float64),arrays['target'][:,j].reshape(ny,nx),rtol=1e-6,atol=1e-5):
                    raise ValueError('Native pixels disagree with cached block targets')
                values.insert(0,pixels/divisor);titles.insert(0,'Measured mIF (native pixels)')
            fig,axes=plt.subplots(1,1+len(values),figsize=(4*(1+len(values)),8),constrained_layout=True)
            axes[0].imshow(he,extent=extent);axes[0].set_title('Registered H&E')
            for ax,v,title in zip(axes[1:],values,titles):
                im=ax.imshow(v,extent=extent,norm=norm,cmap='viridis',interpolation='nearest');ax.set_title(title)
                stats.append({'marker':marker,'map':title,**clipping(v,low,high)})
            for ax in axes:ax.set_xlabel('Native x');ax.set_ylabel('Native y')
            fig.colorbar(im,ax=list(axes[1:]),label=f'Intensity / {divisor:g} — fixed across ALL markers',shrink=.55,extend='both')
            fig.suptitle(f'{meta["sample_id"]} | {marker} | fixed limits [{low:g}, {high:g}]')
            name=f'marker_{j:02d}.png';fig.savefig(out/name,dpi=dpi);plt.close(fig)
            note='' if native_present else '<p>Native pixel map unavailable in this cache; measured 8×8 means are shown.</p>'
            cards.append(f'<h2>{html.escape(marker)}</h2>{note}<img style="max-width:100%" src="{name}">')
    write_json(out/'display_statistics.json',stats)
    write_json(out/'manifest.json',{'complete':True,'source_dense_root':str(root),'sample_id':meta['sample_id'],
        'roi':meta['roi'],'markers':markers,'heads':heads,'divisor':divisor,'vmin':low,'vmax':high,
        'original_display_divisor':recorded,'numeric_predictions_changed':False,'inference_rerun':False,
        'units':'original measured intensity divided by fixed divisor; NOT per-slide P99 normalization'})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed-scale dense ROI</h1>'
        f'<p>All maps use intensity / {divisor:g}, limits [{low:g}, {high:g}]. '
        'Colors saturate outside these limits; numeric arrays remain unchanged. '
        'Block maps represent 8×8 means, not individual cells.</p>'+''.join(cards))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--dense-roots',nargs='+');g.add_argument('--scan-root',help='Discover completed direct-child dense caches')
    p.add_argument('--out',required=True);p.add_argument('--markers',nargs='+',default=['CAV','CD8','CD138','COLIV','POSTN','VIM'])
    p.add_argument('--divisor',type=float,default=255);p.add_argument('--vmin',type=float,default=0);p.add_argument('--vmax',type=float,default=1)
    p.add_argument('--dpi',type=int,default=160);a=p.parse_args()
    if not all(np.isfinite(v) for v in (a.divisor,a.vmin,a.vmax)) or a.divisor<=0 or a.vmax<=a.vmin or a.dpi<=0:
        raise ValueError('Invalid scale or dpi')
    if len(a.markers)!=len(set(a.markers)):raise ValueError('Duplicate markers')
    roots=[Path(r).resolve() for r in a.dense_roots] if a.dense_roots else sorted(p.parent.resolve() for p in Path(a.scan_root).glob('*/dense_predictions.npz'))
    if not roots or len(roots)!=len(set(roots)):raise ValueError('No caches or duplicate roots')
    # Preflight all references; incomplete discovered caches are reported, never treated as success.
    skipped=[];ready=[]
    for r in roots:
        try:
            load_reference(r)
            for name in ('he_roi.png','measured_mif_native_roi.npz'):
                if not (r/name).is_file():raise ValueError(f'Missing {name}')
            ready.append(r)
        except (OSError,ValueError,KeyError) as exc:
            if a.dense_roots:raise
            skipped.append({'source':str(r),'reason':str(exc)})
    if not ready:raise ValueError(f'No usable caches: {skipped}')
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('Use a NEW output directory: '+str(out))
    if any(out==r or r in out.parents or out in r.parents for r in roots):raise ValueError('Output overlaps inputs')
    out.mkdir(parents=True)
    write_json(out/'invocation.json',vars(a));write_json(out/'skipped.json',skipped)
    (out/'render_source.py').write_bytes(Path(__file__).read_bytes())
    links=[]
    for i,r in enumerate(ready):
        name=f'{i:02d}_{r.name}'
        print(f'Rendering {r} -> {out/name}',flush=True)
        render(r,out/name,a.markers,a.divisor,a.vmin,a.vmax,a.dpi)
        links.append(f'<li><a href="{html.escape(name)}/index.html">{html.escape(r.name)}</a></li>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Shared fixed scale across markers and runs</h1><ul>'+''.join(links)+'</ul>')
    write_json(out/'manifest.json',{'complete':True,'sources':[str(r) for r in ready],'skipped':skipped,
        'divisor':a.divisor,'vmin':a.vmin,'vmax':a.vmax,'markers':a.markers,
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(f'Gallery: {out}/index.html',flush=True)

if __name__=='__main__':main()
