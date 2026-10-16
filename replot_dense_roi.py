"""Render extra dense ROI markers from saved predictions; no encoder/head inference."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from audit import snapshot,write_json
from prepare import discover,header
from raw_safety import guard_output,source_stats


def read_channel(reader,channel,roi):
    x0,y0,x1,y1=roi
    selectors=[slice(y0,y1) if k=='Y' else slice(x0,x1) if k=='X' else channel if k==reader.channel_axis else 0 for k in reader.axes]
    pixels=np.asarray(reader.array[tuple(selectors)])
    remaining=''.join(k for k in reader.axes if k in 'YX')
    return pixels.transpose([remaining.index('Y'),remaining.index('X')])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['CAV'])
    a=p.parse_args();root=Path(a.dense_root).resolve();out=Path(a.out).resolve()
    meta=json.loads((root/'manifest.json').read_text());inv=json.loads((root/'invocation.json').read_text())['args']
    if not meta.get('complete') or meta['stride']!=8 or meta['target_block']!=8:raise ValueError('Expected completed stride-8 dense run')
    if out==root or root in out.parents or out in root.parents:raise ValueError('Use an independent new output root')
    if out.exists():raise FileExistsError(out)
    if len(set(a.markers))!=len(a.markers) or set(a.markers)-set(meta['panel']):raise ValueError('Unknown/duplicate marker')
    divisor=inv['intensity_max']
    if not np.isfinite(divisor) or divisor<=0:raise ValueError('Invalid original intensity scale')
    x0,y0,x1,y1=meta['roi'];ny,nx=meta['grid_shape']
    if (ny*8,nx*8)!=(y1-y0,x1-x0):raise ValueError('ROI/grid mismatch')
    native={}
    with np.load(root/'measured_mif_native_roi.npz',allow_pickle=False) as z:
        for marker in a.markers:
            if marker in z:native[marker]=z[marker].copy()
    missing=set(a.markers)-set(native);row=None;before=None
    if missing:
        matches=[r for r in discover(inv['base_dir']) if r['sample_id']==meta['sample_id']]
        if len(matches)!=1 or matches[0]['error']:raise ValueError('Missing/ambiguous original pair')
        row=matches[0];guard_output(out,[row]);before=source_stats(row)
        original_check=json.loads((root/'raw_input_check.json').read_text())
        if before!=original_check['after']:raise ValueError('Raw source metadata differs from completed dense run')
        info=header(row['mif']);labels=info['markers']
        if len(labels)!=len(set(labels)) or set(labels)!=set(meta['panel']):raise ValueError('Source marker labels differ')
        from image_io import TiffRegionReader
        axes='CYX' if len(info['shape'])==3 and info['axes'].endswith('YX') else None
        reader=TiffRegionReader(row['mif'],len(meta['panel']),axes)
        try:
            for marker in sorted(missing):
                print(f'Reading native {marker} ROI only (no GPU)',flush=True)
                native[marker]=read_channel(reader,labels.index(marker),meta['roi'])
        finally:reader.close()
    from PIL import Image
    he=np.asarray(Image.open(root/'he_roi.png').convert('RGB'))
    if he.shape!=(y1-y0,x1-x0,3):raise ValueError('H&E ROI size mismatch')
    out.mkdir(parents=True);snapshot(out,vars(a))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cards=[];settings=[];extent=(x0,x1,y1,y0)
    with np.load(root/'dense_predictions.npz',allow_pickle=False) as z:
        xy=z['xy'];xx,yy=np.meshgrid(np.arange(x0+4,x1,8),np.arange(y0+4,y1,8))
        if not np.array_equal(xy,np.column_stack((xx.ravel(),yy.ravel()))):raise ValueError('Saved coordinate grid differs')
        arrays=[z[k] for k in ('target','linear','mlp')]
        if any(v.shape!=(ny*nx,len(meta['panel'])) or not np.isfinite(v).all() for v in arrays):raise ValueError('Invalid saved dense arrays')
        for marker in a.markers:
            j=meta['panel'].index(marker);maps=[v[:,j].reshape(ny,nx)/divisor for v in arrays]
            pixels=native[marker]
            if pixels.shape!=(ny*8,nx*8) or not np.isfinite(pixels).all():raise ValueError('Invalid native mIF ROI')
            means=pixels.reshape(ny,8,nx,8).mean((1,3),dtype=np.float64)
            if not np.allclose(means,arrays[0][:,j].reshape(ny,nx),rtol=1e-6,atol=1e-5):raise ValueError('Native mIF means differ from saved ground truth')
            low,high=map(float,np.percentile(np.concatenate([v.ravel() for v in maps]),[1,99]))
            if high<=low:high=low+1e-6
            fig,axes=plt.subplots(1,5,figsize=(19,8),constrained_layout=True)
            axes[0].imshow(he,extent=extent);axes[0].set_title('Registered H&E')
            for ax,v,title in zip(axes[1:],(pixels/divisor,*maps),('Measured mIF (native pixels)','Measured mIF (8x8 means)','Linear — held-out head','MLP — held-out head')):
                im=ax.imshow(v,extent=extent,vmin=low,vmax=high,cmap='viridis',interpolation='nearest');ax.set_title(title)
            for ax in axes:ax.set_xlabel('Native x');ax.set_ylabel('Native y')
            fig.colorbar(im,ax=list(axes[1:]),label=f'Intensity / {divisor:g}; shared display scale',shrink=.5)
            fig.suptitle(f'{meta["sample_id"]} | {marker} | cached stride-8 predictions')
            filename=f'marker_{j:02d}.png';fig.savefig(out/filename,dpi=160);plt.close(fig)
            cards.append(f'<h2>{html.escape(marker)}</h2><img style="max-width:100%" src="{filename}">')
            settings.append({'marker':marker,'low':low,'high':high,'intensity_max':divisor})
    if row is not None:
        after=source_stats(row)
        if before!=after:raise ValueError('Raw metadata changed during rendering')
        write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':True})
    write_json(out/'display_settings.json',settings)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Additional dense ROI markers</h1><p>Saved predictions reused without inference. Each predicted pixel represents an 8x8 mean. Percentile clipping affects display only.</p>'+''.join(cards))
    write_json(out/'manifest.json',{'complete':True,'source_dense_root':str(root),'sample_id':meta['sample_id'],'markers':a.markers,'roi':meta['roi'],'encoder_rerun':False})
    print(f'Open {out}/index.html',flush=True)

if __name__=='__main__':main()
