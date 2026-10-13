"""Export a portable HTML gallery of matched H&E/mIF ROI candidates; no GUI required."""
import argparse
from contextlib import ExitStack
import html
import json
from pathlib import Path
import numpy as np
from prepare import discover, header
from image_io import TiffRegionReader


def tissue_fraction(rgb):
    """Heuristic excluding near-white background and black padding; not a validity mask."""
    v=rgb.astype(np.float32)
    brightness=v.mean(-1)
    chroma=v.max(-1)-v.min(-1)
    return float(((brightness>20)&(brightness<235)&(chroma>15)).mean())


def candidates(width,height,size,grid):
    if size>min(width,height):
        raise ValueError('ROI larger than image')
    xs=np.unique((np.linspace(0,width-size,grid)//8*8).astype(int))
    ys=np.unique((np.linspace(0,height-size,grid)//8*8).astype(int))
    return [(int(x),int(y),int(x+size),int(y+size)) for y in ys for x in xs]


def display_channel(values):
    values=np.asarray(values,dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite channel values')
    low,high=np.percentile(values,[1,99.5])
    if high<=low:
        return np.zeros(values.shape,np.uint8),(float(low),float(high))
    image=(np.clip((values-low)/(high-low),0,1)*255).astype(np.uint8)
    return image,(float(low),float(high))


def save_png(path,array):
    from PIL import Image
    Image.fromarray(array).save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',required=True)
    p.add_argument('--samples',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['DAPI','CDH1','UMOD','COLIV'])
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--grid',type=int,default=7,help='Probe grid x grid windows per sample')
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--min-tissue',type=float,default=0.25)
    p.add_argument('--roi',type=int,nargs=4,help='Inspect explicit same coordinates in selected samples instead of ranking')
    for modality in ('he','mif'):
        p.add_argument('--'+modality+'-axes')
        p.add_argument('--'+modality+'-series',type=int,default=0)
        p.add_argument('--'+modality+'-level',type=int,default=0)
    a=p.parse_args()
    if a.size<8 or a.size%8 or min(a.grid,a.top_k)<1 or not 0<=a.min_tissue<=1:
        p.error('size must be a positive multiple of 8; grid/top-k positive; min-tissue in [0,1]')
    if len(set(a.markers))!=len(a.markers):
        p.error('Duplicate display markers')
    rows={row['sample_id']:row for row in discover(a.base_dir)}
    if set(a.samples)-set(rows):
        p.error(f'Unknown samples: {set(a.samples)-set(rows)}')
    out=Path(a.out)
    if out.exists():
        raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True)
    records=[]; sections=[]; errors=[]
    for sample in a.samples:
        row=rows[sample]
        try:
            if row['error']:
                raise ValueError(row['error'])
            info=header(row['mif'],a.mif_series,a.mif_level)
            panel=info['markers']
            if not panel or len(panel)!=len(set(panel)) or set(a.markers)-set(panel):
                raise ValueError(f'Need unique embedded labels including {a.markers}; found {panel}')
            axes=a.mif_axes or ('CYX' if len(info['shape'])==3 and info['axes'].endswith('YX') else None)
            with ExitStack() as stack:
                he=TiffRegionReader(row['he'],3,a.he_axes,a.he_series,a.he_level); stack.callback(he.close)
                mif=TiffRegionReader(row['mif'],len(panel),axes,a.mif_series,a.mif_level); stack.callback(mif.close)
                if (he.height,he.width)!=(mif.height,mif.width) or he.dtype!=np.uint8:
                    raise ValueError('Pair dimensions differ or H&E is not uint8 RGB')
                bounds=[tuple(a.roi)] if a.roi else candidates(he.width,he.height,a.size,a.grid)
                scored=[]
                for x0,y0,x1,y1 in bounds:
                    if any(v%8 for v in (x0,y0,x1,y1)) or not (0<=x0<x1<=he.width and 0<=y0<y1<=he.height):
                        raise ValueError('ROI bounds must be multiples of 8 within image')
                    if a.roi and (x1-x0!=a.size or y1-y0!=a.size):
                        raise ValueError('Explicit ROI must match --size in both dimensions')
                    rgb=he.read(y0,y1,x0,x1)
                    score=tissue_fraction(rgb)
                    if a.roi or score>=a.min_tissue:
                        scored.append((score,(x0,y0,x1,y1)))
                scored.sort(key=lambda item:item[0],reverse=True)
                chosen=[]
                for item in scored:
                    # Avoid showing overlapping candidate regions.
                    x,y,x1,y1=item[1]
                    if any(x<b[2] and x1>b[0] and y<b[3] and y1>b[1] for _,b in chosen):
                        continue
                    chosen.append(item)
                    if len(chosen)>=a.top_k:
                        break
                cards=[]
                for i,(score,bounds) in enumerate(chosen,1):
                    x0,y0,x1,y1=bounds
                    rgb=he.read(y0,y1,x0,x1)
                    measured=mif.read(y0,y1,x0,x1)
                    prefix=f'{sample}_roi{i}'
                    save_png(out/(prefix+'_he.png'),rgb)
                    panels=[f'<figure><img src="{html.escape(prefix)}_he.png"><figcaption>H&amp;E</figcaption></figure>']
                    ranges={}
                    overlay=None
                    for j,marker in enumerate(a.markers):
                        channel,limits=display_channel(measured[:,:,panel.index(marker)])
                        filename=prefix+f'_channel{j}.png'
                        save_png(out/filename,channel)
                        ranges[marker]=limits
                        panels.append(f'<figure><img src="{html.escape(filename)}"><figcaption>{html.escape(marker)} | display {limits[0]:.2f}–{limits[1]:.2f}</figcaption></figure>')
                        if marker=='DAPI':
                            # Cyan DAPI over grayscale H&E preserves coordinate correspondence.
                            overlay=np.zeros_like(rgb)
                            overlay[:,:,1]=channel; overlay[:,:,2]=channel
                    overlay_html=''
                    if overlay is not None:
                        filename=prefix+'_dapi_cyan.png'; save_png(out/filename,overlay)
                        overlay_html=f'<p>Overlay opacity <input type="range" min="0" max="1" step="0.05" value="0.5" oninput="this.parentElement.nextElementSibling.lastElementChild.style.opacity=this.value"></p><div class="overlay"><img src="{html.escape(prefix)}_he.png"><img style="opacity:0.5" src="{html.escape(filename)}"></div>'
                    command='--samples '+sample+' --roi '+' '.join(map(str,bounds))
                    cards.append(f'<article><h3>ROI {i} · H&amp;E tissue heuristic {score:.0%}</h3><code>{html.escape(command)}</code><div class="panels">'+''.join(panels)+'</div>'+overlay_html+'</article>')
                    records.append(dict(sample_id=sample,roi=list(bounds),tissue_fraction=score,
                        display_ranges=ranges,he=row['he'],mif=row['mif'],mif_mpp_xy=info['mpp_xy'],
                        reviewed=False,common_support_verified=False))
                sections.append(f'<section><h2>{html.escape(sample)}</h2>'+(''.join(cards) or '<p>No candidates passed the heuristic. Lower --min-tissue, increase --grid, or supply --roi.</p>')+'</section>')
                print(f'{sample}: {len(chosen)} ROIs exported',flush=True)
        except Exception as exc:
            errors.append({'sample_id':sample,'error':f'{type(exc).__name__}: {exc}'})
            sections.append(f'<section><h2>{html.escape(sample)}</h2><p>{html.escape(str(exc))}</p></section>')
            print(json.dumps(errors[-1]),flush=True)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Kidney ROI review</title>
<style>body{font-family:system-ui;margin:24px;background:#f4f5f7;color:#18212c}article{background:white;padding:18px;margin:18px 0;border-radius:10px}code{display:block;user-select:all;margin-bottom:12px}.panels{display:flex;gap:12px;overflow:auto}figure{margin:0;flex:0 0 320px}figure img{width:320px;height:320px;object-fit:contain;background:black}figcaption{font-size:13px}.overlay{position:relative;width:512px;max-width:100%;aspect-ratio:1}.overlay img{position:absolute;width:100%;height:100%}section{border-top:2px solid #bbc5d0}</style></head><body>
<h1>Matched kidney ROI review</h1><p>Rows are independent samples; columns show the same coordinates within a sample. Tissue fraction is an H&amp;E heuristic, not a support mask or compartment classifier. The sparse grid may miss useful tissue.</p>
<p>Each mIF channel uses its own ROI percentile display scaling. Compare structures and locations; brightness is not comparable across ROIs. Raw marker data are unchanged. Inspect matching nuclei/tubules and reject padding, tears, missing regions, and misalignment. Cyan overlay shows DAPI; dark regions can reflect biology and do not automatically mean missing coverage.</p>
<p>Candidate coordinates are suggestions, not approved training regions. Record reviewed coordinates and support decisions separately before preparation.</p>'''+''.join(sections)+'</body></html>'
    (out/'index.html').write_text(page)
    (out/'rois.json').write_text(json.dumps({'args':vars(a),'rois':records,'errors':errors},indent=2))
    print(f'Open {out.resolve() / "index.html"}; copy the entire directory to view locally.')
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
