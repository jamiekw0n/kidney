"""Compare pixel versus micrometer interpretations of annotation centroids on H&E/DAPI."""
import argparse
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
from audit import snapshot,write_json
from data import read_meta
from prepare import discover,header,check_scale
from raw_safety import guard_output,source_stats


def annotation_files(root,samples):
    result=[]
    for sample in sorted(set(samples)):
        folder=root/sample
        files=sorted(p for p in folder.glob('*') if p.is_file() and p.suffix.lower()=='.csv')
        if len(files)!=1:raise ValueError(f'Need exactly one CSV in {folder}; found {len(files)}')
        result.append(files[0])
    return result


def convert_xy(xy,units,mpp):
    if not np.isfinite(xy).all() or not np.isfinite(mpp) or mpp<=0:raise ValueError('Invalid coordinates/scale')
    if units=='native-pixels':return np.asarray(xy,dtype=float)
    if units=='micrometers':return np.asarray(xy,dtype=float)/mpp
    raise ValueError('Unknown units')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',required=True);p.add_argument('--cache-root',required=True)
    p.add_argument('--annotation-root',required=True);p.add_argument('--sample',default='K2110291')
    p.add_argument('--out',required=True);p.add_argument('--preview-cells',type=int,default=4)
    p.add_argument('--crop-size',type=int,default=128);p.add_argument('--he-cache-gb',type=float,default=8)
    a=p.parse_args()
    if a.preview_cells<1 or a.crop_size<8 or a.crop_size%2 or a.he_cache_gb<0:p.error('Invalid preview/cache settings')
    meta=read_meta(Path(a.cache_root)/a.sample);file=annotation_files(Path(a.annotation_root),[a.sample])[0]
    xy=[];ids=[];size=[];stamp=file.stat()
    with file.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not {'sample_id','cell_id','x','y'}<=set(reader.fieldnames or []):raise ValueError('Missing annotation columns')
        for line,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):raise ValueError(f'Malformed CSV row at {file}:{line}')
            if row['sample_id'].strip()!=a.sample:raise ValueError('Annotation folder/sample_id mismatch')
            xy.append([float(row['x']),float(row['y'])]);ids.append(row['cell_id'])
            if row.get('size'):size.append(float(row['size']))
    after=file.stat()
    if not xy or not np.isfinite(xy).all() or (stamp.st_size,stamp.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Empty/nonfinite/changed annotations')
    xy=np.asarray(xy);hypotheses={u:convert_xy(xy,u,meta['he_mpp']) for u in ('native-pixels','micrometers')}
    rows=[r for r in discover(a.base_dir) if r['sample_id']==a.sample]
    if len(rows)!=1 or rows[0]['error']:raise ValueError('Missing registered pair')
    row=rows[0];out=Path(a.out);guard_output(out,rows)
    if out.exists():raise FileExistsError(out)
    before=source_stats(row);hi,mi=header(row['he']),header(row['mif']);check_scale(hi,meta['he_mpp']);check_scale(mi,meta['he_mpp'])
    if mi['markers'].count('DAPI')!=1:raise ValueError('Need unambiguous DAPI channel')
    from image_io import TiffRegionReader,CachedRegionReader
    from replot_dense_roi import read_channel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True);snapshot(out,vars(a))
    h,w=meta['shape'];half=a.crop_size//2
    report={'sample_id':a.sample,'annotation_file':str(file.resolve()),'rows':len(xy),'shape':[h,w],'mpp':meta['he_mpp'],'coordinate_quantiles':np.percentile(xy,[0,1,50,99,100],axis=0).tolist(),'hypotheses':{},'conclusion':'Unresolved: bounds and overlays are checks, not automatic unit/origin proof'}
    if size:report['size_quantiles']=np.percentile(size,[0,50,100]).tolist()
    selected=np.sort(np.random.default_rng(42).choice(len(xy),min(a.preview_cells,len(xy)),replace=False))
    with ExitStack() as stack:
        he=TiffRegionReader(row['he'],3);stack.callback(he.close)
        axes='CYX' if len(mi['shape'])==3 and mi['axes'].endswith('YX') else None
        mif=TiffRegionReader(row['mif'],len(mi['markers']),axes);stack.callback(mif.close)
        if he.dtype!=np.uint8 or [he.height,he.width]!=[h,w] or (mif.height,mif.width)!=(h,w):raise ValueError('Source geometry/dtype mismatch')
        rawbytes=h*w*3
        if tuple(he.array.chunks)==tuple(he.array.shape) and rawbytes<=a.he_cache_gb*1024**3:
            print('Caching single-chunk H&E once for previews',flush=True);he=CachedRegionReader(he)
        fig,axes=plt.subplots(len(selected),4,figsize=(14,3.5*len(selected)),squeeze=False,constrained_layout=True)
        for k,(unit,positions) in enumerate(hypotheses.items()):
            valid=(positions[:,0]>=0)&(positions[:,0]<w)&(positions[:,1]>=0)&(positions[:,1]<h)
            report['hypotheses'][unit]={'in_bounds':int(valid.sum()),'out_of_bounds':int((~valid).sum()),'coordinate_min':positions.min(0).tolist(),'coordinate_max':positions.max(0).tolist()}
            for i,index in enumerate(selected):
                x,y=positions[index];xx,yy=int(np.floor(x)),int(np.floor(y));left,top,right,bottom=xx-half,yy-half,xx+half,yy+half
                pair=axes[i,k*2:k*2+2]
                for ax,title in zip(pair,('H&E','DAPI')):ax.set_title(f'{unit} | {title}\ncell {ids[index]} → ({x:.1f}, {y:.1f})')
                if not (0<=left<right<=w and 0<=top<bottom<=h):
                    for ax in pair:ax.text(.5,.5,'Context outside image',ha='center',transform=ax.transAxes);ax.axis('off')
                    continue
                rgb=he.read(top,bottom,left,right);dapi=read_channel(mif,mi['markers'].index('DAPI'),[left,top,right,bottom])
                extent=(left,right,bottom,top)
                pair[0].imshow(rgb,extent=extent);pair[1].imshow(dapi,cmap='gray',extent=extent)
                for ax in pair:ax.plot(x,y,'r+',markersize=12);ax.set_xlabel('Native x');ax.set_ylabel('Native y')
        fig.suptitle('Compare annotation coordinate units; red cross is the candidate cell centroid')
        fig.savefig(out/'coordinate_hypotheses.png',dpi=150);plt.close(fig)
    after=source_stats(row)
    if before!=after:raise ValueError('Raw source metadata changed')
    write_json(out/'coordinate_report.json',report);write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':True})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Annotation coordinate QA</h1><p>Floating-point coordinates do not establish units. Inspect hypotheses and verify exporter metadata. Membrane-derived cell centroids need not coincide exactly with nuclear centers. If both interpretations are ambiguous, inspect cell segmentation overlays; do not guess. Same scale does not prove same origin/orientation.</p><img style="max-width:100%" src="coordinate_hypotheses.png">')
    print(json.dumps(report,indent=2));print(f'Open {out}/index.html',flush=True)

if __name__=='__main__':main()
