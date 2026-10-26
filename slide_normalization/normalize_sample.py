"""Read-only whole-slide histogram scan; regenerate targets; copy features unchanged."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit import snapshot,write_json,event
from data import read_meta
from normalization import UNITS,recipe,divisors_from_histograms,target_groups
from prepare import header,check_scale
from image_io import TiffRegionReader
from raw_safety import guard_output,source_stats
from io_utils import independent_output
from source_paths import resolve_sources


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def normalize(source,out,percentile=99.,tile_size=512,axes=None,series=0,level=0):
    source=Path(source).resolve();meta=read_meta(source);manifest_hash=sha(source/'manifest.json')
    if meta.get('target_units')==UNITS:raise ValueError('Refuse double normalization')
    if meta.get('encoder')!='UNI2-h' or meta['input_patch']!=128 or meta['target_block']!=8:
        raise ValueError('Expected existing 128-context/8-target UNI2-h cache')
    if tile_size<8 or tile_size%8:raise ValueError('Tile size must be a positive multiple of 8')
    r=recipe(percentile)
    paths,provenance=resolve_sources(source)
    meta={**meta,**paths}
    row={'folder':str(Path(meta['mif']).resolve().parent),'mif':meta['mif'],'he':meta['he']}
    out=independent_output(out,[source,Path(meta['mif']).resolve().parent,Path(meta['he']).resolve().parent])
    guard_output(out,[row]);before=source_stats(row)
    info=header(row['mif'],series,level);check_scale(info,meta['he_mpp'])
    labels=info['markers'];panel=meta['panel']
    if len(labels)!=len(set(labels)) or set(labels)!=set(panel):raise ValueError('Source marker panel mismatch/ambiguity')
    order=[labels.index(m) for m in panel]
    if axes is None:axes='CYX' if len(info['shape'])==3 and info['axes'].endswith('YX') else info['axes']
    out.mkdir(parents=True);snapshot(out,{'source':str(source),'out':str(out),'percentile':percentile,'tile_size':tile_size,'axes':axes,'series':series,'level':level})
    started=time.monotonic();source_records=[];coordinates=[];previous=[]
    for name in meta['shards']:
        path=source/name
        if path.resolve().parent!=source:raise ValueError('Unsafe source shard path')
        with np.load(path,allow_pickle=False) as z:coordinates.append(z['xy'].copy());previous.append(z['target'].copy())
        source_records.append({'name':name,'sha256':sha(path)})
    xy=np.concatenate(coordinates);old=np.concatenate(previous)
    if old.shape!=(len(xy),len(panel)):raise ValueError('Cached target dimensions differ')
    with ExitStack() as stack:
        reader=TiffRegionReader(row['mif'],len(panel),axes,series,level);stack.callback(reader.close)
        if [reader.height,reader.width]!=meta['shape']:raise ValueError('MIF/cache geometry mismatch')
        if reader.dtype not in (np.dtype('uint8'),np.dtype('uint16')):raise ValueError('Exact histogram scan currently supports uint8/uint16 only')
        chunks=getattr(reader.array,'chunks',None)
        if chunks:
            pixels_per_chunk=chunks[reader.axes.index('Y')]*chunks[reader.axes.index('X')]
            if pixels_per_chunk*len(panel)*reader.dtype.itemsize>1024**3:
                raise ValueError('MIF storage chunks imply >1GiB decode per region; use a reviewed bounded reader/derived tiled copy, never rewrite originals')
        levels=np.iinfo(reader.dtype).max+1
        hist=np.zeros((len(panel),levels),dtype=np.int64)
        total=((reader.height+tile_size-1)//tile_size)*((reader.width+tile_size-1)//tile_size)
        done=0
        print(json.dumps({'sample':meta['sample_id'],'stage':'histogram_start','pixels':reader.height*reader.width,'channels':len(panel),'tiles':total}),flush=True)
        for y in range(0,reader.height,tile_size):
            for x in range(0,reader.width,tile_size):
                block=reader.read(y,min(y+tile_size,reader.height),x,min(x+tile_size,reader.width))
                for j,index in enumerate(order):hist[j]+=np.bincount(block[:,:,index].ravel().astype(np.int64),minlength=levels)
                done+=1
                if done==1 or done%25==0 or done==total:
                    print(json.dumps({'sample':meta['sample_id'],'stage':'histogram_progress','done':done,'total':total,'seconds':time.monotonic()-started}),flush=True)
        if not np.all(hist.sum(1)==reader.height*reader.width):raise ValueError('Histogram pixel count mismatch')
        divisors,records=divisors_from_histograms(hist,panel,percentile)
        write_json(out/'normalization.json',{'recipe':r,'markers':records,'source_axes':axes,'series':series,'level':level})
        np.savez_compressed(out/'whole_slide_histograms.npz',histograms=hist,panel=np.asarray(panel))
        def progress(done,total):
            print(json.dumps({'sample':meta['sample_id'],'stage':'target_groups','done':done,'total':total}),flush=True)
        measured,new=target_groups(reader,xy,order,divisors,tile_size,progress)
        if not np.allclose(measured,old,rtol=1e-6,atol=1e-5):
            raise ValueError('Original cached means differ from raw TIFF blocks; refusing normalization of unmatched sources')
    offset=0;checks=[]
    for name,record in zip(meta['shards'],source_records):
        with np.load(source/name,allow_pickle=False) as z:
            features=z['features'].copy();coords=z['xy'].copy();raw_target=z['target'].copy()
        n=len(coords)
        if features.shape!=(n,1536) or not np.isfinite(features).all():raise ValueError('Invalid source features')
        target=new[offset:offset+n]
        np.savez(out/name,features=features,xy=coords,target=target,raw_target=raw_target)
        with np.load(out/name,allow_pickle=False) as z:
            if not np.array_equal(z['features'],features) or not np.array_equal(z['xy'],coords):raise ValueError('Feature/coordinate copy differs')
        offset+=n
        if sha(source/name)!=record['sha256']:raise ValueError('Source cache changed while preparing targets')
        checks.append({'shard':name,'rows':n,'features_and_xy_identical':True})
    if sha(source/'manifest.json')!=manifest_hash:raise ValueError('Source cache manifest changed during normalization')
    after=source_stats(row)
    write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':before==after})
    if before!=after:raise ValueError('Raw source metadata changed; no complete manifest written')
    for record in provenance:
        if sha(record['manifest'])!=record['sha256']:raise ValueError('Source provenance manifest changed during normalization')
    write_json(out/'source_cache.json',{'raw_path_provenance':provenance,'root':str(source),'manifest_sha256':manifest_hash,'shards':source_records,'checks':checks})
    write_json(out/'manifest.json',{**meta,'complete':True,'target_units':UNITS,'normalization_recipe':r,
        'normalization_report':'normalization.json','raw_target_source_cache':str(source),
        'target_definition':'mean of pixel-normalized-and-clipped native central 8x8 region',
        'feature_reextraction':False,'elapsed_seconds':time.monotonic()-started})
    event(out,'complete',sample=meta['sample_id'],rows=len(xy),fallback_markers=[v['marker'] for v in records if v['fallback_dtype_max']])
    print(f'Complete normalized cache: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--percentile',type=float,default=99.);p.add_argument('--tile-size',type=int,default=512)
    p.add_argument('--mif-axes');p.add_argument('--mif-series',type=int,default=0);p.add_argument('--mif-level',type=int,default=0)
    a=p.parse_args()
    if min(a.mif_series,a.mif_level)<0:raise ValueError('Negative image series/level')
    normalize(a.source_cache,a.out,a.percentile,a.tile_size,a.mif_axes,a.mif_series,a.mif_level)

if __name__=='__main__':main()
