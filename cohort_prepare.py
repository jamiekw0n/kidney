"""Auditable H&E tissue selection across registered patients, with bounded sampled caches."""
import argparse
import csv
import json
import time
import faulthandler
import traceback
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from prepare import discover,header,check_scale,patient_map
from image_io import TiffRegionReader,CachedRegionReader
from audit import write_json,event,snapshot
from raw_safety import guard_output,source_stats


def progress(root, sample, stage, **values):
    record={'sample_id':sample,'stage':stage,**values}
    print(json.dumps(record,allow_nan=False),flush=True)
    if root.exists():
        write_json(root/'progress.json',record)


def scan_centers(width,height,stride,limit=0):
    xs=range(68,width-64,stride); ys=range(68,height-64,stride)
    total=len(xs)*len(ys)
    if limit and total>limit:
        indices=np.unique(np.linspace(0,total-1,limit,dtype=np.int64))
    else:
        indices=range(total)
    for index in indices:
        yield xs[int(index)%len(xs)],ys[int(index)//len(xs)]


def storage_info(reader):
    return {'shape':list(reader.array.shape),'chunks':list(reader.array.chunks),
            'axes':reader.axes,'dtype':str(reader.dtype)}


def tissue_mask(rgb,bright=235,dark=20,chroma=15):
    v=rgb.astype(np.float32)
    return (v.mean(-1)>dark)&(v.mean(-1)<bright)&((v.max(-1)-v.min(-1))>chroma)


def eligible(rgb,mask,context_min,center_min,bright,dark,chroma):
    tissue=tissue_mask(rgb,bright,dark,chroma)
    context=float(tissue.mean()); center=float(tissue[60:68,60:68].mean())
    if not mask.all():
        return 'invalid_support',context,center
    if context<context_min:
        return 'low_context_tissue',context,center
    if center<center_min:
        return 'low_center_tissue',context,center
    return 'eligible',context,center


def reservoir_add(items,item,seen,cap,rng):
    if len(items)<cap:
        items.append(item)
    else:
        j=int(rng.integers(seen))
        if j<cap:
            items[j]=item


def prepare(row,a,panel,patient,index):
    if row['error']:
        raise ValueError(row['error'])
    root=Path(a.out)/row['sample_id']; root.mkdir()
    started=time.monotonic()
    progress(root,row['sample_id'],'he_header_start')
    hi=header(row['he'])
    progress(root,row['sample_id'],'mif_header_start',elapsed_seconds=time.monotonic()-started)
    mi=header(row['mif'])
    progress(root,row['sample_id'],'headers_complete',elapsed_seconds=time.monotonic()-started)
    check_scale(hi,a.mpp); check_scale(mi,a.mpp)
    source=mi['markers']
    if len(source)!=len(panel) or len(set(source))!=len(source) or set(source)!=set(panel):
        raise ValueError('Embedded QPTIFF labels do not match panel')
    order=[source.index(marker) for marker in panel]
    mif_axes='CYX' if len(mi['shape'])==3 and mi['axes'].endswith('YX') else None
    rng=np.random.default_rng(np.random.SeedSequence([a.seed,a.seed_index_offset+index]))
    counts={key:0 for key in ('candidates','invalid_support','low_context_tissue','low_center_tissue','eligible')}
    selected=[]; rejected_previews=[]; preview_counts={}
    with ExitStack() as stack:
        progress(root,row['sample_id'],'he_reader_open_start')
        he=TiffRegionReader(row['he'],3); stack.callback(he.close)
        progress(root,row['sample_id'],'he_reader_open_complete',storage=storage_info(he),elapsed_seconds=time.monotonic()-started)
        progress(root,row['sample_id'],'mif_reader_open_start')
        mif=TiffRegionReader(row['mif'],len(panel),mif_axes); stack.callback(mif.close)
        progress(root,row['sample_id'],'mif_reader_open_complete',storage=storage_info(mif),elapsed_seconds=time.monotonic()-started)
        if he.dtype!=np.uint8 or (he.height,he.width)!=(mif.height,mif.width):
            raise ValueError('Need uint8 H&E and matching registered dimensions')
        raw_bytes=he.height*he.width*3*he.dtype.itemsize
        if a.he_cache_gb and tuple(he.array.chunks)==tuple(he.array.shape) and raw_bytes<=a.he_cache_gb*1024**3:
            progress(root,row['sample_id'],'he_cache_decode_start',raw_bytes=raw_bytes)
            before=time.monotonic(); he=CachedRegionReader(he)
            progress(root,row['sample_id'],'he_cache_decode_complete',seconds=time.monotonic()-before)
        support=None
        if a.valid_mask_name:
            support=TiffRegionReader(str(Path(row['folder'])/a.valid_mask_name),None)
            stack.callback(support.close)
            if (support.height,support.width)!=(he.height,he.width):
                raise ValueError('Support mask dimensions differ')
        # Center 68+stride*k preserves target origin 4+8*k and full 128 context.
        with (root/'candidate_decisions.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['x','y','decision','context_tissue_fraction','center_tissue_fraction'])
            f.flush()
            scan_total=len(range(68,he.width-64,a.stride))*len(range(68,he.height-64,a.stride))
            timings={'he_read_seconds':0.0,'support_read_seconds':0.0,'target_read_seconds':0.0}
            progress(root,row['sample_id'],'scan_start',full_grid_candidates=scan_total,diagnostic_limit=a.diagnostic_max_candidates)
            for x,y in scan_centers(he.width,he.height,a.stride,a.diagnostic_max_candidates):
                if counts['candidates']==0:
                    progress(root,row['sample_id'],'first_he_read_start',xy=[x,y])
                before=time.monotonic()
                rgb=he.read(y-64,y+64,x-64,x+64)
                if rgb.shape!=(128,128,3) or rgb.dtype!=np.uint8:
                    raise ValueError(f'Unexpected H&E read: {rgb.shape}, {rgb.dtype}')
                duration=time.monotonic()-before; timings['he_read_seconds']+=duration
                if counts['candidates']==0 or duration>10:
                    progress(root,row['sample_id'],'he_read_complete',xy=[x,y],read_seconds=duration,returned_shape=list(rgb.shape))
                before=time.monotonic()
                valid=support.read(y-64,y+64,x-64,x+64)!=0 if support else np.ones((128,128),bool)
                timings['support_read_seconds']+=time.monotonic()-before
                reason,context,center=eligible(rgb,valid,a.min_context,a.min_center,a.bright,a.dark,a.chroma)
                counts['candidates']+=1; counts[reason]+=1
                writer.writerow([x,y,reason,context,center]); f.flush()
                if counts['candidates']==1 or counts['candidates']%a.progress_every==0:
                    progress(root,row['sample_id'],'scan_progress',counts=counts.copy(),timings=timings.copy(),elapsed_seconds=time.monotonic()-started)
                if reason!='eligible' and preview_counts.get(reason,0)<3:
                    from PIL import Image
                    filename=f'rejected_{reason}_{preview_counts.get(reason,0)}.png'
                    Image.fromarray(rgb).resize((256,256)).save(root/filename)
                    rejected_previews.append({'file':filename,'xy':[x,y],'reason':reason})
                    preview_counts[reason]=preview_counts.get(reason,0)+1
                if reason=='eligible':
                    reservoir_add(selected,(x,y,context,center),counts['eligible'],a.max_per_sample,rng)
        progress(root,row['sample_id'],'scan_complete',counts=counts.copy(),timings=timings.copy(),elapsed_seconds=time.monotonic()-started)
        selected.sort(key=lambda item:(item[1],item[0]))
        if not selected:
            write_json(root/'filtering.json',{'counts':counts,'args':vars(a)})
            raise ValueError('No tissue windows retained; inspect images/thresholds')
        shards=[]; all_xy=[]; preview=[]
        for start in range(0,len(selected),a.shard_size):
            chunk=selected[start:start+a.shard_size]; xy=[]; targets=[]
            for x,y,context,center in chunk:
                if not all_xy and not xy:
                    progress(root,row['sample_id'],'first_mif_target_read_start',xy=[x,y],channels=len(panel))
                before=time.monotonic()
                block=mif.read(y-4,y+4,x-4,x+4)
                duration=time.monotonic()-before; timings['target_read_seconds']+=duration
                if (len(all_xy)+len(xy))%a.progress_every==0 or duration>10:
                    progress(root,row['sample_id'],'mif_target_read_complete',xy=[x,y],read_seconds=duration,returned_shape=list(block.shape),targets_done=len(all_xy)+len(xy)+1)
                if block.shape!=(8,8,len(panel)):
                    raise ValueError(f'Unexpected marker target read: {block.shape}')
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite measured targets')
                xy.append([x,y]); targets.append(block.mean((0,1),dtype=np.float64)[order])
                if len(preview)<12:
                    from PIL import Image
                    image=Image.fromarray(he.read(y-64,y+64,x-64,x+64)).resize((256,256))
                    image.save(root/f'preview_{len(preview):02d}.png')
                    preview.append({'file':f'preview_{len(preview):02d}.png','xy':[x,y]})
            name=f'targets_{len(shards):05d}.npz'; shards.append(name)
            xy=np.asarray(xy,dtype=np.int64); all_xy.extend(xy.tolist())
            np.savez(root/name,xy=xy,measured=np.asarray(targets,dtype=np.float32))
        counts['retained']=len(selected); counts['eligible_not_sampled']=counts['eligible']-len(selected)
        write_json(root/'selected_coordinates.json',all_xy)
        write_json(root/'filtering.json',{'counts':counts,'args':vars(a),'timings':timings,'elapsed_seconds':time.monotonic()-started,'diagnostic_only':bool(a.diagnostic_max_candidates),'preview':preview,'rejected_previews':rejected_previews,
            'boundary_rule':'Only fully in-bounds 128x128 contexts; center grid begins at (68,68)',
            'sampling':'Seeded uniform reservoir over eligible grid centers',
            'support':'explicit full-support assumption' if a.assume_all_valid else a.valid_mask_name,
            'warning':'H&E heuristic is not a compartment or registration classifier; review retained/excluded regions.'})
        import html
        cards=[]
        for item in preview+rejected_previews:
            caption=f'{item.get("reason","retained")} | xy={item["xy"]}'
            cards.append(f'<figure><img width=256 src="{item["file"]}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        (root/'filter_review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Retained and rejected H&amp;E contexts</h1><p>Examples only, not exhaustive QA. Each center target is the central 8x8 pixels.</p><div style="display:flex;flex-wrap:wrap">'+''.join(cards)+'</div>')
        write_json(root/'manifest.json',dict(complete=True,sample_id=row['sample_id'],patient_id=patient,
            he=row['he'],mif=row['mif'],he_axes=None,he_series=0,he_level=0,
            he_mpp=a.mpp,shape=[he.height,he.width],panel=panel,input_patch=128,target_block=8,
            shards=shards,retained=len(selected),source_mif_markers=source,source_to_panel_reorder=order,
            he_metadata=hi,mif_metadata=mi,filtering='filtering.json',selection_config=vars(a),
            diagnostic_only=bool(a.diagnostic_max_candidates),
            selection_policy={key:getattr(a,key) for key in ('stride','max_per_sample','min_context','min_center','bright','dark','chroma','seed')},
            patient_identity_assumption='one sample per patient' if a.samples_as_patients else 'CSV mapping'))
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',required=True); p.add_argument('--panel',required=True); p.add_argument('--out',required=True)
    p.add_argument('--mpp',required=True,type=float)
    ids=p.add_mutually_exclusive_group(required=True)
    ids.add_argument('--samples-as-patients',action='store_true'); ids.add_argument('--patient-map')
    sup=p.add_mutually_exclusive_group(required=True)
    sup.add_argument('--assume-all-valid',action='store_true'); sup.add_argument('--valid-mask-name')
    p.add_argument('--samples',nargs='+'); p.add_argument('--stride',type=int,default=128)
    p.add_argument('--max-per-sample',type=int,default=5000)
    p.add_argument('--min-context',type=float,default=0.5); p.add_argument('--min-center',type=float,default=0.5)
    p.add_argument('--bright',type=float,default=235); p.add_argument('--dark',type=float,default=20)
    p.add_argument('--chroma',type=float,default=15); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seed-index-offset',type=int,default=0,help='Preserve sample seed index when launching one sample per task')
    p.add_argument('--shard-size',type=int,default=1000)
    p.add_argument('--he-cache-gb',type=float,default=0,help='Decode single-chunk H&E once within GiB budget; 0 disables')
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--diagnostic-max-candidates',type=int,default=0,help='0=full scan; positive limit samples spread-out grid locations for diagnostics only')
    p.add_argument('--traceback-after',type=int,default=120,help='Repeated stack dumps during slow stages; 0 disables')
    a=p.parse_args()
    if a.seed_index_offset<0 or a.he_cache_gb<0 or a.progress_every<1 or a.diagnostic_max_candidates<0 or a.traceback_after<0:
        p.error('Invalid progress/diagnostic/traceback settings')
    if a.traceback_after:
        faulthandler.dump_traceback_later(a.traceback_after,repeat=True)
    if a.stride<8 or a.stride%8 or a.max_per_sample<1 or a.shard_size<1 or a.mpp<=0 or a.seed<0:
        p.error('Invalid positive stride/cap/shard/scale or seed; stride must be multiple of 8')
    if not 0<=a.min_context<=1 or not 0<=a.min_center<=1 or not 0<=a.dark<a.bright<=255 or not 0<=a.chroma<=255:
        p.error('Invalid tissue thresholds')
    panel=json.loads(Path(a.panel).read_text())
    if not isinstance(panel,list) or not panel or len(set(panel))!=len(panel) or any(not isinstance(s,str) or not s for s in panel):
        p.error('Panel must be unique nonempty marker names')
    rows=discover(a.base_dir)
    if a.samples:
        if set(a.samples)-{r['sample_id'] for r in rows}: p.error('Requested sample absent')
        rows=[r for r in rows if r['sample_id'] in a.samples]
    if not rows: p.error('No eligible folders')
    out=Path(a.out)
    guard_output(out,discover(a.base_dir))
    if out.exists(): raise FileExistsError('Use a new cohort output directory')
    out.mkdir(parents=True); snapshot(out,vars(a)); write_json(out/'discovery.json',rows)
    mapping=patient_map(a.patient_map) if a.patient_map else {r['sample_id']:r['sample_id'] for r in rows}
    results=[]
    for i,row in enumerate(rows):
        event(out,'sample_start',sample_id=row['sample_id'])
        try:
            if row['sample_id'] not in mapping: raise ValueError('Missing patient mapping')
            before=source_stats(row)
            counts=prepare(row,a,panel,mapping[row['sample_id']],i)
            after=source_stats(row)
            unchanged=before==after
            write_json(out/row['sample_id']/'raw_input_check.json',{'before':before,'after':after,'metadata_unchanged':unchanged,'check':'file size and mtime; not a content hash'})
            if not unchanged:
                (out/row['sample_id']/'manifest.json').rename(out/row['sample_id']/'invalidated_manifest.json')
                raise ValueError('Raw source metadata changed during processing; complete manifest invalidated')
            record={'sample_id':row['sample_id'],'status':'complete','counts':counts}
        except Exception as exc:
            traceback.print_exc()
            record={'sample_id':row['sample_id'],'status':'excluded_or_failed','error':f'{type(exc).__name__}: {exc}'}
        results.append(record); event(out,'sample_end',**record); print(json.dumps(record),flush=True)
        write_json(out/'cohort_report.json',results)
    write_json(out/'cohort_status.json',{'finished':True,'completed':sum(r['status']=='complete' for r in results),
        'failed_or_excluded':sum(r['status']!='complete' for r in results)})
    if a.traceback_after: faulthandler.cancel_dump_traceback_later()
    return int(any(r['status']!='complete' for r in results))

if __name__=='__main__': raise SystemExit(main())
