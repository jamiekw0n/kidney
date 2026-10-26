"""Discover registered pairs and prepare bounded kidney targets without external source code."""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def discover(base):
    base = Path(base)
    if not base.is_dir():
        raise ValueError(f'Base directory does not exist: {base}')
    rows = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name[:1] not in 'BFKMR':
            continue
        files = [f for f in folder.iterdir() if f.is_file()]
        he = sorted(f for f in files if f.suffix.lower() in ('.tif','.tiff') and 'HandE' in f.name)
        mif = sorted(f for f in files if f.suffix.lower() == '.qptiff')
        rows.append({'sample_id':folder.name, 'folder':str(folder.resolve()),
            'he':str(he[0].resolve()) if len(he)==1 else None,
            'mif':str(mif[0].resolve()) if len(mif)==1 else None,
            'he_candidates':[str(f) for f in he], 'mif_candidates':[str(f) for f in mif],
            'error':None if len(he)==len(mif)==1 else f'Expected one HandE TIFF and one QPTIFF; found {len(he)}, {len(mif)}'})
    return rows


def header(path, series=0, level=0):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        selected = tif.series[series].levels[level]
        markers = []
        for page in selected.pages:
            try:
                marker = ET.fromstring(page.description).findtext('.//Biomarker')
            except (ET.ParseError,TypeError):
                marker = None
            if marker:
                markers.append(marker.strip())
        first = selected.pages[0]
        unit = first.tags.get('ResolutionUnit')
        mpp = None
        if unit is not None and int(unit.value) in (2,3):
            length = 25400 if int(unit.value)==2 else 10000
            tags=[first.tags.get(key) for key in ('XResolution','YResolution')]
            if all(tag is not None for tag in tags):
                mpp = [length*tag.value[1]/tag.value[0] for tag in tags]
        return {'axes':selected.axes,'shape':list(selected.shape),'dtype':str(selected.dtype),
                'markers':markers,'mpp_xy':mpp}


def patient_map(path):
    result = {}
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if not {'sample_id','patient_id'} <= set(reader.fieldnames or []):
            raise ValueError('Patient CSV requires sample_id,patient_id columns')
        for row in reader:
            sample, patient=row['sample_id'].strip(),row['patient_id'].strip()
            if not sample or not patient or patient.lower()=='unknown' or sample in result:
                raise ValueError('Empty/unknown patient or duplicate sample mapping')
            result[sample]=patient
    return result


def check_scale(info, expected):
    if info['mpp_xy'] is not None and not np.allclose(info['mpp_xy'],expected,rtol=0.01,atol=0):
        raise ValueError(f'Pixel size argument {expected} disagrees with TIFF metadata {info["mpp_xy"]}')


def block_targets(image, valid, order, x0, y0, white_threshold=None, he=None):
    """Return fully supported 8x8 means and centers for one aligned slab."""
    h,w,c=image.shape
    if h%8 or w%8 or valid.shape!=(h,w):
        raise ValueError('Slabs must contain whole 8x8 blocks and matching validity')
    values=image.reshape(h//8,8,w//8,8,c)
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite marker intensities')
    mean=values.mean((1,3),dtype=np.float64).astype(np.float32)[:,:,order]
    keep=valid.reshape(h//8,8,w//8,8).all((1,3))
    if white_threshold is not None:
        keep &= he.reshape(h//8,8,w//8,8,3).mean((1,3,4)) < white_threshold
    iy,ix=np.nonzero(keep)
    xy=np.column_stack((x0+4+8*ix,y0+4+8*iy)).astype(np.int64)
    return xy,mean[keep]


def prepare_sample(row, a, panel, patients):
    from image_io import TiffRegionReader
    from contextlib import ExitStack
    out=Path(a.out)/row['sample_id']
    if out.exists():
        raise FileExistsError(f'Preserving existing output: {out}')
    if row['sample_id'] not in patients:
        raise ValueError('No verified patient mapping for sample')
    he_info=header(row['he'],a.he_series,a.he_level)
    mif_info=header(row['mif'],a.mif_series,a.mif_level)
    check_scale(he_info,a.mpp); check_scale(mif_info,a.mpp)
    source=mif_info['markers']
    if source:
        if len(source)!=len(panel) or len(set(source))!=len(source) or set(source)!=set(panel):
            raise ValueError(f'QPTIFF marker labels differ from supplied panel: {source}')
        order=[source.index(marker) for marker in panel]
    elif a.confirm_unlabeled_panel_order:
        order=list(range(len(panel)))
    else:
        raise ValueError('No QPTIFF marker labels; require explicit --confirm-unlabeled-panel-order after verification')
    mif_axes=a.mif_axes
    if not mif_axes and source and len(mif_info['shape'])==3 and mif_info['axes'].endswith('YX'):
        mif_axes='CYX'
    maskpath=Path(row['folder'])/a.valid_mask_name if a.valid_mask_name else None
    if maskpath and not maskpath.is_file():
        raise FileNotFoundError(maskpath)
    with ExitStack() as stack:
        he=TiffRegionReader(row['he'],3,a.he_axes,a.he_series,a.he_level)
        stack.callback(he.close)
        mif=TiffRegionReader(row['mif'],len(panel),mif_axes,a.mif_series,a.mif_level)
        stack.callback(mif.close)
        if (he.height,he.width)!=(mif.height,mif.width) or he.dtype!=np.uint8:
            raise ValueError('Pair dimensions must match and H&E must be uint8 RGB')
        mask=None
        if maskpath:
            mask=TiffRegionReader(str(maskpath),None)
            stack.callback(mask.close)
            if (mask.height,mask.width)!=(he.height,he.width):
                raise ValueError('Validity mask dimensions differ')
        bounds=a.roi or [0,0,he.width//8*8,he.height//8*8]
        x0,y0,x1,y1=bounds
        if any(v%8 for v in bounds) or not (0<=x0<x1<=he.width and 0<=y0<y1<=he.height):
            raise ValueError('ROI must be within image, with bounds multiples of 8')
        out.mkdir(parents=True)
        coords, targets, shards=[],[],[]
        buffered,total=0,0
        def flush():
            nonlocal buffered
            if not buffered:
                return
            name=f'targets_{len(shards):05d}.npz'
            np.savez(out/name,xy=np.concatenate(coords),measured=np.concatenate(targets))
            shards.append(name); coords.clear(); targets.clear(); buffered=0
        for top in range(y0,y1,64):
            bottom=min(top+64,y1)
            for left in range(x0,x1,1024):
                right=min(left+1024,x1)
                valid=mask.read(top,bottom,left,right)!=0 if mask else np.ones((bottom-top,right-left),bool)
                image=mif.read(top,bottom,left,right)
                rgb=he.read(top,bottom,left,right) if a.white_threshold is not None else None
                xy,y=block_targets(image,valid,order,left,top,a.white_threshold,rgb)
                for start in range(0,len(xy),a.shard_size):
                    partxy,party=xy[start:start+a.shard_size],y[start:start+a.shard_size]
                    coords.append(partxy); targets.append(party); buffered+=len(partxy); total+=len(partxy)
                    if buffered>=a.shard_size:
                        flush()
        flush()
        if not total:
            raise ValueError('No valid target blocks retained')
        meta=dict(complete=True,sample_id=row['sample_id'],patient_id=patients[row['sample_id']],
            he=row['he'],mif=row['mif'],he_axes=a.he_axes,he_series=a.he_series,he_level=a.he_level,
            mif_axes=mif_axes,mif_series=a.mif_series,mif_level=a.mif_level,
            he_mpp=a.mpp,mif_mpp=a.mpp,shape=[he.height,he.width],panel=panel,
            input_patch=128,target_block=8,shards=shards,retained=total,roi=bounds,
            source_mif_markers=source,source_to_panel_reorder=order,
            he_metadata=he_info,mif_metadata=mif_info,
            valid_mask=str(maskpath) if maskpath else None,assume_all_valid=a.assume_all_valid,
            patient_identity_assumption='one sample per patient' if a.samples_as_patients else 'supplied patient map',
            white_threshold=a.white_threshold,intensity_transform='none; raw measured mean')
        (out/'manifest.json').write_text(json.dumps(meta,indent=2))
        return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--inspect',action='store_true',help='Read TIFF metadata during dry run')
    p.add_argument('--report',help='Optional discovery JSON file')
    p.add_argument('--out')
    p.add_argument('--panel',help='JSON list of exact measured marker identities')
    identity=p.add_mutually_exclusive_group()
    identity.add_argument('--patient-map',help='CSV sample_id,patient_id')
    identity.add_argument('--samples-as-patients',action='store_true',help='Explicit assumption: every sample is one independent patient')
    p.add_argument('--mpp',type=float,help='Verified common selected-level pixel size in microns')
    support=p.add_mutually_exclusive_group()
    support.add_argument('--valid-mask-name',help='Common YX support mask filename inside each sample folder')
    support.add_argument('--assume-all-valid',action='store_true',help='Explicitly assert full common support')
    p.add_argument('--roi',type=int,nargs=4,metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--samples',nargs='+',help='Optional explicit sample subset, preserving discovery rules')
    p.add_argument('--white-threshold',type=float,help='Optional mean central RGB cutoff; heuristic, review tissue retention')
    p.add_argument('--shard-size',type=int,default=50000)
    p.add_argument('--confirm-unlabeled-panel-order',action='store_true')
    for modality in ('he','mif'):
        p.add_argument('--'+modality+'-axes')
        p.add_argument('--'+modality+'-series',type=int,default=0)
        p.add_argument('--'+modality+'-level',type=int,default=0)
    a=p.parse_args()
    rows=discover(a.base_dir)
    if a.samples:
        available={row['sample_id'] for row in rows}
        if set(a.samples)-available:
            p.error(f'Samples absent from eligible folders: {set(a.samples)-available}')
        rows=[row for row in rows if row['sample_id'] in a.samples]
    if not rows:
        p.error('No eligible sample folders')
    if a.dry_run:
        if a.inspect:
            for row in rows:
                if not row['error']:
                    try:
                        row['he_header']=header(row['he'],a.he_series,a.he_level)
                        row['mif_header']=header(row['mif'],a.mif_series,a.mif_level)
                    except Exception as exc:
                        row['error']=f'{type(exc).__name__}: {exc}'
        report=json.dumps(rows,indent=2)
        print(report)
        if a.report:
            Path(a.report).write_text(report)
        return int(any(row['error'] for row in rows))
    if not all((a.out,a.panel,a.mpp)) or not (a.patient_map or a.samples_as_patients) or a.mpp<=0 or a.shard_size<1:
        p.error('Preparation requires --out --panel and (--patient-map or --samples-as-patients), plus positive --mpp, --shard-size')
    if not (a.valid_mask_name or a.assume_all_valid):
        p.error('Specify --valid-mask-name or explicitly --assume-all-valid')
    if a.white_threshold is not None and not 0<a.white_threshold<=255:
        p.error('white-threshold must be in (0,255]')
    panel=json.loads(Path(a.panel).read_text())
    if not isinstance(panel,list) or not panel or any(not isinstance(s,str) or not s for s in panel) or len(set(panel))!=len(panel):
        p.error('Panel must be a nonempty JSON list of unique marker names')
    patients=patient_map(a.patient_map) if a.patient_map else {row['sample_id']:row['sample_id'] for row in rows}
    Path(a.out).mkdir(parents=True,exist_ok=True)
    for row in rows:
        if not row['error']:
            try:
                row['retained']=prepare_sample(row,a,panel,patients)
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
        print(json.dumps(row),flush=True)
    (Path(a.out)/'batch_report.json').write_text(json.dumps(rows,indent=2))
    return int(any(row['error'] for row in rows))

if __name__=='__main__':
    raise SystemExit(main())
