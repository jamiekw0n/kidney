"""Unsupervised UMAP of cached tissue-context embeddings with centroid annotation joins."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from audit import snapshot,write_json,event
from data import read_meta


def summarize_counts(counts,min_cells,min_purity):
    n=sum(counts.values())
    if not n:return {'label':'UNANNOTATED','dominant':None,'purity':None,'labeled_cells':0,'composition':{}}
    dominant=sorted(counts,key=lambda k:(-counts[k],k))[0];purity=counts[dominant]/n
    tied=sum(v==counts[dominant] for v in counts.values())>1
    label=dominant if n>=min_cells and purity>=min_purity and not tied else 'MIXED_OR_LOW_SUPPORT'
    return {'label':label,'dominant':dominant,'purity':purity,'labeled_cells':n,'composition':dict(sorted(counts.items()))}


def half_open_matches(tree,centers,points,half):
    neighbors=tree.query_ball_point(points,half,p=np.inf)
    return [[i for i in ids if np.all(point>=centers[i]-half) and np.all(point<centers[i]+half)]
            for point,ids in zip(points,neighbors)]


def load_caches(root,samples,cap,seed):
    paths=sorted(Path(root).glob('*/manifest.json'))
    if samples:paths=[p for p in paths if p.parent.name in samples]
    if samples and {p.parent.name for p in paths}!=set(samples):raise ValueError('Requested samples missing from cache root')
    if not paths:raise ValueError('No caches')
    xs=[];targets=[];tiles=[];metas={};reference=None;inventory=[]
    for index,path in enumerate(paths):
        m=read_meta(path.parent);sid=m['sample_id']
        if sid in metas or sid!=path.parent.name or m.get('encoder')!='UNI2-h' or m.get('diagnostic_only') or m['input_patch']!=128 or m['target_block']!=8:
            raise ValueError('Duplicate/mismatched/non-UNI2-h/diagnostic cache')
        signature=[m[k] for k in ('panel','encoder','revision','checkpoint_sha256','transform','input_patch','he_mpp','target_block')]
        if reference is not None and signature!=reference:raise ValueError('Cache signatures differ')
        reference=signature
        arrays={k:[] for k in ('features','target','xy')}
        for shard in m['shards']:
            with np.load(path.parent/shard,allow_pickle=False) as z:
                for k in arrays:arrays[k].append(z[k].copy())
        x,y,xy=[np.concatenate(arrays[k]) for k in ('features','target','xy')]
        if x.shape!=(len(xy),1536) or y.shape!=(len(xy),len(m['panel'])) or xy.shape!=(len(x),2) or not len(x):raise ValueError('Invalid cache arrays')
        if not all(np.isfinite(v).all() for v in (x,y,xy)) or not np.issubdtype(xy.dtype,np.integer):raise ValueError('Nonfinite cache or noninteger coordinates')
        if len(np.unique(xy,axis=0))!=len(xy):raise ValueError('Duplicate cache locations')
        h,w=m['shape']
        if (xy<0).any() or (xy[:,0]>=w).any() or (xy[:,1]>=h).any():raise ValueError('Cache locations outside slide')
        rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
        keep=np.sort(rng.choice(len(x),min(cap,len(x)),replace=False))
        xs.append(x[keep]);targets.append(y[keep]);metas[sid]=m
        for k in keep:tiles.append({'sample_id':sid,'patient_id':m['patient_id'],'cache_row':int(k),'x':int(xy[k,0]),'y':int(xy[k,1])})
        inventory.append({'sample_id':sid,'patient_id':m['patient_id'],'cache_rows':len(x),'analysis_rows':len(keep),'cache':str(path.parent.resolve()),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'Loaded {sid}: {len(keep)}/{len(x)} embeddings',flush=True)
    return np.concatenate(xs),np.concatenate(targets),tiles,metas,inventory


def join_annotations(files,tiles,metas,columns,units,half,out):
    from scipy.spatial import cKDTree
    indices=defaultdict(list)
    for i,t in enumerate(tiles):indices[t['sample_id']].append(i)
    centers={s:np.array([[tiles[i]['x'],tiles[i]['y']] for i in ids]) for s,ids in indices.items()}
    trees={s:cKDTree(v) for s,v in centers.items()}
    counts={col:[Counter() for _ in tiles] for col in columns};cell_counts=np.zeros(len(tiles),np.int64)
    stats={s:Counter() for s in indices};ignored=Counter();blank=Counter();file_records=[]
    db=sqlite3.connect(out/'annotation_ids.sqlite3')
    db.execute('CREATE TABLE ids (sample TEXT, cell TEXT, PRIMARY KEY(sample,cell)) WITHOUT ROWID')
    required=['sample_id','cell_id','x','y',*columns]
    def consume(batch):
        for sid,records in batch.items():
            points=np.array([[r[0],r[1]] for r in records],float)
            matches=half_open_matches(trees[sid],centers[sid],points,half)
            for record,local in zip(records,matches):
                if local:stats[sid]['cells_matching_selected_contexts']+=1
                for k in local:
                    i=indices[sid][k];cell_counts[i]+=1
                    for col,label in zip(columns,record[2]):
                        if label:counts[col][i][label]+=1
    try:
        for file in files:
            before=file.stat();batch=defaultdict(list);batch_size=0
            with file.open(newline='',encoding='utf-8-sig') as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames and len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError(f'Duplicate CSV column names: {file}')
                if not set(required)<=set(reader.fieldnames or []):raise ValueError(f'Missing required columns in {file}: {set(required)-set(reader.fieldnames or [])}')
                for line,row in enumerate(reader,2):
                    if None in row or any(v is None for v in row.values()):raise ValueError(f'Malformed CSV row at {file}:{line}; column count differs from header')
                    sid=row['sample_id'].strip()
                    if sid not in metas:ignored[sid]+=1;continue
                    cell=row['cell_id'].strip()
                    if not cell:raise ValueError(f'Missing cell_id at {file}:{line}')
                    try:db.execute('INSERT INTO ids VALUES (?,?)',(sid,cell))
                    except sqlite3.IntegrityError as exc:raise ValueError(f'Duplicate sample/cell_id: {sid}/{cell}; avoid overlapping annotation files') from exc
                    try:x,y=float(row['x']),float(row['y'])
                    except ValueError as exc:raise ValueError(f'Invalid coordinates {file}:{line}') from exc
                    if units=='micrometers':x/=metas[sid]['he_mpp'];y/=metas[sid]['he_mpp']
                    h,w=metas[sid]['shape']
                    if not np.isfinite([x,y]).all() or not (0<=x<w and 0<=y<h):raise ValueError(f'Annotation coordinates outside registered image: {sid}/{cell}; check scale/origin/axes')
                    labels=[]
                    for col in columns:
                        label=row[col].strip()
                        if label.lower() in ('','nan','none','null','na'):label=''
                        if not label:blank[col]+=1
                        labels.append(label)
                    batch[sid].append((x,y,labels));stats[sid]['annotation_cells']+=1;batch_size+=1
                    if batch_size>=10000:
                        consume(batch);db.commit();batch=defaultdict(list);batch_size=0
                        print(f'Annotation rows processed: {sum(v["annotation_cells"] for v in stats.values())}',flush=True)
                consume(batch);db.commit()
            after=file.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Annotation file changed during read')
            file_records.append({'path':str(file.resolve()),'size_bytes':after.st_size,'mtime_ns':after.st_mtime_ns})
    finally:db.close()
    absent=[s for s in metas if not stats[s]['annotation_cells']]
    if absent:raise ValueError(f'No annotations for selected samples: {absent}; use --samples to select annotated samples')
    for sid,ids in indices.items():
        stats[sid]['selected_contexts']=len(ids)
        stats[sid]['contexts_with_cells']=int((cell_counts[ids]>0).sum())
    return counts,cell_counts,{'per_sample':{s:dict(v) for s,v in stats.items()},'ignored_sample_rows':dict(ignored),'blank_label_rows':dict(blank),'files':file_records}


def scatter_categories(plt,coordinates,labels,title,path):
    labels=np.asarray(labels);counts=Counter(labels);levels=sorted(counts,key=lambda k:(-counts[k],k))
    fig,ax=plt.subplots(figsize=(12,8),constrained_layout=True);cmap=plt.get_cmap('turbo')
    for i,label in enumerate(levels):
        keep=labels==label
        color='#bdbdbd' if label in ('UNANNOTATED','MIXED_OR_LOW_SUPPORT') else cmap(i/max(1,len(levels)-1))
        ax.scatter(coordinates[keep,0],coordinates[keep,1],s=4,alpha=.65,color=color,label=f'{label} ({counts[label]})',rasterized=True)
    ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2');ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=max(4,min(8,180/max(1,len(levels)))) ,markerscale=2)
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--annotations',nargs='+',help='One combined or several nonoverlapping CSVs')
    inputs.add_argument('--annotation-root',help='Sample directories, each containing exactly one CSV')
    p.add_argument('--out',required=True);p.add_argument('--samples',nargs='+')
    p.add_argument('--coordinate-units',choices=['native-pixels','micrometers'],required=True)
    p.add_argument('--confirm-registered-coordinates',action='store_true',help='Confirm same origin, orientation and registered H&E grid; units alone are insufficient')
    p.add_argument('--label-columns',nargs='+',default=['cell_type_merged','cell_type_new','compartment_refined','cell_state_primary'])
    p.add_argument('--markers',nargs='+',default=['CAV','CD8'],help='Measured cached target means for post hoc coloring only')
    p.add_argument('--join-region',choices=['context','target'],default='context')
    p.add_argument('--max-per-sample',type=int,default=1000);p.add_argument('--min-cells',type=int,default=3);p.add_argument('--min-purity',type=float,default=.6)
    p.add_argument('--pca-components',type=int,default=50);p.add_argument('--neighbors',type=int,default=15);p.add_argument('--min-dist',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--join-only',action='store_true')
    a=p.parse_args()
    if not a.confirm_registered_coordinates:p.error('Confirm coordinate registration before joining; there is no automatic registration')
    if min(a.max_per_sample,a.min_cells)<1 or a.pca_components<0 or a.neighbors<2 or a.seed<0 or not 0<=a.min_dist<=1 or not .5<=a.min_purity<=1:p.error('Invalid numeric settings')
    if not a.label_columns or len(set(a.label_columns))!=len(a.label_columns):p.error('Duplicate label columns')
    if not a.join_only:
        try:
            import sklearn,umap
        except ImportError as exc:raise ImportError('Install requirements-embedding.txt in the analysis environment, or use --join-only first') from exc
    if a.annotation_root:
        from annotation_coordinates import annotation_files
        selected=a.samples or [p.parent.name for p in Path(a.cache_root).glob('*/manifest.json')]
        files=annotation_files(Path(a.annotation_root),selected)
    else:files=list(map(Path,a.annotations))
    if len({f.resolve() for f in files})!=len(files) or any(not f.is_file() for f in files):raise ValueError('Missing/duplicate annotation files')
    out=Path(a.out).resolve();cache=Path(a.cache_root).resolve()
    if out==cache or cache in out.parents or out in cache.parents:raise ValueError('Output must be separate from caches')
    for file in files:
        if out==file.resolve().parent or out in file.resolve().parents:raise ValueError('Output overlaps annotation storage')
    if out.exists():raise FileExistsError(out)
    x,y,tiles,metas,inventory=load_caches(cache,a.samples,a.max_per_sample,a.seed)
    panel=next(iter(metas.values()))['panel']
    if set(a.markers)-set(panel):raise ValueError('Unknown marker color column')
    from raw_safety import guard_output
    guard_output(out,[{'folder':str(Path(m['he']).parent)} for m in metas.values()])
    out.mkdir(parents=True);snapshot(out,vars(a));write_json(out/'inventory.json',inventory)
    event(out,'annotation_join_start',embeddings=len(x),join_region=a.join_region)
    columns=a.label_columns;half=64 if a.join_region=='context' else 4
    counts,cell_counts,audit=join_annotations(files,tiles,metas,columns,a.coordinate_units,half,out)
    for i,tile in enumerate(tiles):
        tile['centroid_cells']=int(cell_counts[i]);tile['annotations']={col:summarize_counts(counts[col][i],a.min_cells,a.min_purity) for col in columns}
    norms=np.linalg.norm(x,axis=1)
    if not np.isfinite(norms).all() or (norms==0).any():raise ValueError('Zero/nonfinite feature norms')
    variance=x.var(axis=0,dtype=np.float64)
    if not np.any(variance>0):raise ValueError('All sampled embeddings are identical')
    audit['feature_qc']={'constant_dimensions':int((variance==0).sum()),'rows':len(x),'dimensions':x.shape[1],'norm_min':float(norms.min()),'norm_median':float(np.median(norms)),'norm_max':float(norms.max()),'join_region':a.join_region,'half_width_native_pixels':half}
    audit['labels']={col:dict(Counter(t['annotations'][col]['label'] for t in tiles)) for col in columns}
    write_json(out/'annotation_join_report.json',audit);write_json(out/'joined_tiles.json',tiles)
    if a.join_only:
        event(out,'join_complete_no_umap');print(f'Join QA saved: {out}',flush=True);return
    if len(x)<=a.neighbors:raise ValueError('Need more embeddings than n_neighbors; lower neighbors or include more samples')
    from sklearn.decomposition import PCA
    import umap
    # Annotation labels, marker measurements and coordinates never enter UMAP/PCA.
    z=(x/norms[:,None]).astype(np.float32);pca_state={}
    if a.pca_components:
        k=min(a.pca_components,len(z)-1,z.shape[1]);pca=PCA(n_components=k,svd_solver='randomized',random_state=a.seed,whiten=False)
        z=pca.fit_transform(z).astype(np.float32)
        pca_state={'components':k,'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'explained_variance_total':float(pca.explained_variance_ratio_.sum())}
        np.savez(out/'pca_transform.npz',mean=pca.mean_,components=pca.components_)
    event(out,'umap_start',rows=len(z),dimensions=z.shape[1])
    reducer=umap.UMAP(n_neighbors=a.neighbors,min_dist=a.min_dist,n_components=2,metric='euclidean',random_state=a.seed,n_jobs=1,verbose=True)
    coordinates=reducer.fit_transform(z)
    if coordinates.shape!=(len(tiles),2) or not np.isfinite(coordinates).all():raise ValueError('Invalid UMAP output')
    np.savez(out/'umap_points.npz',umap=coordinates,cache_row=np.array([t['cache_row'] for t in tiles]),sample_id=np.array([t['sample_id'] for t in tiles]),xy=np.array([[t['x'],t['y']] for t in tiles]))
    for t,point in zip(tiles,coordinates):t['umap']=point.astype(float).tolist()
    write_json(out/'joined_tiles.json',tiles)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cards=[]
    categorical=[('sample_id',[t['sample_id'] for t in tiles]),('patient_id',[t['patient_id'] for t in tiles])]+[(col,[t['annotations'][col]['label'] for t in tiles]) for col in columns]
    for i,(name,labels) in enumerate(categorical):
        filename=f'categories_{i:02d}.png';scatter_categories(plt,coordinates,labels,f'UNI2-h context embeddings — {name}',out/filename)
        cards.append(f'<h2>{html.escape(name)}</h2><img style="max-width:100%" src="{filename}">')
    for i,(name,values) in enumerate([('Cells with centroids in window',cell_counts),('Feature L2 norm',norms)]+[(f'Measured {marker}: central 8x8 raw mean',y[:,panel.index(marker)]) for marker in a.markers]+[(f'{col} dominant fraction',[t['annotations'][col]['purity'] if t['annotations'][col]['purity'] is not None else np.nan for t in tiles]) for col in columns]):
        fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True);im=ax.scatter(coordinates[:,0],coordinates[:,1],c=values,s=4,cmap='viridis',rasterized=True);fig.colorbar(im,ax=ax);ax.set(title=name,xlabel='UMAP 1',ylabel='UMAP 2')
        filename=f'continuous_{i:02d}.png';fig.savefig(out/filename,dpi=150);plt.close(fig);cards.append(f'<h2>{html.escape(name)}</h2><img style="max-width:100%" src="{filename}">')
    versions={}
    for package in ('numpy','scipy','scikit-learn','umap-learn','matplotlib'):
        versions[package]=importlib.metadata.version(package)
    write_json(out/'reduction_settings.json',{'l2_normalize':True,'pca':pca_state,'umap_metric':'euclidean','n_neighbors':a.neighbors,'min_dist':a.min_dist,'seed':a.seed,'labels_used_to_fit':False,'versions':versions})
    notes='Each point is a tissue-context embedding, not a cell. Cell annotations summarize centroids inside the selected window; no segmentation-mask overlap is inferred. Labels only color an unsupervised embedding. Mixed/low-support and unannotated contexts remain included. UMAP separation alone is not proof of correct embeddings, cell-type classification or predictive accuracy. Inspect sample/patient structure and join QA.'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>UNI2-h annotation-linked UMAP</h1><p>'+notes+'</p>'+''.join(cards))
    write_json(out/'manifest.json',{'complete':True,'embeddings':len(x),'samples':len(metas),'join_region':a.join_region,'labels_used_to_fit':False})
    event(out,'complete');print(f'Open {out}/index.html',flush=True)

if __name__=='__main__':main()
