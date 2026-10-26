"""Compare measured mIF, baseline and expression-weighted + FDS head at matched OOF locations."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from audit import snapshot, write_json
from metrics import metrics
from primary_evaluation import save_primary


def load_sample(path):
    meta=json.loads(path.read_text())
    if not meta.get('complete'): raise ValueError('Incomplete predictions')
    arrays={k:[] for k in ('xy','target','prediction')}
    for name in meta['shards']:
        with np.load(path.parent/name,allow_pickle=False) as z:
            for k in arrays: arrays[k].append(z[k].copy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    xy,y,p=arrays['xy'],arrays['target'],arrays['prediction']
    if xy.shape!=(len(y),2) or y.ndim!=2 or y.shape!=p.shape or y.shape[1]!=len(meta['panel']) or not len(y):
        raise ValueError('Invalid prediction arrays')
    if not all(np.isfinite(v).all() for v in arrays.values()) or not np.issubdtype(xy.dtype,np.integer):
        raise ValueError('Nonfinite values or noninteger coordinates')
    if len(np.unique(xy,axis=0))!=len(xy): raise ValueError('Duplicate locations')
    order=np.lexsort((xy[:,0],xy[:,1]))
    return meta,xy[order],y[order],p[order]


def inventory(root):
    rows=json.loads((root/'inventory.json').read_text())
    result={row['sample_id']:row for row in rows}
    if len(result)!=len(rows): raise ValueError('Duplicate inventory samples')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',required=True);p.add_argument('--candidate-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--baseline-label',default='Expression-weighted MLP')
    p.add_argument('--candidate-label',default='Expression-weighted MLP + FDS')
    p.add_argument('--intensity-max',type=float,required=True,help='Declared source intensity ceiling; use 255 only after verifying uint8')
    p.add_argument('--positive-threshold',type=float,required=True,help='Raw measured mean must be strictly above this value; 0 means nonzero, not biological positivity')
    p.add_argument('--markers',nargs='+',default=['VIM','CD10','PCNA','CAV','CD34','CD31','PODXL','CD8','CD138','COLIV','POSTN'])
    p.add_argument('--samples',nargs='+',help='Restrict plots only; numeric reports still include full cohort')
    a=p.parse_args()
    if not np.isfinite(a.intensity_max) or a.intensity_max<=0 or not np.isfinite(a.positive_threshold) or not 0<=a.positive_threshold<a.intensity_max:
        p.error('Require positive finite ceiling and threshold in [0, ceiling)')
    roots=[Path(a.baseline_root),Path(a.candidate_root)];invs=[inventory(r) for r in roots]
    if {s:(v['patient_id'],v['rows']) for s,v in invs[0].items()}!={s:(v['patient_id'],v['rows']) for s,v in invs[1].items()}:
        raise ValueError('Sample/patient/row inventories differ')
    folds=[json.loads((r/'folds.json').read_text()) for r in roots]
    if folds[0]!=folds[1]: raise ValueError('Patient folds differ')
    expected={}
    for i,f in enumerate(folds[0]):
        groups=[set(f[k]) for k in ('train','val','test')]
        if any(groups[j]&groups[k] for j,k in ((0,1),(0,2),(1,2))): raise ValueError('Patient leakage')
        for patient in f['test']:
            if patient in expected:raise ValueError('Repeated outer patient')
            expected[patient]=i
    if set(expected)!={v['patient_id'] for v in invs[0].values()}:raise ValueError('Outer inventory incomplete')
    manifests=[]
    for root in roots:
        paths=list(root.glob('fold_[0-9][0-9]/outer_predictions/*/manifest.json'))
        index={json.loads(path.read_text())['sample_id']:path for path in paths}
        if len(index)!=len(paths) or set(index)!=set(invs[0]):raise ValueError('Missing/duplicate prediction samples')
        manifests.append(index)
    if a.samples and set(a.samples)-set(invs[0]):raise ValueError('Unknown requested samples')
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values={};counts=[];cards=[];panel=None
    for sid in sorted(invs[0]):
        m,xy,y,linear=load_sample(manifests[0][sid]);mm,mxy,my,mlp=load_sample(manifests[1][sid])
        if any(m[k]!=mm[k] for k in ('patient_id','panel','shape','outer_fold','selection_policy','encoder','revision','checkpoint_sha256','transform','input_patch','he_mpp','target_block')) or not np.array_equal(xy,mxy) or not np.array_equal(y,my):
            raise ValueError(f'Unmatched ground truth/coordinates/metadata: {sid}')
        for meta,path in zip((m,mm),(manifests[0][sid],manifests[1][sid])):
            if meta['outer_fold']!=expected[meta['patient_id']] or path.parents[2].name!=f'fold_{meta["outer_fold"]:02d}':raise ValueError('Wrong OOF role')
        if m['patient_id']!=invs[0][sid]['patient_id'] or len(y)!=invs[0][sid]['rows']:raise ValueError('Inventory rows/identity mismatch')
        if panel is not None and panel!=m['panel']:raise ValueError('Panel mismatch across cohort')
        panel=m['panel']
        if set(a.markers)-set(panel):raise ValueError('Unknown marker')
        if (y<0).any() or (y>a.intensity_max).any():raise ValueError('Targets outside declared source intensity range')
        counts.append({'sample_id':sid,'patient_id':m['patient_id'],'fold':m['outer_fold'],'tiles':len(y)})
        group=values.setdefault(m['patient_id'],[[],[],[]])
        for dest,v in zip(group,(y,linear,mlp)):dest.append(v/a.intensity_max)
        if a.samples and sid not in a.samples:continue
        directory=out/sid;directory.mkdir();items=[];selections=[]
        for j,marker in enumerate(panel):
            if marker not in a.markers:continue
            mask=y[:,j]>a.positive_threshold
            selections.append({'marker':marker,'all_locations':len(y),'gt_above_threshold':int(mask.sum())})
            combined=np.concatenate((y[:,j],linear[:,j],mlp[:,j]))/a.intensity_max
            low,high=float(combined.min()),float(combined.max())
            if high<=low:high=low+1e-6
            fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
            for row,keep in enumerate((np.ones(len(y),bool),mask)):
                for ax,v,title in zip(axes[row],(y[:,j],linear[:,j],mlp[:,j]),('Measured mIF',a.baseline_label,a.candidate_label)):
                    im=ax.scatter(xy[keep,0],xy[keep,1],c=v[keep]/a.intensity_max,s=9,marker='s',vmin=low,vmax=high,cmap='viridis')
                    ax.set_xlim(0,m['shape'][1]);ax.set_ylim(m['shape'][0],0);ax.set_aspect('equal')
                    ax.set_facecolor('#e5e7eb');ax.set_title(title+(' — all sampled' if row==0 else ' — GT-selected'))
                    ax.set_xlabel('Native x');ax.set_ylabel('Native y')
            fig.colorbar(im,ax=list(axes.flat),label=f'Intensity / {a.intensity_max:g}; same scale in all six panels')
            fig.suptitle(f'{sid} | {marker} | GT > {a.positive_threshold:g}: {mask.sum()}/{len(y)} locations\nSquares mark sampled centers, not dense expression or cell labels')
            filename=f'marker_{j:02d}.png';fig.savefig(directory/filename,dpi=140);plt.close(fig)
            items.append(f'<h2>{html.escape(marker)}</h2><img style="max-width:100%" src="{filename}">')
        write_json(directory/'selection_counts.json',selections)
        (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><p>Gray is unshown. Bottom panels use the identical ground-truth mask for all models; this hides model false positives outside that mask, so inspect the top row too. Threshold denotes an exploratory intensity selection, not biological positivity.</p>'+''.join(items))
        cards.append(f'<li><a href="{html.escape(sid)}/index.html">{html.escape(sid)}</a></li>')
        print(f'Compared {sid}: {len(y)} tiles',flush=True)
    total=sum(v['tiles'] for v in counts)
    write_json(out/'tile_counts.json',{'total_unique_tiles':total,'samples':len(counts),'patients':len(values),'per_sample':counts,'definition':'Unique selected locations, counted once, not epochs/folds or dense coverage'})
    reports={name:{marker:{'patients':{}} for marker in panel} for name in ('baseline','candidate')}
    for patient,arrays in values.items():
        y,linear,mlp=map(np.concatenate,arrays)
        for name,pred in (('baseline',linear),('candidate',mlp)):
            result=metrics(y,pred,np.full(len(y),patient),panel)
            for marker in panel:reports[name][marker]['patients'][patient]=result[marker]['patients'][patient]
    for name,report in reports.items():
        dest=out/name;dest.mkdir();save_primary(dest,report,'out_of_fold_normalized_by_declared_intensity_max')
    write_json(out/'units.json',{'normalization':'measured and predicted divided by fixed declared intensity_max; no clipping','intensity_max':a.intensity_max,'positive_threshold_raw':a.positive_threshold,'positive_threshold_normalized':a.positive_threshold/a.intensity_max,'metric_population':'All sampled locations, not only GT-selected plots'})
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>Baseline / expression-weighted + FDS OOF review</h1><p>Exact unique selected tiles: {total}; {len(values)} patients. Metrics use all sampled locations. GT-selected panels are exploratory and omit negatives.</p><ul>'+''.join(cards)+'</ul>')
    print(json.dumps({'total_unique_tiles':total,'samples':len(counts),'patients':len(values),'report':str(out/'index.html')},indent=2))

if __name__=='__main__':main()
