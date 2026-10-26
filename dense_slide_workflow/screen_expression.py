"""Rank measured expression per marker using exact whole-slide histograms and sampled tissue targets."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from io_utils import independent_output
from data import read_meta
from encoder_runtime import sha


def histogram_stats(counts,divisor):
    counts=np.asarray(counts)
    if counts.ndim!=1 or (counts<0).any() or counts.sum()<=0 or divisor<=0:raise ValueError('Invalid histogram/divisor')
    n=int(counts.sum());values=np.arange(len(counts),dtype=float)
    return {'whole_slide_raw_mean':float(counts@values/n),
        'whole_slide_normalized_mean':float(counts@np.clip(values/divisor,0,1)/n),
        'whole_slide_nonzero_fraction':float(1-counts[0]/n)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['PODXL','VIM','CAV','CD138']);p.add_argument('--exclude',nargs='*',default=['F03600','K2120228','K2140231'])
    a=p.parse_args();run=Path(a.run).resolve();cfg=json.loads((run/'experiment.json').read_text());folds=json.loads((run/'folds.json').read_text())
    if not a.markers or len(a.markers)!=len(set(a.markers)):raise ValueError('Unique markers required')
    out=independent_output(a.out,[run,*[Path(t['normalized_cache']) for t in cfg['tasks']],*[Path(t['raw_paths'][k]).parent for t in cfg['tasks'] for k in ('he','mif')]])
    rows=[]
    for task in cfg['tasks']:
        if task['sample_id'] in a.exclude:continue
        original=Path(task['normalized_cache']);prepared=run/'prepared'/task['sample_id'];meta=read_meta(prepared)
        panel=meta['panel']
        if set(a.markers)-set(panel):raise ValueError('Unknown marker')
        if sha(original/'normalization.json')!=task['normalization_sha256']:raise ValueError('Normalization reference changed')
        report=json.loads((original/'normalization.json').read_text())
        if [r['marker'] for r in report['markers']]!=panel:raise ValueError('Normalization panel order differs')
        indices=[i for i,f in enumerate(folds) if task['patient_id'] in f['test']]
        if len(indices)!=1:raise ValueError('Need exactly one outer fold')
        state=json.loads((run/'cv'/f'part_{indices[0]:02d}'/f'fold_{indices[0]:02d}'/'training/expression_state.json').read_text())
        if state['panel']!=panel:raise ValueError('Threshold panel mismatch')
        targets=[];raw=[]
        for name in meta['shards']:
            with np.load(prepared/name,allow_pickle=False) as z:targets.append(z['measured']);raw.append(z['raw_target'])
        y=np.concatenate(targets);raw=np.concatenate(raw)
        if y.shape!=raw.shape or not np.isfinite(y).all() or not np.isfinite(raw).all():raise ValueError('Invalid sampled targets')
        with np.load(original/'whole_slide_histograms.npz',allow_pickle=False) as z:
            if list(z['panel'])!=panel:raise ValueError('Histogram panel order mismatch')
            hist=z['histograms']
        if not np.all(hist.sum(1)==meta['shape'][0]*meta['shape'][1]):raise ValueError('Whole-slide histogram pixel count differs')
        for marker in a.markers:
            j=panel.index(marker);r=report['markers'][j];threshold=state['thresholds'][j]
            rows.append({'sample_id':task['sample_id'],'patient_id':task['patient_id'],'marker':marker,
                **histogram_stats(hist[j],r['effective_divisor']),
                'sampled_tissue_raw_mean':float(raw[:,j].mean()),'sampled_tissue_normalized_mean':float(y[:,j].mean()),
                'sampled_high_fraction':float((y[:,j]>threshold).mean()),'sampled_locations':len(y),
                'training_high_threshold':threshold,'outer_fold':indices[0],
                'percentile_value':r['percentile_value'],'zero_p99_fallback':r['fallback_dtype_max']})
    if not rows:raise ValueError('No eligible slides')
    out.mkdir(parents=True)
    with (out/'expression_by_slide_marker.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Measured expression screening. Whole-slide values include background; tissue values estimate retained eligible tissue.',
           'Selection is for qualitative examples, not an unbiased performance estimate. High expression is NOT biological positivity.',
           'Raw intensity scales can differ technically between slides; P99-normalized mean alone does not establish stronger absolute expression.']
    for marker in a.markers:
        lines.append('\n'+marker+' — top slides by sampled tissue normalized mean:')
        for r in sorted([r for r in rows if r['marker']==marker],key=lambda r:r['sampled_tissue_normalized_mean'],reverse=True)[:5]:
            lines.append(f"{r['sample_id']}: tissue normalized mean={r['sampled_tissue_normalized_mean']:.4f}; tissue raw mean={r['sampled_tissue_raw_mean']:.3f}; full-slide raw mean={r['whole_slide_raw_mean']:.3f}; training-threshold high fraction={r['sampled_high_fraction']:.3f}; zero-P99 fallback={r['zero_p99_fallback']}")
    text='\n'.join(lines)+'\n';(out/'REVIEW.txt').write_text(text);print(text,flush=True)
    (out/'manifest.json').write_text(json.dumps({'complete':True,'run':str(run),'markers':a.markers,'excluded':a.exclude,'rows':len(rows)},indent=2))

if __name__=='__main__':main()
