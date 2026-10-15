"""Patient distributions, confidence intervals, expression-range diagnostics and OOF maps."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from audit import snapshot,write_json
from evaluate_head import render_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['VIM','CD10','PCNA','CAV','CD34','CD31','PODXL'])
    p.add_argument('--maps-per-fold',type=int,default=1,help='Alphabetically selected samples per fold; 0 disables')
    a=p.parse_args()
    if a.maps_per_fold<0:p.error('maps-per-fold must be nonnegative')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(a.cv_root);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    # Recalculate from all saved outer predictions; verifies role and inventory first.
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_oof.py')),
        '--cv-root',str(root),'--out',str(out/'primary')],check=True)
    report=json.loads((out/'primary/primary_patient_metrics.json').read_text())
    summary=json.loads((out/'primary/primary_summary.json').read_text())
    if set(a.markers)-set(report):raise ValueError('Requested marker absent')
    panels=[]
    for index,marker in enumerate(a.markers):
        patients=report[marker]['patients']
        figure,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for ax,key,ideal in zip(axes.flat,('mae','rmse','pearson','spearman','mean_bias','prediction_on_measured_slope'),(None,None,1,1,0,1)):
            values=[v[key] for v in patients.values() if v[key] is not None]
            ax.scatter(np.arange(len(values)),values,s=18,alpha=.75)
            item=summary['metrics'][marker][key]
            if item['patient_mean'] is not None:ax.axhline(item['patient_mean'],color='tab:orange',label='Patient mean')
            if ideal is not None:ax.axhline(ideal,color='gray',linestyle='--',label='Ideal')
            if item['ci95_low'] is not None:ax.axhspan(item['ci95_low'],item['ci95_high'],color='tab:orange',alpha=.15,label='Mean bootstrap CI')
            ax.set_title(f'{key}\nDefined patients: {item["defined_patients"]}');ax.set_xlabel('Patient index (defined values, sorted IDs)');ax.legend(fontsize=7)
        figure.suptitle(f'{marker}: out-of-fold patient metrics; CI conditional on fixed predictions')
        filename=f'patients_{index:02d}.png';figure.savefig(out/filename,dpi=140);plt.close(figure)
        panels.append(f'<h2>{html.escape(marker)}</h2><img style="max-width:100%" src="{filename}">')
    folds=json.loads((root/'folds.json').read_text());map_links=[];diagnostics=[];map_selection=[]
    for i in range(len(folds)):
        samples=sorted((root/f'fold_{i:02d}'/'outer_predictions').glob('*/manifest.json'))
        selected={m.parent.name for m in samples[:a.maps_per_fold]}
        for manifest in samples:
            meta=json.loads(manifest.read_text());ys=[];ps=[];coords=[]
            for name in meta['shards']:
                with np.load(manifest.parent/name,allow_pickle=False) as z:
                    ys.append(z['target'].copy());ps.append(z['prediction'].copy());coords.append(z['xy'].copy())
            y,pred,xy=np.concatenate(ys),np.concatenate(ps),np.concatenate(coords)
            for j,marker in enumerate(meta['panel']):
                true=y[:,j].astype(float);estimated=pred[:,j].astype(float)
                true_std=true.std();pred_std=estimated.std()
                diagnostics.append({'fold':i,'patient_id':meta['patient_id'],'sample_id':meta['sample_id'],'marker':marker,
                    'locations':len(y),'measured_std':float(true_std),'predicted_std':float(pred_std),
                    'std_ratio':float(pred_std/true_std) if true_std>0 else None,
                    'measured_p50':float(np.percentile(true,50)),'measured_p95':float(np.percentile(true,95)),
                    'predicted_p50':float(np.percentile(estimated,50)),'predicted_p95':float(np.percentile(estimated,95)),
                    'measured_zero_fraction':float((true==0).mean()),'predicted_negative_fraction':float((estimated<0).mean())})
            if meta['sample_id'] in selected:
                target=out/f'fold_{i:02d}'/meta['sample_id'];target.mkdir(parents=True)
                render_sample(target,{**meta,'evaluation_role':'outer out-of-fold'},xy,y,pred,a.markers)
                relative=str(target.relative_to(out)/'index.html')
                map_links.append(f'<li><a href="{html.escape(relative)}">Fold {i}: {html.escape(meta["sample_id"])}</a></li>')
                map_selection.append({'fold':i,'sample_id':meta['sample_id'],'method':'First alphabetically sorted sample(s), not metric-selected'})
            print(f'Reviewed saved OOF sample {meta["sample_id"]}',flush=True)
    write_json(out/'expression_diagnostics.json',diagnostics);write_json(out/'map_selection.json',map_selection)
    with (out/'expression_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>OOF cohort review</title><h1>Patient-held-out expression review</h1><p>Each point is one patient, equally weighted. Intervals quantify mean-metric patient resampling conditional on fixed CV predictions, not full retraining uncertainty. Selected maps are exploratory examples, not best-case samples. Sparse location cells are not dense protein maps.</p><h2>Selected OOF maps</h2><ul>'+''.join(map_links)+'</ul>'+''.join(panels))
    print(f'Open {out}/index.html after copying the whole report directory locally')

if __name__=='__main__':main()
