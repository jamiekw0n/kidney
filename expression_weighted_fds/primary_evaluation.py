"""Equal-patient primary metrics with patient bootstrap uncertainty; no tile bootstrap."""
import csv
from pathlib import Path
import numpy as np
from metrics import METRIC_KEYS
from audit import write_json


def summarize(report,replicates=2000,seed=42):
    if replicates<1:raise ValueError('Positive bootstrap count required')
    patient_sets=[set(record['patients']) for record in report.values()]
    if not patient_sets or any(s!=patient_sets[0] for s in patient_sets):raise ValueError('Every marker must have the same patient inventory')
    patients=sorted(patient_sets[0]);n=len(patients)
    if not n:raise ValueError('No evaluation patients')
    draws=np.random.default_rng(seed).integers(0,n,size=(replicates,n)) if n>=2 else None
    results={}
    for marker,record in report.items():
        results[marker]={}
        for metric in METRIC_KEYS:
            values=np.array([np.nan if record['patients'][p][metric] is None else record['patients'][p][metric] for p in patients],float)
            defined=np.isfinite(values);count=int(defined.sum())
            lower=upper=None;effective=0
            if n>=2 and count>=2:
                sampled=values[draws];valid=np.isfinite(sampled);denom=valid.sum(1)
                means=np.divide(np.where(valid,sampled,0).sum(1),denom,out=np.full(replicates,np.nan),where=denom>0)
                means=means[np.isfinite(means)];effective=len(means)
                lower,upper=map(float,np.percentile(means,[2.5,97.5]))
            results[marker][metric]={'patient_mean':float(values[defined].mean()) if count else None,
                'patient_sd':float(values[defined].std(ddof=1)) if count>=2 else None,
                'defined_patients':count,'total_patients':n,'ci95_low':lower,'ci95_high':upper,
                'effective_bootstrap_replicates':effective}
    return results


def save_primary(out,report,role,replicates=2000,seed=42):
    out=Path(out);summary=summarize(report,replicates,seed)
    write_json(out/'primary_patient_metrics.json',report)
    write_json(out/'primary_summary.json',{'role':role,'metrics':summary,
        'method':'Per-patient metrics then equal-patient arithmetic means; percentile patient bootstrap',
        'bootstrap_replicates':replicates,'bootstrap_seed':seed,
        'bias_definition':'mean(predicted - measured)',
        'calibration_definition':'OLS predicted = intercept + slope * measured; ideal slope 1/intercept 0',
        'undefined':'Constant-vector correlations and constant-measured slope/intercept are null',
        'limitations':['No CI for fewer than two defined patients; few-patient intervals are unstable',
            'Intervals reflect evaluation-patient sampling conditional on trained predictions; no retraining bootstrap',
            'Cross-validation models share training patients; these intervals do not capture all fold/model uncertainty',
            'Means for undefined metrics use only defined patients; counts reported',
            'Validation used for stopping is not independent testing; no individual prediction confidence estimated']})
    with (out/'primary_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['marker','metric','patient_mean','patient_sd','defined_patients','total_patients','ci95_low','ci95_high'])
        for marker,values in summary.items():
            for metric,item in values.items():writer.writerow([marker,metric,*[item[k] for k in ('patient_mean','patient_sd','defined_patients','total_patients','ci95_low','ci95_high')]])
    with (out/'primary_patient_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['marker','patient_id','locations',*METRIC_KEYS])
        for marker,record in report.items():
            for patient,item in record['patients'].items():writer.writerow([marker,patient,item['locations'],*[item[k] for k in METRIC_KEYS]])
