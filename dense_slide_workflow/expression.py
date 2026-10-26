"""Training-only expression strata, fixed capped weights, and patient-level evaluation."""
import json
from pathlib import Path
import numpy as np


def patient_weights(patients):
    ids, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    return 1.0 / (len(ids) * counts[inverse])


def fit_expression(y, patients, panel, threshold_file=None, quantile=.9, max_weight=5.):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape != (len(patients), len(panel)) or not len(y) or not np.isfinite(y).all():
        raise ValueError('Invalid expression training targets')
    if not 0 < quantile < 1 or not np.isfinite(max_weight) or max_weight < 1:
        raise ValueError('Need quantile in (0,1), finite weight cap >=1')
    weights = patient_weights(patients)
    if threshold_file:
        specified = json.loads(Path(threshold_file).read_text())
        if set(specified) != set(panel):
            raise ValueError('Threshold JSON must match the full panel exactly')
        thresholds = np.asarray([specified[m] for m in panel], dtype=float)
        if not np.isfinite(thresholds).all(): raise ValueError('Nonfinite thresholds')
        method = 'user_supplied_normalized_block_thresholds'
    else:
        thresholds = []
        for j in range(len(panel)):
            order = np.argsort(y[:, j], kind='stable')
            idx = min(np.searchsorted(np.cumsum(weights[order]), quantile), len(y)-1)
            thresholds.append(y[order[idx], j])
        thresholds = np.asarray(thresholds)
        method = 'training_only_equal_patient_weighted_quantile_NOT_biological_positivity'
    positive = y > thresholds
    p = (positive * weights[:, None]).sum(0)
    active = (p > 1e-12) & (p < 1-1e-12)
    wp, wn = np.ones(len(panel)), np.ones(len(panel))
    # Inverse stratum prevalence gives 50/50 expected loss mass before capping.
    wp[active] = np.minimum(.5 / p[active], max_weight)
    wn[active] = np.minimum(.5 / (1-p[active]), max_weight)
    normalizer = p*wp + (1-p)*wn
    wp /= normalizer; wn /= normalizer
    # One common per-marker rescale keeps the ACTUAL weights below the cap.
    rescale = np.maximum(1, np.maximum(wp, wn)/max_weight)
    wp /= rescale; wn /= rescale
    return {'method':method,'quantile':quantile,'comparison':'normalized_block_target > threshold; ties are low',
            'max_weight':max_weight,'panel':list(panel),'thresholds':thresholds.tolist(),
            'positive_prevalence_equal_patient':p.tolist(),'positive_weight':wp.tolist(),
            'negative_weight':wn.tolist(),'active':active.tolist(),
            'positive_locations':positive.sum(0).tolist(),
            'positive_patients':[int(sum(positive[np.asarray(patients)==pid,j].any() for pid in np.unique(patients))) for j in range(len(panel))],
            'expected_weight':(p*wp+(1-p)*wn).tolist(),
            'note':'Degenerate markers use unit weights; cap can prevent exact 50/50 balance. No validation/test fitting.'}


def element_weights(y, state):
    return np.where(np.asarray(y)>state['thresholds'],state['positive_weight'],state['negative_weight']).astype(np.float32)


def average_precision(truth, score):
    """Threshold-grouped non-interpolated AP; ties handled together."""
    if not truth.any() or truth.all(): return None
    order=np.argsort(-score,kind='stable'); t=truth[order]; s=score[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(t)[ends]; recall=tp/t.sum(); precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def subgroup_metrics(y, pred, patients, panel, state):
    from metrics import metrics
    y,pred=np.asarray(y),np.asarray(pred)
    if list(panel)!=state['panel']: raise ValueError('Expression panel mismatch')
    # Also validates finite values and dimensions.
    metrics(y,pred,patients,panel)
    result={}
    for j,marker in enumerate(panel):
        result[marker]={}
        threshold=state['thresholds'][j]
        for patient in np.unique(patients):
            keep=patients==patient; a,b=y[keep,j],pred[keep,j]
            pos=a>threshold; called=b>threshold
            tp=int((pos & called).sum()); fp=int((~pos & called).sum())
            def err(mask): return float(np.abs(a[mask]-b[mask]).mean()) if mask.any() else None
            result[marker][str(patient)]={'threshold':threshold,'locations':len(a),'positive_locations':int(pos.sum()),
                'negative_locations':int((~pos).sum()),'positive_mae':err(pos),'negative_mae':err(~pos),
                'precision':tp/int(called.sum()) if called.any() else None,
                'recall':tp/int(pos.sum()) if pos.any() else None,
                'false_positive_rate':fp/int((~pos).sum()) if (~pos).any() else None,
                'average_precision':average_precision(pos,b),'measured_positive_fraction':float(pos.mean()),
                'predicted_positive_fraction':float(called.mean())}
    return result


def summarize_subgroups(records,panel):
    summary={}
    keys=['positive_mae','negative_mae','precision','recall','false_positive_rate','average_precision',
          'measured_positive_fraction','predicted_positive_fraction']
    for marker in panel:
        rows=[r[marker][patient] for patient,r in records.items()]
        summary[marker]={}
        for key in keys:
            values=[row[key] for row in rows if row[key] is not None]
            summary[marker][key]={'patient_mean':float(np.mean(values)) if values else None,
                                  'defined_patients':len(values),'total_patients':len(rows)}
    return {'role':'out_of_fold','aggregation':'equal patient mean; undefined metrics excluded with counts',
            'warning':'Quantile-defined high expression is not assay-defined positivity. Thresholds differ by training fold.',
            'metrics':summary}
