"""Finite float64 regression metrics; undefined Pearson correlations are JSON null."""
import numpy as np


def pearson(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if len(a)<2:
        return None
    # Float32 mean/std can falsely report variation in a constant vector.
    if np.ptp(a)==0 or np.ptp(b)==0:
        return None
    a=a-a.mean(); b=b-b.mean()
    an,bn=np.linalg.norm(a),np.linalg.norm(b)
    if an==0 or bn==0:
        return None
    value=float(np.dot(a/an,b/bn))
    return float(np.clip(value,-1,1)) if np.isfinite(value) else None


def metrics(y,pred,patients,panel):
    y,pred=np.asarray(y,dtype=np.float64),np.asarray(pred,dtype=np.float64)
    if y.shape!=pred.shape or y.ndim!=2 or y.shape[1]!=len(panel) or len(patients)!=len(y):
        raise ValueError('Metric input shapes differ')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite measured values or predictions')
    result={}
    for j,marker in enumerate(panel):
        individual={}
        for patient in np.unique(patients):
            keep=patients==patient
            a,b=y[keep,j],pred[keep,j]
            diff=a-b
            individual[str(patient)]={'mae':float(np.abs(diff).mean()),
                'rmse':float(np.sqrt(np.square(diff).mean())), 'pearson':pearson(a,b)}
        result[marker]={'patients':individual}
        for key in ('mae','rmse','pearson'):
            values=[v[key] for v in individual.values() if v[key] is not None]
            result[marker]['patient_mean_'+key]=float(np.mean(values)) if values else None
    return result
