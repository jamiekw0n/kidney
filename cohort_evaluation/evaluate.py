"""Recompute harmonized OOF metrics from saved predictions; no training/GPU/raw reads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import snapshot,write_json
from metrics import metrics,METRIC_KEYS
from expression import subgroup_metrics
from plan import default_plan


SUBKEYS=('positive_mae','negative_mae','precision','recall','false_positive_rate','average_precision',
         'measured_positive_fraction','predicted_positive_fraction')
EXTRA=('high_bias','low_bias','roc_auc','diagnostic_recall_at_fpr_0.01')
LOWER={'mae','rmse','positive_mae','negative_mae'}
HIGHER={'pearson','spearman','precision','recall','average_precision','roc_auc','diagnostic_recall_at_fpr_0.01'}


def load_sample(path):
    m=json.loads(path.read_text())
    if not m.get('complete'):raise ValueError(f'Incomplete predictions: {path}')
    values={k:[] for k in ['xy','target','prediction']}
    for name in m['shards']:
        shard=path.parent/name
        if shard.resolve().parent!=path.parent.resolve():raise ValueError('Shard must be a direct child')
        with np.load(shard,allow_pickle=False) as z:
            for k in values:values[k].append(z[k].copy())
    xy,y,p=[np.concatenate(values[k]) for k in ['xy','target','prediction']]
    if xy.shape!=(len(y),2) or y.ndim!=2 or p.shape!=y.shape or y.shape[1]!=len(m['panel']) or len(y)==0:
        raise ValueError('Invalid prediction shapes')
    if not np.issubdtype(xy.dtype,np.integer) or not all(np.isfinite(v).all() for v in [xy,y,p]):raise ValueError('Nonfinite values or invalid coordinates')
    if len(np.unique(xy,axis=0))!=len(xy):raise ValueError('Duplicate sampled centers')
    order=np.lexsort((xy[:,0],xy[:,1]))
    return m,xy[order],y[order].astype(float),p[order].astype(float)


def index_run(root):
    folds=json.loads((root/'folds.json').read_text());inventory=json.loads((root/'inventory.json').read_text())
    inv={r['sample_id']:(r['patient_id'],r['rows']) for r in inventory}
    if len(inv)!=len(inventory):raise ValueError('Duplicate inventory samples')
    expected={}
    for i,f in enumerate(folds):
        if set(f)!={'train','val','test'}:raise ValueError('Invalid split fields')
        lists=[f[k] for k in ['train','val','test']]
        flat=sum(lists,[])
        if len(flat)!=len(set(flat)):raise ValueError('Patient leakage/duplicates')
        if set(flat)!={v[0] for v in inv.values()}:raise ValueError('Fold inventory mismatch')
        for pid in f['test']:
            if pid in expected:raise ValueError('Repeated outer patient')
            expected[pid]=i
    if set(expected)!={v[0] for v in inv.values()}:raise ValueError('Missing outer patients')
    paths={}
    for path in root.glob('fold_*/outer_predictions/*/manifest.json'):
        m=json.loads(path.read_text());sid=m['sample_id'];pid=m['patient_id'];i=m.get('outer_fold')
        if sid in paths or sid not in inv or not m.get('complete'):raise ValueError('Duplicate/incomplete/unexpected sample')
        if inv[sid][0]!=pid or expected.get(pid)!=i or path.parents[2].name!=f'fold_{i:02d}':raise ValueError('Wrong outer role')
        paths[sid]=path
    if set(paths)!=set(inv):raise ValueError(f'Missing prediction samples: {root}')
    return folds,inv,paths


def roc_diagnostics(truth,score):
    if not truth.any() or truth.all():return None,None
    order=np.argsort(-score,kind='stable');truth=truth[order];score=score[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    tp=np.r_[0,np.cumsum(truth)[ends]]/truth.sum()
    fp=np.r_[0,np.cumsum(~truth)[ends]]/(~truth).sum()
    auc=float(np.sum((tp[1:]+tp[:-1])*.5*np.diff(fp)))
    # Descriptive outer ROC operating point, NOT a deployable calibrated threshold.
    recall=float(tp[fp<=.01+1e-12].max())
    return auc,recall


def patient_metrics(y,p,pid,panel,state):
    ids=np.full(len(y),pid);reg=metrics(y,p,ids,panel);sub=subgroup_metrics(y,p,ids,panel,state)
    result={}
    for j,m in enumerate(panel):
        row={**reg[m]['patients'][pid],**sub[m][pid]}
        high=y[:,j]>state['thresholds'][j];diff=p[:,j]-y[:,j]
        row['high_bias']=float(diff[high].mean()) if high.any() else None
        row['low_bias']=float(diff[~high].mean()) if (~high).any() else None
        row['roc_auc'],row['diagnostic_recall_at_fpr_0.01']=roc_diagnostics(high,p[:,j])
        result[m]=row
    return result


def bootstrap(values,draws):
    values=np.asarray(values,float);defined=np.isfinite(values)
    low=high=None
    if defined.sum()>=2:
        samples=values[draws];valid=np.isfinite(samples);n=valid.sum(1)
        means=np.divide(np.where(valid,samples,0).sum(1),n,out=np.full(len(samples),np.nan),where=n>0)
        means=means[np.isfinite(means)]
        low,high=map(float,np.percentile(means,[2.5,97.5]))
    return {'patient_mean':float(values[defined].mean()) if defined.any() else None,
            'defined_patients':int(defined.sum()),'total_patients':len(values),'ci95_low':low,'ci95_high':high}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_protocols(runs,pairs,folds):
    protocols={};audits=[]
    for label,root in runs.items():
        protocols[label]=[]
        for i in range(len(folds)):
            training=root/f'fold_{i:02d}'/'training'
            invocation=json.loads((training/'invocation.json').read_text())
            scaler=json.loads((training/'target_scaler.json').read_text()) if (training/'target_scaler.json').exists() else None
            config=json.loads((training/'config.json').read_text()) if (training/'config.json').exists() else None
            protocols[label].append({'args':invocation['args'],'scaler':scaler,'config':config,
                                     'source_sha256':invocation.get('source_sha256',{})})
    for pair in pairs:
        b,c=pair['baseline'],pair['candidate'];different=[];code_differences=[]
        for i,(left,right) in enumerate(zip(protocols[b],protocols[c])):
            for field in ['args','config','scaler']:
                a=left[field].copy() if isinstance(left[field],dict) else left[field]
                z=right[field].copy() if isinstance(right[field],dict) else right[field]
                if field in ['args','config']:
                    for v in [a,z]:
                        if isinstance(v,dict):
                            for key in ['out','split','mode']:v.pop(key,None)
                if a!=z:different.append({'fold':i,'field':field})
            code=['train.py','model.py','fds.py','loss.py','adaptive_loss.py','data.py','expression.py']
            for name in code:
                if left['source_sha256'].get(name)!=right['source_sha256'].get(name):code_differences.append({'fold':i,'file':name})
        if pair.get('matched_settings') and different:
            raise ValueError(f'Claimed matched pair settings differ: {b} vs {c}: {different}')
        audits.append({**pair,'setting_differences':different,'source_differences':code_differences,
                       'note':'Different source hashes require review before attributing an isolated method effect.'})
    return {'protocols_by_fold':protocols,'pair_audits':audits}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',default='/scratch/groups/emmalu/hande_annotation')
    parser.add_argument('--runs-json',help='Explicit plan {runs:{label:path},threshold_reference:label,pairs:[...]}')
    parser.add_argument('--out',required=True);parser.add_argument('--bootstrap-replicates',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--plan-only',action='store_true')
    a=parser.parse_args()
    if a.bootstrap_replicates<1 or a.seed<0:raise ValueError('Invalid bootstrap settings')
    plan=json.loads(Path(a.runs_json).read_text()) if a.runs_json else default_plan(a.base_dir)
    runs={k:Path(v).resolve() for k,v in plan['runs'].items()};ref=plan['threshold_reference']
    if ref not in runs or len(runs)<2:raise ValueError('Need at least two runs and a threshold reference')
    for pair in plan['pairs']:
        if pair['baseline'] not in runs or pair['candidate'] not in runs:raise ValueError('Unknown pair label')
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(out)
    for root in runs.values():
        if out==root or root in out.parents or out in root.parents:raise ValueError('Output overlaps input run')
    indices={k:index_run(v) for k,v in runs.items()}
    folds,inventory,reference_paths=indices[ref]
    for label,(f,inv,paths) in indices.items():
        if f!=folds or inv!=inventory:raise ValueError(f'Patient folds/sample rows differ: {label}')
    out.mkdir(parents=True);snapshot(out,vars(a));write_json(out/'resolved_runs.json',plan)
    if a.plan_only:
        print(f'Validated run inventories/folds; plan saved to {out}');return
    write_json(out/'protocol_audit.json',audit_protocols(runs,plan['pairs'],folds))
    states=[]
    for i,f in enumerate(folds):
        train=runs[ref]/f'fold_{i:02d}'/'training'
        # Verify declared threshold origin partition against the shared reviewed fold.
        split=json.loads((train/'invocation.json').read_text())['args']['split']
        saved_split=json.loads(Path(split).read_text()) if Path(split).exists() else json.loads((runs[ref]/f'fold_{i:02d}/split.json').read_text())
        if saved_split!=f:raise ValueError('Threshold reference split differs from shared fold')
        states.append(json.loads((train/'expression_state.json').read_text()))
    write_json(out/'common_expression_states.json',states)
    records={k:{} for k in runs};pending={};panel=None;counts=[]
    signature=('panel','patient_id','shape','outer_fold','encoder','revision','checkpoint_sha256','transform','he_mpp','input_patch','target_block','selection_policy')
    for sid in sorted(inventory):
        m,xy,y,refpred=load_sample(reference_paths[sid]);pid=m['patient_id'];fold=m['outer_fold']
        if len(y)!=inventory[sid][1]:raise ValueError('Inventory row count mismatch')
        if panel is not None and panel!=m['panel']:raise ValueError('Panel order differs')
        panel=m['panel'];state=states[fold]
        if state['panel']!=panel:raise ValueError('Reference threshold panel differs')
        counts.append({'sample_id':sid,'patient_id':pid,'rows':len(y),'outer_fold':fold})
        for label in runs:
            mm,xx,yy,pred=(m,xy,y,refpred) if label==ref else load_sample(indices[label][2][sid])
            if any(mm.get(k)!=m.get(k) for k in signature) or not np.array_equal(xx,xy) or not np.array_equal(yy,y):
                raise ValueError(f'Unmatched targets/coordinates/provenance: {label}/{sid}')
            group=pending.setdefault(pid,{}).setdefault(label,[[],[]]);group[0].append(y);group[1].append(pred)
        # Score after all samples belonging to this patient have been collected.
        if sum(v['patient_id']==pid for v in counts)==sum(v[0]==pid for v in inventory.values()):
            for label,(ys,ps) in pending.pop(pid).items():
                records[label][pid]=patient_metrics(np.concatenate(ys),np.concatenate(ps),pid,panel,state)
        print(f'Validated and evaluated {sid}',flush=True)
    if pending:raise ValueError('Unscored patients')
    patients=sorted(records[ref]);draws=np.random.default_rng(a.seed).integers(0,len(patients),size=(a.bootstrap_replicates,len(patients)))
    keys=(*METRIC_KEYS,*SUBKEYS,*EXTRA);summaries={};rows=[]
    for label,by_patient in records.items():
        summaries[label]={}
        for marker in panel:
            summaries[label][marker]={}
            for key in keys:
                vals=[np.nan if by_patient[pid][marker][key] is None else by_patient[pid][marker][key] for pid in patients]
                item=bootstrap(vals,draws);summaries[label][marker][key]=item
                rows.append({'run':label,'marker':marker,'metric':key,**item})
    paired=[]
    for pair in plan['pairs']:
        b,c=pair['baseline'],pair['candidate']
        for marker in panel:
            for key in keys:
                values=[];wins=ties=0
                for pid in patients:
                    bv,cv=records[b][pid][marker][key],records[c][pid][marker][key]
                    values.append(np.nan if bv is None or cv is None else cv-bv)
                    if bv is not None and cv is not None:
                        if cv==bv:ties+=1
                        elif (key in LOWER and cv<bv) or (key in HIGHER and cv>bv):wins+=1
                paired.append({'baseline':b,'candidate':c,'interpretation':pair['interpretation'],'marker':marker,'metric':key,
                    **bootstrap(values,draws),'delta_definition':'candidate minus baseline',
                    'patient_wins':wins if key in LOWER|HIGHER else None,'patient_ties':ties})
    write_json(out/'patient_metrics.json',records);write_json(out/'summary.json',summaries)
    write_json(out/'paired_differences.json',paired);write_csv(out/'summary.csv',rows);write_csv(out/'paired_differences.csv',paired)
    write_json(out/'inventory.json',counts)
    lines=['# Harmonized quantitative OOF evaluation','',
      'All models evaluated at identical measured locations; equal patient means. Errors are in original intensity units.',
      'High/low expression uses ONE shared training-derived threshold set per fold, copied from '+ref+'.',
      'Historical cross-architecture comparisons are descriptive; matched ablations should be interpreted within their family.','']
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    selected=[m for m in ['CAV','CD8','CD138','COLIV','POSTN','VIM'] if m in panel]
    for marker in selected:
        lines+=['## '+marker,'','| Run | MAE | Pearson | Spearman | Bias | High MAE | Low MAE | AP | Recall | FPR |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in runs:
            items=summaries[label][marker]
            fields=['mae','pearson','spearman','mean_bias','positive_mae','negative_mae','average_precision','recall','false_positive_rate']
            lines.append('| '+label+' | '+' | '.join(fmt(items[k]['patient_mean']) for k in fields)+' |')
        lines.append('')
    lines+=['## Paired comparisons','',
      'Deltas below are candidate minus baseline. Negative MAE deltas favor the candidate; positive correlation/AP deltas favor the candidate.',
      'The high-MAE interval is a descriptive paired patient-bootstrap CI; it is not a multiple-testing-adjusted significance result.','']
    for pair in plan['pairs']:
        b,c=pair['baseline'],pair['candidate']
        lines += ['### '+b+' → '+c,'',pair['interpretation'],'',
                  '| Marker | Δ MAE | Δ high MAE [95% CI] | Δ low MAE | Δ Pearson | Δ AP | High-MAE patient wins / defined |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for marker in selected:
            d={r['metric']:r for r in paired if r['baseline']==b and r['candidate']==c and r['marker']==marker}
            h=d['positive_mae']
            interval=fmt(h['patient_mean'])+' ['+fmt(h['ci95_low'])+', '+fmt(h['ci95_high'])+']'
            lines.append('| '+marker+' | '+fmt(d['mae']['patient_mean'])+' | '+interval+' | '+
                         fmt(d['negative_mae']['patient_mean'])+' | '+fmt(d['pearson']['patient_mean'])+' | '+
                         fmt(d['average_precision']['patient_mean'])+' | '+str(h['patient_wins'])+' / '+str(h['defined_patients'])+' |')
        lines.append('')
    lines+=['## Interpretation limits','',
       '- Percentile high expression is not biological positivity; undefined values remain null with counts.',
       '- Paired differences use the SAME defined patients in each comparison, not differences between two independently averaged summaries.',
       '- Bootstrap CIs are descriptive conditional-on-prediction patient resampling; shared CV training data and model-selection uncertainty are not fully captured.',
       '- No multiple-testing-adjusted significance claim or universal best-model ranking is made.',
       '- diagnostic_recall_at_fpr_0.01 uses the outer ROC curve; it is NOT an inner-validation-selected or deployable operating threshold.',
       '- Ground-truth labels are used only for evaluation; no model, threshold or calibration was fitted on outer patients.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write_json(out/'manifest.json',{'complete':True,'runs':list(runs),'patients':len(patients),'markers':len(panel),
        'locations':sum(v['rows'] for v in counts),'threshold_reference':ref,'paired_bootstrap_seed':a.seed})
    print(f'Complete: {out}/REPORT.md')

if __name__=='__main__':main()
