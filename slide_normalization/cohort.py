"""Plan independent per-slide CPU tasks, run one, and collect complete derived caches."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
from audit import write_json,snapshot
from data import read_meta,partitions
from validate import validate
from normalization import UNITS
from raw_safety import guard_output
from source_paths import resolve_sources


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--cache-root',required=True);plan.add_argument('--out',required=True)
    plan.add_argument('--reference-folds',required=True)
    task=sub.add_parser('task');task.add_argument('--run',required=True);task.add_argument('--task-id',type=int,required=True)
    collect=sub.add_parser('collect');collect.add_argument('--run',required=True)
    a=p.parse_args()
    if a.command=='plan':
        roots=sorted(v.parent.resolve() for v in Path(a.cache_root).glob('*/manifest.json'))
        if not roots:raise ValueError('No source caches')
        from cross_validate import reviewed_folds
        folds=reviewed_folds(a.reference_folds,roots)
        out=Path(a.out).resolve()
        if out.exists():raise FileExistsError(out)
        if out==Path(a.cache_root).resolve() or Path(a.cache_root).resolve() in out.parents or out in Path(a.cache_root).resolve().parents:
            raise ValueError('Derived cohort overlaps source cache root')
        for source in roots:
            m=read_meta(source)
            paths,_=resolve_sources(source)
            m={**m,**paths}
            for key in ('he','mif'):
                if key in m:guard_output(out,[{'folder':str(Path(m[key]).resolve().parent)}])
        out.mkdir(parents=True);snapshot(out,vars(a));(out/'cache').mkdir()
        inventory=[]
        seen=set()
        for i,root in enumerate(roots):
            report=validate(root);m=read_meta(root)
            if m['sample_id'] in seen or Path(m['sample_id']).name!=m['sample_id']:raise ValueError('Duplicate or unsafe sample_id')
            seen.add(m['sample_id'])
            if m.get('target_units')==UNITS:raise ValueError('Source is already normalized')
            inventory.append({'task_id':i,'source_cache':str(root),'sample_id':m['sample_id'],'patient_id':m['patient_id'],'rows':report['rows']})
        write_json(out/'tasks.json',inventory);write_json(out/'folds.json',folds)
        write_json(out/'inventory.json',[{'sample_id':r['sample_id'],'patient_id':r['patient_id'],'rows':r['rows']} for r in inventory])
        print(f'Tasks: {len(inventory)}; Slurm array: 0-{len(inventory)-1}; new cache root: {out}/cache')
    else:
        run=Path(a.run).resolve();tasks=json.loads((run/'tasks.json').read_text())
        if a.command=='task':
            if not 0<=a.task_id<len(tasks):raise ValueError('Task index outside plan')
            t=tasks[a.task_id]
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('normalize_sample.py')),'--source-cache',t['source_cache'],
                 '--out',str(run/'cache'/t['sample_id'])]
            subprocess.run(cmd,check=True)
        else:
            records=[];quality=[]
            for task in tasks:
                path=run/'cache'/task['sample_id']
                try:
                    report=validate(path);meta=read_meta(path)
                    if meta.get('target_units')!=UNITS or report['rows']!=task['rows'] or meta['patient_id']!=task['patient_id']:
                        raise ValueError('Normalized inventory mismatch')
                    norm=json.loads((path/'normalization.json').read_text())
                    quality.extend({'sample_id':task['sample_id'],**m} for m in norm['markers'])
                    records.append({'sample_id':task['sample_id'],'state':'complete',
                        'fallback_markers':[m['marker'] for m in norm['markers'] if m['fallback_dtype_max']]})
                except (OSError,ValueError,KeyError) as exc:
                    records.append({'sample_id':task['sample_id'],'state':'incomplete','error':str(exc)})
            write_json(run/'collection_report.json',records)
            if quality:
                with (run/'normalization_summary.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(quality)
            print(json.dumps({'complete':sum(v['state']=='complete' for v in records),'total':len(tasks),'states':records},indent=2))
            if any(v['state']!='complete' for v in records):raise SystemExit(1)
            roots=[run/'cache'/t['sample_id'] for t in tasks]
            for f in json.loads((run/'folds.json').read_text()):partitions(roots,f)
            write_json(run/'cache/normalization_cohort_manifest.json',{'complete':True,'samples':len(tasks),'target_units':UNITS,
                       'source_plan':str(run/'tasks.json')})

if __name__=='__main__':main()
