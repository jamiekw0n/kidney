"""Merge successful filtering tasks and launch isolated GPU feature extraction per sample."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from audit import snapshot,write_json,event
from data import read_meta
from validate import validate
from raw_safety import guard_output


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_inputs(runs):
    selected={};expected=set();audit=[];excluded=[]
    for run in runs:
        root=Path(run).resolve();plan=json.loads((root/'plan.json').read_text())
        excluded.extend(plan.get('excluded',[]))
        for task in plan['tasks']:
            name=task['sample_id'];expected.add(name)
            directory=root/'tasks'/f'{task["task_id"]:04d}'
            status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else None
            record={'sample_id':name,'run':str(root),'status':status};audit.append(record)
            if status is None or status['child_returncode']!=0:continue
            prepared=directory/'prepared'/name
            validate(prepared);meta=read_meta(prepared)
            if meta.get('diagnostic_only'):raise ValueError('Diagnostic samples cannot enter cohort extraction')
            if name in selected:raise ValueError(f'Duplicate successful sample {name}; supply unambiguous runs')
            selected[name]={'sample_id':name,'patient_id':meta['patient_id'],'prepared':str(prepared),
                'manifest_sha256':digest(prepared/'manifest.json'),'rows':meta['retained']}
    missing=expected-set(selected)
    if missing:raise ValueError(f'Samples still lack successful filtering: {sorted(missing)}')
    reference=None
    for task in selected.values():
        m=read_meta(task['prepared'])
        signature=[m['panel'],m['he_mpp'],m['input_patch'],m['target_block'],m.get('selection_policy')]
        if reference is not None and signature!=reference:raise ValueError('Filtering settings, panel or geometry differ')
        reference=signature
    return [selected[k] for k in sorted(selected)],audit,excluded


def plan_run(a):
    tasks,records,excluded=resolve_inputs(a.filter_runs)
    if a.samples:
        if set(a.samples)-{t['sample_id'] for t in tasks}:raise ValueError('Requested sample absent')
        tasks=[t for t in tasks if t['sample_id'] in a.samples]
    if not tasks:raise ValueError('No valid samples')
    if not re.fullmatch('[0-9a-f]{40}',a.revision):raise ValueError('Full checkpoint commit SHA required')
    if not Path(a.checkpoint).is_file():raise FileNotFoundError(a.checkpoint)
    out=Path(a.out).resolve()
    guard_output(out,[{'folder':str(Path(read_meta(t['prepared'])['he']).parent)} for t in tasks])
    if out.exists():raise FileExistsError('Use a new feature array directory')
    out.mkdir(parents=True);(out/'logs').mkdir();(out/'tasks').mkdir();snapshot(out,vars(a))
    shutil.copy2(Path(__file__).parent/'slurm/feature_sample_array.sbatch',out/'source_snapshot/feature_sample_array.sbatch')
    write_json(out/'filtering_provenance.json',{'attempts':records,'discovery_exclusions':excluded})
    for i,t in enumerate(tasks):t['task_id']=i
    config={'checkpoint':str(Path(a.checkpoint).resolve()),'revision':a.revision,'batch_size':a.batch_size,'he_cache_gb':a.he_cache_gb}
    write_json(out/'plan.json',{'tasks':tasks,'config':config,'code':str(out/'source_snapshot')})
    command=['sbatch','--parsable','--partition='+a.partition,'--mem='+a.mem,'--time='+a.time,
        f'--array=0-{len(tasks)-1}%{a.concurrency}',f'--output={out}/logs/%A_%a.out',f'--error={out}/logs/%A_%a.err',
        str(out/'source_snapshot/feature_sample_array.sbatch'),str(out)]
    write_json(out/'submission_command.json',command)
    print(json.dumps({'run':str(out),'samples':len(tasks),'rows':sum(t['rows'] for t in tasks),'command':command},indent=2),flush=True)
    if a.submit:
        result=subprocess.run(command,text=True,capture_output=True)
        write_json(out/'submission.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr)
        print('Submitted GPU array '+result.stdout.strip(),flush=True)


def task_run(a):
    root=Path(a.run).resolve();plan=json.loads((root/'plan.json').read_text())
    if not 0<=a.task_id<len(plan['tasks']):raise ValueError('Invalid task ID')
    task=plan['tasks'][a.task_id];cfg=plan['config']
    if digest(Path(task['prepared'])/'manifest.json')!=task['manifest_sha256']:raise ValueError('Input manifest changed')
    validate(task['prepared'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in allocated GPU task')
    print('GPU: '+torch.cuda.get_device_name(0),flush=True)
    directory=root/'tasks'/f'{a.task_id:04d}'
    if directory.exists():raise FileExistsError('Task output exists; use new retry plan')
    directory.mkdir();write_json(directory/'task.json',task)
    command=[sys.executable,'-u',str(Path(plan['code'])/'extract.py'),'--prepared',task['prepared'],
        '--out',str(directory/'cache'/task['sample_id']),'--checkpoint',cfg['checkpoint'],'--revision',cfg['revision'],
        '--batch-size',str(cfg['batch_size']),'--he-cache-gb',str(cfg['he_cache_gb']),'--device','cuda']
    write_json(directory/'command.json',command);event(directory,'extraction_start',sample_id=task['sample_id'])
    started=time.monotonic();code=subprocess.run(command).returncode;error=None
    if code==0:
        try:
            checked=validate(directory/'cache'/task['sample_id'])
            if checked['rows']!=task['rows']:raise ValueError('Output row count changed')
        except Exception as exc:code=1;error=str(exc)
    write_json(directory/'status.json',{'sample_id':task['sample_id'],'child_returncode':code,'error':error,
        'complete':code==0,'elapsed_seconds':time.monotonic()-started})
    event(directory,'extraction_end',returncode=code)
    return int(code!=0)


def collect_run(a):
    root=Path(a.run).resolve();plan=json.loads((root/'plan.json').read_text());report=[];complete=[]
    for task in plan['tasks']:
        directory=root/'tasks'/f'{task["task_id"]:04d}'
        status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else None
        state='not_started' if not directory.exists() else 'running_or_interrupted';error=None
        if status is not None:state='complete' if status['complete'] else 'failed'
        cache=directory/'cache'/task['sample_id']
        if state=='complete':
            try:
                validate(cache);complete.append(cache)
            except Exception as exc:state='invalid';error=str(exc)
        report.append({'sample_id':task['sample_id'],'task_id':task['task_id'],'state':state,'status':status,'error':error})
    write_json(root/'feature_array_report.json',report)
    print(json.dumps({'complete':len(complete),'total':len(report),'states':report},indent=2))
    if a.assemble:
        if len(complete)!=len(report):raise ValueError('All feature tasks must pass before assembly')
        out=Path(a.assemble).resolve()
        guard_output(out,[{'folder':str(Path(read_meta(t['prepared'])['he']).parent)} for t in plan['tasks']])
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        for cache in complete:(out/cache.name).symlink_to(cache,target_is_directory=True)
        write_json(out/'extraction_report.json',report)
        write_json(out/'preparation_cohort_report.json',json.loads((root/'filtering_provenance.json').read_text()))
        write_json(out/'assembly.json',{'source_array':str(root),'samples':len(complete),'source_paths':[str(p) for p in complete]})
        print('CV-ready cache root: '+str(out))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('plan');p.add_argument('--filter-runs',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--revision',required=True);p.add_argument('--samples',nargs='+')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--he-cache-gb',type=float,default=8)
    p.add_argument('--concurrency',type=int,default=2);p.add_argument('--partition',default='emmalu')
    p.add_argument('--mem',default='64G');p.add_argument('--time',default='04:00:00');p.add_argument('--submit',action='store_true')
    t=sub.add_parser('task');t.add_argument('--run',required=True);t.add_argument('--task-id',type=int,required=True)
    c=sub.add_parser('collect');c.add_argument('--run',required=True);c.add_argument('--assemble')
    a=parser.parse_args()
    if a.mode=='plan':
        if min(a.batch_size,a.concurrency)<1 or a.he_cache_gb<0:parser.error('Invalid batch/concurrency/cache budget')
        plan_run(a)
    elif a.mode=='task':return task_run(a)
    else:collect_run(a)

if __name__=='__main__':raise SystemExit(main())
