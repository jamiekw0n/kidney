"""Plan/submit one-sample Slurm tasks, run isolated filtering, and collect completed outputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from audit import snapshot,write_json,event
from prepare import discover
from raw_safety import guard_output


def make_plan(a):
    rows=discover(a.base_dir)
    requested=set(a.samples or [])
    if requested-{r['sample_id'] for r in rows}: raise ValueError('Requested samples missing from eligible folder discovery')
    tasks=[];excluded=[]
    for index,row in enumerate(rows):
        if requested and row['sample_id'] not in requested:continue
        if row['error']:excluded.append(row)
        else:tasks.append({**row,'seed_index':index,'task_id':len(tasks)})
    if not tasks:raise ValueError('No valid pairs to submit')
    root=Path(a.out).resolve();guard_output(root,rows)
    if root.exists():raise FileExistsError('Use a new array run directory')
    root.mkdir(parents=True);(root/'logs').mkdir();(root/'tasks').mkdir();snapshot(root,vars(a))
    panel=json.loads(Path(a.panel).read_text())
    if not isinstance(panel,list) or not panel or len(set(panel))!=len(panel):raise ValueError('Invalid panel')
    write_json(root/'panel.json',panel)
    code=root/'source_snapshot'
    shutil.copy2(Path(__file__).parent/'slurm/filter_sample_array.sbatch',code/'filter_sample_array.sbatch')
    config={k:getattr(a,k) for k in ('base_dir','mpp','stride','max_per_sample','min_context','min_center','he_cache_gb','seed','valid_mask_name','assume_all_valid')}
    write_json(root/'plan.json',{'config':config,'tasks':tasks,'excluded':excluded,'code':str(code)})
    command=['sbatch','--parsable','--partition='+a.partition,'--mem='+a.mem,'--time='+a.time,
        f'--array=0-{len(tasks)-1}%{a.concurrency}',f'--output={root}/logs/%A_%a.out',f'--error={root}/logs/%A_%a.err',
        str(code/'filter_sample_array.sbatch'),str(root)]
    write_json(root/'submission_command.json',command)
    print(json.dumps({'run':str(root),'tasks':len(tasks),'excluded':len(excluded),'command':command},indent=2),flush=True)
    if a.submit:
        result=subprocess.run(command,text=True,capture_output=True)
        write_json(root/'submission.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr)
        job=result.stdout.strip().split(';')[0]
        event(root,'submitted',array_job_id=job)
        print('Submitted array job '+job,flush=True)


def run_task(a):
    root=Path(a.run).resolve();plan=json.loads((root/'plan.json').read_text())
    if not 0<=a.task_id<len(plan['tasks']):raise ValueError('Task ID outside plan')
    task=plan['tasks'][a.task_id];config=plan['config']
    current={r['sample_id']:r for r in discover(config['base_dir'])}.get(task['sample_id'])
    if current is None or current['error'] or any(current[k]!=task[k] for k in ('he','mif','folder')):
        raise ValueError('Discovered source paths changed since planning')
    directory=root/'tasks'/f'{a.task_id:04d}'
    if directory.exists():raise FileExistsError('Task output exists; create a new retry plan instead of overwriting')
    directory.mkdir();write_json(directory/'task.json',task)
    event(directory,'start',sample_id=task['sample_id'])
    command=[sys.executable,'-u',str(Path(plan['code'])/'cohort_prepare.py'),
        '--base-dir',config['base_dir'],'--panel',str(root/'panel.json'),'--out',str(directory/'prepared'),
        '--samples',task['sample_id'],'--mpp',str(config['mpp']),'--samples-as-patients',
        '--seed-index-offset',str(task['seed_index']),'--progress-every','100']
    for flag,key in [('stride','stride'),('max-per-sample','max_per_sample'),('min-context','min_context'),
                     ('min-center','min_center'),('he-cache-gb','he_cache_gb'),('seed','seed')]:
        command.extend(['--'+flag,str(config[key])])
    command.extend(['--valid-mask-name',config['valid_mask_name']] if config['valid_mask_name'] else ['--assume-all-valid'])
    write_json(directory/'command.json',command)
    print(f'Task {a.task_id}: {task["sample_id"]}',flush=True)
    started=time.monotonic()
    code=subprocess.run(command).returncode
    write_json(directory/'status.json',{'sample_id':task['sample_id'],'child_returncode':code,'elapsed_seconds':time.monotonic()-started,
        'complete':code==0,'signal':-code if code<0 else None})
    event(directory,'end',returncode=code)
    return int(code!=0)


def collect(a):
    from validate import validate
    root=Path(a.run).resolve();plan=json.loads((root/'plan.json').read_text())
    report=[];completed=[]
    for task in plan['tasks']:
        directory=root/'tasks'/f'{task["task_id"]:04d}'
        sample=directory/'prepared'/task['sample_id']
        status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else None
        state='not_started' if not directory.exists() else 'running_or_interrupted'
        error=None
        if status is not None:state='failed' if status['child_returncode'] else 'complete'
        if state=='complete':
            try:
                validate(sample);completed.append(sample)
            except Exception as exc:state='invalid_completed_output';error=str(exc)
        report.append({'sample_id':task['sample_id'],'task_id':task['task_id'],'state':state,'status':status,'error':error})
    write_json(root/'array_report.json',{'tasks':report,'discovery_exclusions':plan['excluded']})
    print(json.dumps({'complete':len(completed),'total':len(report),'states':report},indent=2))
    if a.assemble:
        out=Path(a.assemble).resolve();guard_output(out,discover(plan['config']['base_dir']))
        if out.exists():raise FileExistsError('Assembly output exists')
        if not completed:raise ValueError('No completed outputs')
        if len(completed)!=len(report) and not a.allow_partial:
            raise ValueError('Some tasks incomplete/failed; review report or explicitly --allow-partial')
        out.mkdir(parents=True)
        for sample in completed:(out/sample.name).symlink_to(sample,target_is_directory=True)
        write_json(out/'cohort_report.json',[{'sample_id':s.name,'status':'complete','source':str(s)} for s in completed])
        write_json(out/'array_report.json',{'tasks':report,'discovery_exclusions':plan['excluded']})
        write_json(out/'cohort_status.json',{'finished':True,'completed':len(completed),'failed_or_excluded':len(report)-len(completed),
            'discovery_exclusions':len(plan['excluded']),'explicit_partial_assembly':a.allow_partial})
        print('Assembled prepared root: '+str(out))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    plan=sub.add_parser('plan')
    plan.add_argument('--base-dir',required=True);plan.add_argument('--panel',required=True);plan.add_argument('--out',required=True)
    plan.add_argument('--samples',nargs='+');plan.add_argument('--mpp',type=float,default=0.5073077046562237)
    # Explicitly acknowledge existing sample=patient convention.
    plan.add_argument('--samples-as-patients',action='store_true',required=True)
    support=plan.add_mutually_exclusive_group(required=True)
    support.add_argument('--assume-all-valid',action='store_true');support.add_argument('--valid-mask-name')
    for name,value in [('stride',128),('max-per-sample',5000),('seed',42),('concurrency',2)]:plan.add_argument('--'+name,type=int,default=value)
    plan.add_argument('--min-context',type=float,default=0.5);plan.add_argument('--min-center',type=float,default=0.5)
    plan.add_argument('--he-cache-gb',type=float,default=8);plan.add_argument('--partition',default='emmalu')
    plan.add_argument('--mem',default='64G');plan.add_argument('--time',default='06:00:00');plan.add_argument('--submit',action='store_true')
    task=sub.add_parser('task');task.add_argument('--run',required=True);task.add_argument('--task-id',type=int,required=True)
    c=sub.add_parser('collect');c.add_argument('--run',required=True);c.add_argument('--assemble');c.add_argument('--allow-partial',action='store_true')
    a=p.parse_args()
    if a.mode=='plan':
        if a.concurrency<1 or a.max_per_sample<1 or a.stride<8 or a.stride%8 or a.seed<0 or a.mpp<=0 or a.he_cache_gb<0 or not 0<=a.min_context<=1 or not 0<=a.min_center<=1:
            p.error('Invalid filtering or concurrency setting')
        make_plan(a)
    elif a.mode=='task':return run_task(a)
    else:collect(a)

if __name__=='__main__':raise SystemExit(main())
