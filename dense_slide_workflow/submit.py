"""Submit bounded CPU/GPU arrays with stage dependencies; never execute remote work locally."""
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from audit import write_json
from workflow import run_config,collect_cache
from validate import validate


def submit(stage,plan,array=None,dependency=None,mode='weighted'):
    plan=Path(plan).resolve()
    run=Path(json.loads((plan/'slide.json').read_text())['run']) if stage in ('infer','assemble') else plan
    log=run/'logs';log.mkdir(exist_ok=True)
    gpu=stage in ('extract','train-fold','infer')
    mem='64G' if stage in ('prepare','train-fold') else '32G'
    args=['sbatch','--parsable','--job-name=uni2-dense-'+stage,'--mem='+mem,
          '--output='+str(log/(stage+'-%A_%a.out')),'--error='+str(log/(stage+'-%A_%a.err'))]
    if gpu:args+=['--gres=gpu:1']
    if array:args+=['--array='+array]
    if dependency:args+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
    args +=[str(run/'source'/'worker.sbatch'),stage,str(plan),mode]
    result=subprocess.run(args,text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise ValueError(f'Unexpected sbatch result: {result.stdout!r}')
    print(f'{stage}: {job}',flush=True)
    records=run/'submissions';records.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(records/f'{stamp}_{stage}_{job}.json',{'job_id':job,'stage':stage,'command':args,'stderr':result.stderr})
    return job


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['pilot','remaining','cohort','train','slide','pipeline'])
    p.add_argument('--run');p.add_argument('--slide-plan');p.add_argument('--sample',default='K2140231');p.add_argument('--markers',nargs='+',default=['ALL']);p.add_argument('--inference-concurrency',type=int,default=4);p.add_argument('--after',help='Succeeded prerequisite Slurm job ID')
    p.add_argument('--cpu-concurrency',type=int,default=4);p.add_argument('--gpu-concurrency',type=int,default=2)
    p.add_argument('--mode',choices=['weighted','weighted_fds'],default='weighted');a=p.parse_args()
    if min(a.cpu_concurrency,a.gpu_concurrency,a.inference_concurrency)<1:raise ValueError('Positive concurrency required')
    dep='afterok:'+a.after if a.after else None
    if a.stage=='slide':
        if not a.slide_plan:raise ValueError('--slide-plan required')
        s=json.loads((Path(a.slide_plan)/'slide.json').read_text())
        job=submit('infer',a.slide_plan,f'0-{s["bands"]-1}%{a.gpu_concurrency}',dep)
        submit('assemble',a.slide_plan,dependency='afterok:'+job)
        return
    if not a.run:raise ValueError('--run required')
    c=run_config(a.run);n=len(c['tasks'])
    if a.stage=='pipeline':
        # Pilot must have succeeded. Plan priority slide BEFORE any new jobs launch.
        validate(Path(a.run)/'cache'/c['tasks'][0]['sample_id'])
        from whole_slide import plan as plan_slide
        from argparse import Namespace
        slide=Path(a.run)/'slides'/f'{a.sample}_whole_v1'
        plan_slide(Namespace(run=a.run,sample=a.sample,out=str(slide),markers=a.markers,band_pixels=512))
        if n<2:raise ValueError('Pipeline expects at least two samples')
        cpu=submit('prepare',a.run,f'1-{n-1}%{a.cpu_concurrency}',dep)
        features=submit('extract',a.run,f'1-{n-1}%{a.gpu_concurrency}','aftercorr:'+cpu)
        folds=json.loads((Path(a.run)/'folds.json').read_text())
        training=submit('train-fold',a.run,f'0-{len(folds)-1}%{a.gpu_concurrency}','afterok:'+features,a.mode)
        submit('collect-cv',a.run,dependency='afterok:'+training)
        s=json.loads((slide/'slide.json').read_text())
        bands=submit('infer',slide,f'0-{s["bands"]-1}%{a.inference_concurrency}','afterok:'+training)
        submit('assemble',slide,dependency='afterok:'+bands)
        return
    if a.stage in ('pilot','remaining','cohort'):
        if a.stage=='remaining':validate(Path(a.run)/'cache'/c['tasks'][0]['sample_id'])
        start=1 if a.stage=='remaining' else 0;end=0 if a.stage=='pilot' else n-1
        if start>end:raise ValueError('No remaining tasks')
        cpu=submit('prepare',a.run,f'{start}-{end}%{a.cpu_concurrency}',dep)
        # aftercorr releases each extraction as its own preparation task succeeds.
        gpu=submit('extract',a.run,f'{start}-{end}%{a.gpu_concurrency}','aftercorr:'+cpu)
        print('Feature array:',gpu)
    else:
        if not a.after:collect_cache(a.run,persist=False)
        folds=json.loads((Path(a.run)/'folds.json').read_text())
        job=submit('train-fold',a.run,f'0-{len(folds)-1}%{a.gpu_concurrency}',dep,a.mode)
        submit('collect-cv',a.run,dependency='afterok:'+job)

if __name__=='__main__':main()
