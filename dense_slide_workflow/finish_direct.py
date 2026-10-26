"""Wait for the direct extraction launcher, then train/render on two allocated GPUs; no Slurm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--launcher-log',required=True)
    p.add_argument('--sample',default='K2140231');p.add_argument('--mode',choices=['weighted','weighted_fds'],default='weighted')
    p.add_argument('--markers',nargs='+',default=['ALL']);p.add_argument('--wait-hours',type=float,default=36)
    a=p.parse_args();run=Path(a.run).resolve();code=run/'source';launcher=Path(a.launcher_log).resolve()
    if not launcher.is_file():raise FileNotFoundError('Specify the existing extraction LAUNCH_LOG: '+str(launcher))
    if a.wait_hours<=0:raise ValueError('Positive wait limit required')
    for name in ('workflow.py','whole_slide.py'):
        if not (code/name).is_file():raise FileNotFoundError(code/name)
    # Fixed downstream destinations deliberately reject duplicate/partial launches.
    for dest in (run/'cv/aggregate',run/'slides'/f'{a.sample}_direct_v1',run/'slides'/f'{a.sample}_direct_v1_figures'):
        if dest.exists():raise FileExistsError(f'Existing downstream output: {dest}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');logs=run/f'direct_finish_{stamp}';logs.mkdir()
    (logs/'invocation.json').write_text(json.dumps(vars(a),indent=2))
    (logs/'finish_source.py').write_bytes(Path(__file__).read_bytes())
    print(f'Waiting for: {launcher}\nStage logs: {logs}',flush=True)
    start=time.monotonic()
    while True:
        text=launcher.read_text(errors='replace')
        if 'SUCCESS: all patient caches validated.' in text:break
        if 'FAILED:' in text or 'Traceback (most recent call last):' in text:
            raise RuntimeError(f'Extraction launcher failed; inspect {launcher}')
        if time.monotonic()-start>a.wait_hours*3600:raise TimeoutError('Extraction wait limit reached; no downstream jobs started')
        print(f'Waiting for extraction success; elapsed {(time.monotonic()-start)/60:.1f} min',flush=True)
        time.sleep(60)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError('Need two visible allocated GPUs')

    def command(label,args):
        print(f'START {label}',flush=True)
        with (logs/f'{label}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'COMPLETE {label}',flush=True)

    command('collect_cache',[code/'workflow.py','collect-cache','--run',run])
    folds=json.loads((run/'folds.json').read_text())
    for i in range(len(folds)):
        if (run/'cv'/f'part_{i:02d}').exists():raise FileExistsError(f'Existing fold output: part_{i:02d}; refusing overwrite')

    def train_worker(gpu):
        for i in range(gpu,len(folds),2):
            command(f'train_fold_{i:02d}_gpu{gpu}',[code/'workflow.py','train-fold','--run',run,
                    '--task-id',i,'--mode',a.mode,'--device',f'cuda:{gpu}'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_worker,gpu) for gpu in range(2)]
        for f in futures:f.result()
    command('collect_cv',[code/'workflow.py','collect-cv','--run',run])
    slide=run/'slides'/f'{a.sample}_direct_v1'
    command('plan_slide',[code/'whole_slide.py','plan','--run',run,'--sample',a.sample,
                         '--out',slide,'--markers',*a.markers,'--band-pixels',512])
    bands=json.loads((slide/'slide.json').read_text())['bands']
    def infer_worker(gpu):
        for i in range(gpu,bands,2):
            command(f'infer_band_{i:05d}_gpu{gpu}',[code/'whole_slide.py','infer','--slide-plan',slide,
                '--task-id',i,'--device',f'cuda:{gpu}'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(infer_worker,gpu) for gpu in range(2)]
        for f in futures:f.result()
    out=Path(str(slide)+'_figures')
    command('assemble',[code/'whole_slide.py','assemble','--slide-plan',slide,'--out',out])
    (logs/'complete.json').write_text(json.dumps({'complete':True,'gallery':str(out/'index.html')}))
    print(f'ALL COMPLETE: {out}/index.html',flush=True)

if __name__=='__main__':main()
