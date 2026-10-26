"""Reuse completed dense-training heads to render more slides on two allocated GPUs. No retraining/Slurm."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--samples',nargs='+',required=True)
    p.add_argument('--markers',nargs='+',default=['ALL']);p.add_argument('--tag',required=True)
    a=p.parse_args();run=Path(a.run).resolve();code=run/'source'
    if Path(a.tag).name!=a.tag or not a.tag or a.tag in ('.','..'):raise ValueError('Tag must be a simple new name')
    if len(a.samples)!=len(set(a.samples)):raise ValueError('Duplicate sample')
    cfg=json.loads((run/'experiment.json').read_text());available={t['sample_id'] for t in cfg['tasks']}
    if set(a.samples)-available:raise ValueError('Unknown sample IDs')
    if not json.loads((run/'cv/aggregate/manifest.json').read_text()).get('complete'):raise ValueError('Need completed CV')
    for sample in a.samples:
        for out in (run/'slides'/f'{sample}_{a.tag}',run/'slides'/f'{sample}_{a.tag}_figures'):
            if out.exists():raise FileExistsError(out)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError('Need two visible allocated GPUs')
    logs=run/('more_slides_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));logs.mkdir()
    print(f'Logs: {logs}',flush=True)
    (logs/'invocation.json').write_text(json.dumps(vars(a),indent=2))
    def command(label,args):
        print(f'START {label}',flush=True)
        with (logs/f'{label}.log').open('x') as f:subprocess.run([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
        print(f'COMPLETE {label}',flush=True)
    for sample in a.samples:
        slide=run/'slides'/f'{sample}_{a.tag}'
        command(sample+'_plan',[code/'whole_slide.py','plan','--run',run,'--sample',sample,'--out',slide,'--markers',*a.markers,'--band-pixels',512])
        bands=json.loads((slide/'slide.json').read_text())['bands']
        def worker(gpu):
            for i in range(gpu,bands,2):
                command(f'{sample}_band_{i:05d}_gpu{gpu}',[code/'whole_slide.py','infer','--slide-plan',slide,'--task-id',i,'--device',f'cuda:{gpu}'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,gpu) for gpu in range(2)]
            for f in futures:f.result()
        out=Path(str(slide)+'_figures')
        command(sample+'_assemble',[code/'whole_slide.py','assemble','--slide-plan',slide,'--out',out])
        print(f'GALLERY {out}/index.html',flush=True)
    (logs/'complete.json').write_text(json.dumps({'complete':True,'samples':a.samples}))
    print('ALL REQUESTED SLIDES COMPLETE',flush=True)

if __name__=='__main__':main()
