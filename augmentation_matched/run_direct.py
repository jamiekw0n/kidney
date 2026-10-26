"""Bounded two-GPU code-server launcher; persistent encoders; no Slurm calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys
from pathlib import Path
from pipeline import config,VARIANTS,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--stage',choices=['pilot','features','train','all'],default='all');p.add_argument('--resume-complete',action='store_true')
    p.add_argument('--device-count',type=int,choices=[1,2],default=2)
    p.add_argument('--encoder-batch-size',type=int,default=16,help='Total images per encoder forward, two views/location')
    a=p.parse_args()
    if a.encoder_batch_size<2 or a.encoder_batch_size%2:p.error('Encoder batch size must be even and >=2')
    plan=Path(a.plan).resolve();c=config(plan);code=plan/'source'
    for name,digest in c['source_hashes'].items():
        if sha(code/name)!=digest:raise ValueError('Recorded source snapshot changed: '+name)
    with (plan/'direct.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another augmentation launcher already holds this plan lock')
        import torch
        if torch.cuda.device_count()<a.device_count:raise RuntimeError('Not enough visible allocated GPUs')
        logs=plan/('logs_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));logs.mkdir()
        print(f'Stage logs: {logs}',flush=True);env=os.environ.copy()
        env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
        def command(label,args):
            print('START '+label,flush=True)
            try:
                with (logs/f'{label}.log').open('x') as f:subprocess.run([sys.executable,'-u',str(code/'pipeline.py'),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
            except subprocess.CalledProcessError:
                print(f'FAILED {label}; inspect {logs}/{label}.log. Other independent worker may still be finishing.',flush=True)
                raise
            print('COMPLETE '+label,flush=True)
        if a.stage in ('pilot','features','all'):
            def features(gpu):
                ids=[0] if a.stage=='pilot' else list(range(gpu,len(c['tasks']),a.device_count))
                if a.stage=='pilot' and gpu>0:return
                if not ids:return
                args=['extract','--plan',plan,'--task-ids',*ids,'--device',f'cuda:{gpu}','--batch-size',a.encoder_batch_size]
                if a.resume_complete:args+=['--resume-complete']
                command(f'features_gpu{gpu}',args)
            with ThreadPoolExecutor(max_workers=a.device_count) as pool:
                futures=[pool.submit(features,gpu) for gpu in range(a.device_count)]
                for f in futures:f.result()
            if a.stage=='pilot':
                print('PILOT COMPLETE: inspect views/'+c['tasks'][0]['sample_id']+'/preview.png and feature logs',flush=True);return
            command('verify_all',['verify','--plan',plan])
        if a.stage in ('train','all'):
            if a.stage=='train':command('verify_all',['verify','--plan',plan])
            jobs=[(variant,fold) for fold in range(len(c['folds'])) for variant in VARIANTS]
            def train(gpu):
                for variant,fold in jobs[gpu::a.device_count]:
                    out=plan/'runs'/variant/f'fold_{fold:02d}'
                    if out.exists():
                        if a.resume_complete and (out/'manifest.json').exists():
                            m=json.loads((out/'manifest.json').read_text())
                            if m.get('complete') and m.get('variant')==variant and m.get('fold')==fold and m.get('plan_fingerprint')==c['fingerprint'] and (out/'training/best.pt').is_file() and (out/'patient_metrics.json').is_file():
                                print(f'SKIP verified completion marker: {variant} fold{fold}',flush=True);continue
                        raise FileExistsError('Existing/partial training output; preserved: '+str(out))
                    command(f'{variant}_fold{fold:02d}_gpu{gpu}',['train','--plan',plan,'--variant',variant,'--fold',fold,'--device',f'cuda:{gpu}'])
            with ThreadPoolExecutor(max_workers=a.device_count) as pool:
                futures=[pool.submit(train,gpu) for gpu in range(a.device_count)]
                for f in futures:f.result()
            command('collect_results',['collect','--plan',plan])
            print(f'ALL THREE EXPERIMENTS COMPLETE: {plan}/comparison/all_marker_metrics.csv',flush=True)

if __name__=='__main__':main()
