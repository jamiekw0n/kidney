"""Submit CPU array sized from an existing plan; never launches training automatically."""
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--max-concurrent',type=int,default=4)
    a=p.parse_args();root=Path(a.run).resolve();tasks=json.loads((root/'tasks.json').read_text())
    if not tasks or a.max_concurrent<1:raise ValueError('Need nonempty plan and positive concurrency')
    command=['sbatch','--parsable',f'--array=0-{len(tasks)-1}%{a.max_concurrent}',str(Path(__file__).with_name('normalize_array.sbatch').resolve()),str(root)]
    # This stage performs no training and no feature extraction.
    print('Submitting:',json.dumps(command),flush=True)
    completed=subprocess.run(command,check=True,capture_output=True,text=True)
    print('Slurm array:',completed.stdout.strip(),flush=True)
    directory=root/'submissions';directory.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (directory/(stamp+'.json')).write_text(json.dumps({'command':command,'slurm_response':completed.stdout.strip()},indent=2))

if __name__=='__main__':main()
