"""Extract every complete selected sample, recording commands, logs, and exclusions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from audit import snapshot,event,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--revision',required=True)
    p.add_argument('--he-cache-gb',type=float,default=0)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--device',default='cuda')
    a=p.parse_args(); out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    prepared=Path(a.prepared_root)
    roots=sorted(path.parent for path in prepared.glob('*/manifest.json'))
    if not roots: p.error('No complete sample manifests found')
    out.mkdir(parents=True); snapshot(out,vars(a))
    if (prepared/'cohort_report.json').exists():
        write_json(out/'preparation_cohort_report.json',json.loads((prepared/'cohort_report.json').read_text()))
    report=[]
    for root in roots:
        command=[sys.executable,'-u',str(Path(__file__).with_name('extract.py')),
            '--prepared',str(root),'--out',str(out/root.name),'--checkpoint',a.checkpoint,
            '--revision',a.revision,'--he-cache-gb',str(a.he_cache_gb),'--batch-size',str(a.batch_size),'--device',a.device]
        event(out,'extraction_start',sample_id=root.name,command=command)
        with (out/(root.name+'.log')).open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        record={'sample_id':root.name,'returncode':code,'log':root.name+'.log'}
        report.append(record); event(out,'extraction_end',**record); print(json.dumps(record),flush=True)
        write_json(out/'extraction_report.json',report)
    return int(any(r['returncode'] for r in report))

if __name__=='__main__': raise SystemExit(main())
