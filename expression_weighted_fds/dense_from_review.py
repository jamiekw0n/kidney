"""Resolve an existing dense review to its inference metadata and reuse its exact ROI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def resolve_reference(path):
    path=Path(path).resolve();seen=set()
    while True:
        if path in seen: raise ValueError('Cycle in dense source references')
        seen.add(path)
        m=json.loads((path/'manifest.json').read_text())
        if not m.get('complete'): raise ValueError('Incomplete dense reference')
        if 'source_dense_root' not in m: break
        target=Path(m['source_dense_root'])
        path=target.resolve() if target.is_absolute() else (path/target).resolve()
    args=json.loads((path/'invocation.json').read_text())['args']
    if m.get('sample_id')!='F03600' or m.get('stride')!=8 or m.get('target_block')!=8:
        raise ValueError('Expected F03600 stride-8 reference')
    if m['roi']!=args['roi']: raise ValueError('Reference ROI provenance mismatch')
    return path,m,args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root',required=True);p.add_argument('--baseline-root',required=True);p.add_argument('--candidate-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--markers',nargs='+',default=['CAV','CD8','CD138','COLIV','POSTN','VIM'])
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    path,m,original=resolve_reference(a.review_root)
    baseline=Path(a.baseline_root).resolve();candidate=Path(a.candidate_root).resolve()
    for run,mode in [(baseline,'weighted'),(candidate,'weighted_fds')]:
        status=json.loads((run/'manifest.json').read_text())
        if not status.get('complete') or status.get('mode')!=mode:
            raise ValueError(f'Expected completed {mode} run: {run}')
        json.loads((run/'primary_summary.json').read_text())
    from verify_pair import verify
    verify(baseline,candidate)
    out=Path(a.out).resolve()
    # Guard review and original dense files even if they are derived rather than raw.
    for source in [Path(a.review_root).resolve(),path,baseline,candidate]:
        if out==source or source in out.parents or out in source.parents:
            raise ValueError('Output must be independent of existing input runs')
    if out.exists():raise FileExistsError(out)
    command=[sys.executable,'-u',str(Path(__file__).with_name('dense_compare.py')),
        '--base-dir',original['base_dir'],'--sample',m['sample_id'],
        '--cache-root',original['cache_root'],'--baseline-root',str(baseline),
        '--candidate-root',str(candidate),'--checkpoint',original['checkpoint'],
        '--out',str(out),'--roi',*[str(v) for v in m['roi']],
        '--markers',*a.markers,'--batch-size',str(a.batch_size),'--device',a.device,
        '--intensity-max',str(original['intensity_max']),'--assume-all-valid']
    print(json.dumps({'reference_dense_root':str(path),'reference_review':a.review_root,
                      'roi':m['roi'],'command':command}),flush=True)
    subprocess.run(command,check=True)
    (out/'reference_review.json').write_text(json.dumps({'review_root':str(Path(a.review_root).resolve()),
        'original_dense_root':str(path),'roi':m['roi'],'sample_id':m['sample_id'],
        'note':'Exact ROI and encoder provenance copied from existing dense run; models are expression-weighted and weighted + FDS. UNI2-h rerun; no dense feature cache existed.'},indent=2))

if __name__=='__main__':main()
