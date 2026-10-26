"""Select completed weighted/weighted_fds runs; never silently choose among repeats."""
import argparse
import json
from pathlib import Path


def resolve(root,baseline=None,candidate=None):
    if bool(baseline)!=bool(candidate):raise ValueError('Provide both explicit run paths or neither')
    selected=[]
    for mode,explicit in [('weighted',baseline),('weighted_fds',candidate)]:
        if explicit:
            choices=[Path(explicit).resolve()]
        else:
            choices=[]
            for path in sorted(Path(root).glob('*/manifest.json')):
                try:meta=json.loads(path.read_text())
                except (OSError,json.JSONDecodeError):continue
                if meta.get('complete') and meta.get('mode')==mode:choices.append(path.parent.resolve())
        if len(choices)!=1:
            raise ValueError(f'Expected one completed {mode} run, found {len(choices)}: {choices}. Pass BOTH run paths explicitly when submitting.')
        meta=json.loads((choices[0]/'manifest.json').read_text())
        if not meta.get('complete') or meta.get('mode')!=mode:raise ValueError(f'Not a completed {mode} run: {choices[0]}')
        selected.append(choices[0])
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--baseline-root');p.add_argument('--candidate-root')
    a=p.parse_args()
    for path in resolve(a.run_root,a.baseline_root,a.candidate_root):print(path)

if __name__=='__main__':main()
