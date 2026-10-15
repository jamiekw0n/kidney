"""Print exact selected tile totals and per-fold epochs from a completed CV inventory."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cv-root',required=True)
    a=p.parse_args();root=Path(a.cv_root)
    rows=json.loads((root/'inventory.json').read_text())
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate samples')
    if any(not isinstance(r['rows'],int) or r['rows']<=0 for r in rows):raise ValueError('Invalid row counts')
    epochs=[]
    for fold in sorted(root.glob('fold_[0-9][0-9]')):
        history=json.loads((fold/'training/history.json').read_text())
        run=json.loads((fold/'training/run.json').read_text())
        epochs.append({'fold':fold.name,'completed_epochs':len(history),'best_epoch':run['best_epoch']})
    print(json.dumps({'total_unique_tiles':sum(r['rows'] for r in rows),'samples':len(rows),
        'patients':len({r['patient_id'] for r in rows}),'per_sample':rows,'epochs':epochs,
        'definition':'Inventory selected locations, counted once, not repeated uses across epochs/folds'},indent=2))

if __name__=='__main__':main()
