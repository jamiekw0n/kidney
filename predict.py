"""Export coordinate-aligned predictions from one or more trained heads."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import read_meta, scale
from train import make_head, predict, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--heads', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--evaluate', action='store_true', help='Score only patients in every head test split')
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    root, out = Path(a.cache), Path(a.out)
    from validate import validate
    validate(root, require_targets=a.evaluate)
    meta = read_meta(root)
    import torch
    if out.exists():
        raise FileExistsError(out)
    models = []
    for path in a.heads:
        ckpt = torch.load(path, map_location=a.device, weights_only=True)
        if ckpt['panel'] != meta['panel'] or any(meta[k] != v for k,v in ckpt['cache_signature'].items()):
            raise ValueError('Checkpoint and cache provenance/panel differ')
        if a.evaluate and meta['patient_id'] not in ckpt['split']['test']:
            raise ValueError('Evaluation requires a held-out test patient for every head')
        model = make_head(torch, len(meta['panel']), ckpt['hidden']).to(a.device)
        model.load_state_dict(ckpt['state_dict'])
        models.append((model,ckpt['scaler']))
    out.mkdir(parents=True)
    all_y, all_pred, shards = [], [], []
    for name in meta['shards']:
        with np.load(root/name, allow_pickle=False) as z:
            x, xy = z['features'].copy(), z['xy'].copy()
            y = z['target'].copy() if a.evaluate else None
        predictions = np.stack([scale(predict(torch,m,x,a.device,a.batch_size),s,inverse=True)
                                for m,s in models])
        if not np.isfinite(predictions).all():
            raise ValueError('Nonfinite predictions')
        mean = predictions.mean(0)
        arrays = {'xy':xy, 'prediction':mean}
        if len(models)>1:
            arrays['head_disagreement_std'] = predictions.std(0, ddof=1)
        np.savez(out/name, **arrays)
        shards.append(name)
        if a.evaluate:
            all_y.append(y); all_pred.append(mean)
    if a.evaluate:
        y, pred = np.concatenate(all_y), np.concatenate(all_pred)
        report = metrics(y,pred,np.full(len(y),meta['patient_id']),meta['panel'])
        (out/'test_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'manifest.json').write_text(json.dumps({**meta, 'shards':shards,
        'heads':[str(Path(h).resolve()) for h in a.heads],
        'output':'Raw-scale central block marker estimates; no clipping',
        'uncertainty':'Head disagreement only; uncalibrated' if len(models)>1 else 'Not estimated',
        'complete':True},indent=2))

if __name__ == '__main__':
    main()
