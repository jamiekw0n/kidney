"""Preflight paired targets or feature caches without PyTorch or image decoding."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import read_meta, partitions


def validate(root):
    root = Path(root)
    meta = read_meta(root)
    if not meta.get('panel') or len(set(meta['panel'])) != len(meta['panel']):
        raise ValueError('Panel must contain unique marker names')
    if not meta.get('shards') or len(set(meta['shards'])) != len(meta['shards']):
        raise ValueError('Nonempty unique shards required')
    n = 0
    for name in meta['shards']:
        path = root/name
        if path.resolve().parent != root.resolve():
            raise ValueError('Shards must be direct children of dataset directory')
        with np.load(path,allow_pickle=False) as z:
            xy = z['xy']
            key = 'target' if 'encoder' in meta else 'measured'
            y = z[key]
            if xy.ndim != 2 or xy.shape[1] != 2 or not np.issubdtype(xy.dtype,np.integer):
                raise ValueError('Coordinates must be integer [N,2]')
            if y.shape != (len(xy),len(meta['panel'])) or not np.isfinite(y).all():
                raise ValueError('Invalid marker targets')
            h,w = meta['shape']
            if (xy < 0).any() or (xy[:,0] >= w).any() or (xy[:,1] >= h).any():
                raise ValueError('Coordinate outside slide')
            if 'encoder' in meta:
                x = z['features']
                if x.shape != (len(xy),1536) or not np.isfinite(x).all():
                    raise ValueError('Invalid UNI2-h features')
            n += len(xy)
    if n == 0:
        raise ValueError('Empty dataset')
    return {'sample_id':meta['sample_id'],'patient_id':meta['patient_id'],
            'rows':n,'markers':len(meta['panel']),'kind':'cache' if 'encoder' in meta else 'prepared'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--split',help='Also validate cache patient partitions')
    a=p.parse_args()
    for root in a.datasets:
        print(json.dumps(validate(root)))
    if a.split:
        groups=partitions(a.datasets,json.loads(Path(a.split).read_text()))
        print(json.dumps({k:len(v) for k,v in groups.items()}))

if __name__=='__main__':
    main()
