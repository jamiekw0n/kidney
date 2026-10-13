"""Preflight paired targets or feature caches without PyTorch or image decoding."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import read_meta, partitions


def validate(root, require_targets=True):
    root = Path(root)
    meta = read_meta(root)
    if not meta.get('panel') or len(set(meta['panel'])) != len(meta['panel']):
        raise ValueError('Panel must contain unique marker names')
    if not meta.get('shards') or len(set(meta['shards'])) != len(meta['shards']):
        raise ValueError('Nonempty unique shards required')
    n = 0
    feature_bytes, target_bytes = 0, 0
    for name in meta['shards']:
        path = root/name
        if path.resolve().parent != root.resolve():
            raise ValueError('Shards must be direct children of dataset directory')
        with np.load(path,allow_pickle=False) as z:
            xy = z['xy']
            key = 'target' if 'encoder' in meta else 'measured'
            if xy.ndim != 2 or xy.shape[1] != 2 or not np.issubdtype(xy.dtype,np.integer):
                raise ValueError('Coordinates must be integer [N,2]')
            if len(xy) == 0:
                raise ValueError('Empty shard')
            if require_targets:
                y = z[key]
                if (not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y)
                        or y.shape != (len(xy),len(meta['panel'])) or not np.isfinite(y).all()):
                    raise ValueError('Invalid marker targets')
                target_bytes += y.nbytes
            h,w = meta['shape']
            if (xy < 0).any() or (xy[:,0] >= w).any() or (xy[:,1] >= h).any():
                raise ValueError('Coordinate outside slide')
            block = meta.get('target_block')
            if not isinstance(block, int) or isinstance(block, bool) or block <= 0 or block % 2:
                raise ValueError('target_block must be a positive even integer')
            half = block // 2
            if (xy < half).any() or (xy[:,0] > w-half).any() or (xy[:,1] > h-half).any():
                raise ValueError('Target block extends outside slide')
            if 'encoder' in meta:
                x = z['features']
                if x.dtype != np.float32 or x.shape != (len(xy),1536) or not np.isfinite(x).all():
                    raise ValueError('Invalid UNI2-h features')
                feature_bytes += x.nbytes
            n += len(xy)
    if n == 0:
        raise ValueError('Empty dataset')
    return {'sample_id':meta['sample_id'],'patient_id':meta['patient_id'],
            'rows':n,'markers':len(meta['panel']),'kind':'cache' if 'encoder' in meta else 'prepared',
            'feature_bytes':feature_bytes, 'target_bytes':target_bytes}


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
