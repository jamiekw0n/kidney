"""Read-only memory-mapped augmented views indexed in ORIGINAL training-row order."""
import json
from pathlib import Path
import numpy as np
from data import read_meta

class FeatureViews:
    def __init__(self,plan,roots):
        from pipeline import config
        self.plan=Path(plan);cfg=config(self.plan)
        self.fingerprint=cfg['fingerprint']
        tasks={t['sample_id']:t for t in cfg['tasks']};self.entries=[];self.ends=[];self.maps={};offset=0
        for root in roots:
            meta=read_meta(root);sample=meta['sample_id'];task=tasks[sample]
            if str(Path(root).resolve())!=task['cache']:raise ValueError('Training root differs from recorded identity cache')
            base=self.plan/'views'/sample;m=json.loads((base/'manifest.json').read_text())
            if not m.get('complete') or m['plan_fingerprint']!=cfg['fingerprint']:raise ValueError('Incomplete/incompatible augmentation cache')
            if [r['name'] for r in m['shards']]!=meta['shards']:raise ValueError('View shard order differs')
            for r in m['shards']:
                n=r['rows'];self.entries.append((offset,base,r));offset+=n;self.ends.append(offset)
        self.ends=np.array(self.ends);self.rows=offset

    def replace(self,batch,indices,chosen,variant):
        if variant=='control':return batch
        key='geometry' if variant=='geometry' else 'geometry_color' if variant=='geometry_color' else None
        if key is None:raise ValueError('Unknown augmentation variant')
        positions=np.flatnonzero(chosen)
        groups=np.searchsorted(self.ends,np.asarray(indices)[positions],side='right')
        for g in np.unique(groups):
            pos=positions[groups==g];offset,base,record=self.entries[int(g)]
            cache_key=(int(g),key)
            if cache_key not in self.maps:
                a=np.load(base/record[key],mmap_mode='r',allow_pickle=False)
                if a.shape!=(record['rows'],1536) or a.dtype!=np.float32:raise ValueError('Invalid augmented feature array')
                self.maps[cache_key]=a
            batch[pos]=self.maps[cache_key][np.asarray(indices)[pos]-offset]
        if not np.isfinite(batch).all():raise ValueError('Nonfinite training features')
        return batch
