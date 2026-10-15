"""Validate and assemble distinct completed cache cohorts without altering inputs."""
import argparse
import hashlib
from pathlib import Path
from data import read_meta
from validate import validate
from raw_safety import guard_output
from audit import snapshot,write_json


def inventory(roots):
    samples={};reference=None
    for root in roots:
        manifests=sorted(Path(root).glob('*/manifest.json'))
        if not manifests:raise ValueError(f'No sample manifests in {root}')
        for manifest in manifests:
            sample=manifest.parent;meta=read_meta(sample)
            if meta.get('encoder')!='UNI2-h' or meta.get('diagnostic_only'):raise ValueError('Need completed non-diagnostic UNI2-h caches')
            report=validate(sample);name=meta['sample_id']
            if name in samples:raise ValueError(f'Duplicate sample: {name}')
            signature=[meta[k] for k in ('panel','encoder','revision','checkpoint_sha256','transform','input_patch','he_mpp','target_block')]+[meta.get('selection_policy')]
            if reference is not None and signature!=reference:raise ValueError('Cache provenance, panel, scale, geometry or selection differ')
            reference=signature
            samples[name]={'sample_id':name,'patient_id':meta['patient_id'],'path':str(sample.resolve()),'rows':report['rows'],
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'he':meta['he']}
    return samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--expected-samples',type=int,required=True);p.add_argument('--require-samples',nargs='+',default=[])
    a=p.parse_args();samples=inventory(a.roots)
    if len(samples)!=a.expected_samples:raise ValueError(f'Expected {a.expected_samples}, found {len(samples)}')
    if set(a.require_samples)-set(samples):raise ValueError('Required new samples absent')
    out=Path(a.out).resolve();guard_output(out,[{'folder':str(Path(r['he']).parent)} for r in samples.values()])
    if out.exists():raise FileExistsError('Use a new merged cache output')
    out.mkdir(parents=True);snapshot(out,vars(a))
    for name,row in samples.items():(out/name).symlink_to(row['path'],target_is_directory=True)
    write_json(out/'inventory.json',[samples[k] for k in sorted(samples)])
    write_json(out/'assembly.json',{'source_roots':[str(Path(r).resolve()) for r in a.roots],'samples':len(samples),
        'patient_ids':sorted(set(r['patient_id'] for r in samples.values())),'required_samples':a.require_samples,
        'note':'New symlink assembly; retain original cache directories. Regenerate CV folds for changed inventory.'})
    print(f'Validated {len(samples)} samples, {len(set(r["patient_id"] for r in samples.values()))} patient IDs: {out}')

if __name__=='__main__':main()
