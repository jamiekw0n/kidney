"""Recover raw paths through recorded prepared-data provenance, without discovery."""
import hashlib
import json
from pathlib import Path


def resolve_sources(source):
    current=Path(source).resolve();seen=set();records=[];first=None;paths={}
    while True:
        manifest=current/'manifest.json'
        if manifest in seen:raise ValueError(f'Cycle in source provenance: {manifest}')
        seen.add(manifest)
        if not manifest.is_file():raise ValueError(f'Missing source provenance manifest: {manifest}')
        content=manifest.read_bytes();meta=json.loads(content)
        if first is None:first=meta
        for key in ('sample_id','patient_id','panel','shape','he_mpp','input_patch','target_block'):
            if key in first and meta.get(key)!=first[key]:raise ValueError(f'Source provenance {key} mismatch: {manifest}')
        records.append({'manifest':str(manifest),'sha256':hashlib.sha256(content).hexdigest()})
        for key in ('he','mif'):
            if meta.get(key):
                path=Path(meta[key])
                path=(path if path.is_absolute() else current/path).resolve()
                if key in paths and paths[key]!=str(path):raise ValueError(f'Conflicting {key} paths in provenance')
                paths[key]=str(path)
        if all(key in paths for key in ('he','mif')):break
        ref=meta.get('source')
        if not isinstance(ref,str) or not ref:raise ValueError(f'Cannot resolve raw H&E/mIF paths from {manifest}; missing recorded source directory')
        parent=Path(ref);current=(parent if parent.is_absolute() else current/parent).resolve()
    for key,path in paths.items():
        if not Path(path).is_file():raise ValueError(f'Recorded raw {key} file does not exist: {path}')
    return paths,records
