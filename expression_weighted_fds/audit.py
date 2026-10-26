"""Portable experiment records and immutable source snapshots."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def event(root,step,**values):
    with (Path(root)/'events.jsonl').open('a') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'step':step,**values},allow_nan=False)+'\n')


def snapshot(root,args):
    root=Path(root); destination=root/'source_snapshot'; destination.mkdir()
    hashes={}
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.md','.txt'):
            shutil.copy2(path,destination/path.name)
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root/'invocation.json',{'args':args,'argv':sys.argv,'python':sys.version,'source_sha256':hashes})
