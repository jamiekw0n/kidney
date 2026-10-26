"""Guard derived output paths and record read-only source metadata."""
from pathlib import Path


def guard_output(out,rows):
    out=Path(out).resolve()
    for row in rows:
        source=Path(row['folder']).resolve()
        if out==source or out in source.parents or source in out.parents:
            raise ValueError(f'Derived output overlaps raw sample directory: {source}')


def source_stats(row):
    result={}
    for key in ('he','mif'):
        if row.get(key):
            path=Path(row[key]).resolve(); stat=path.stat()
            result[key]={'path':str(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    return result
