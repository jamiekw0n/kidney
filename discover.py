"""Inventory registered image pairs without inferring patients or decoding images."""
import argparse
import json
from pathlib import Path


def discover(base):
    base = Path(base).resolve(strict=True)
    if not base.is_dir():
        raise ValueError('Base must be a directory')
    samples = []
    for root in sorted(base.iterdir()):
        if not root.is_dir() or root.name[:1] not in 'BFKMR':
            continue
        files = sorted(p for p in root.iterdir() if p.is_file())
        he = [p for p in files if p.suffix.lower() in ('.tif', '.tiff') and 'HandE' in p.name]
        codex = [p for p in files if p.suffix.lower() == '.qptiff']
        errors = []
        if len(he) != 1:
            errors.append(f'Expected one HandE TIFF; found {len(he)}')
        if len(codex) != 1:
            errors.append(f'Expected one QPTIFF; found {len(codex)}')
        samples.append({'sample_id': root.name, 'directory': str(root),
                        'he_candidates': [str(p) for p in he],
                        'codex_candidates': [str(p) for p in codex],
                        'errors': errors})
    return {'base': str(base), 'samples': samples,
            'note': 'File pairing only; patient identity, channels, scale, and support require verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', help='New inventory JSON path; existing files are refused')
    args = parser.parse_args()
    report = discover(args.base)
    payload = json.dumps(report, indent=2)
    if args.out:
        with Path(args.out).open('x') as stream:
            stream.write(payload + '\n')
    else:
        print(payload)
    return int(not report['samples'] or any(s['errors'] for s in report['samples']))


if __name__ == '__main__':
    raise SystemExit(main())
