"""Create an audited per-sample filtering table for the exact CV inventory (no image reads)."""
import argparse
import json
from pathlib import Path
from audit import snapshot,write_json

KEYS=('candidates','invalid_support','low_context_tissue','low_center_tissue','eligible','retained','eligible_not_sampled')
LABELS=('Candidate 128×128 contexts','Invalid support','Low context tissue','Low center tissue','Eligible','Retained','Eligible, not sampled')


def checked_counts(report,meta,expected_rows):
    if report.get('diagnostic_only') or meta.get('diagnostic_only'):
        raise ValueError('Diagnostic-only report is not a cohort scan')
    c=report['counts']
    if any(type(c.get(k)) is not int or c[k]<0 for k in KEYS):
        raise ValueError('Missing/noninteger/negative filtering count')
    if c['candidates']!=sum(c[k] for k in ('invalid_support','low_context_tissue','low_center_tissue','eligible')):
        raise ValueError('Candidate counts do not reconcile')
    if c['eligible']!=c['retained']+c['eligible_not_sampled'] or c['retained']!=expected_rows:
        raise ValueError('Retained counts differ from CV inventory or sampling counts')
    args=report['args'];stride=args['stride']
    if meta['input_patch']!=128 or stride!=meta['selection_policy']['stride'] or stride<=0:
        raise ValueError('Geometry mismatch')
    h,w=meta['shape']
    grid=len(range(68,w-64,stride))*len(range(68,h-64,stride))
    if c['candidates']!=grid:
        raise ValueError('Scanned candidates differ from expected full grid')
    return {k:c[k] for k in KEYS}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cv=Path(a.cv_root).resolve()
    inv=json.loads((cv/'inventory.json').read_text())
    if not inv or len({r['sample_id'] for r in inv})!=len(inv):raise ValueError('Empty/duplicate inventory')
    rows=[]
    for item in sorted(inv,key=lambda r:r['sample_id']):
        cache=Path(item['path']);meta=json.loads((cache/'manifest.json').read_text())
        if not meta.get('complete') or any(meta[k]!=item[k] for k in ('sample_id','patient_id')):
            raise ValueError('Incomplete/mismatched cache identity')
        report_path=cache/'filtering.json'
        # Older caches may reference their preparation instead of containing a copy.
        if not report_path.exists():report_path=Path(meta['source'])/'filtering.json'
        report=json.loads(report_path.read_text())
        counts=checked_counts(report,meta,item['rows'])
        rows.append({'sample_id':item['sample_id'],'patient_id':item['patient_id'],**counts,
            'stride':report['args']['stride'],'height':meta['shape'][0],'width':meta['shape'][1],
            'support':report.get('support'),'report':str(report_path.resolve())})
    total={k:sum(r[k] for r in rows) for k in KEYS}
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    write_json(out/'filtering_summary.json',{'cv_root':str(cv),'samples':len(rows),'totals':total,'rows':rows})
    lines=['# Cohort filtering counts','',f'Inventory: `{cv}`; {len(rows)} samples.','',
        'Candidate contexts are the fully in-bounds stride grid actually scanned, not every possible sliding window. Rejection reasons are sequential first failures. Eligible-but-not-sampled locations passed filtering and were excluded by the sampling cap. Border positions outside this grid are not counted as rejections.','',
        '| Sample | '+' | '.join(LABELS)+' |','|---|'+'---:|'*len(KEYS)]
    for row in rows:lines.append('| '+row['sample_id']+' | '+' | '.join(f'{row[k]:,}' for k in KEYS)+' |')
    lines.append('| **TOTAL** | '+' | '.join(f'{total[k]:,}' for k in KEYS)+' |')
    lines+=['','Counts verified: candidates = three rejection categories + eligible; eligible = retained + eligible-not-sampled; retained = CV inventory rows. Full scan size checked against dimensions and stride.','',
        'See filtering_summary.json for patient IDs, slide dimensions, stride, support assumptions and exact source report paths. No raw images were read or changed.']
    (out/'filtering_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f'\nSaved {out}/filtering_summary.md and filtering_summary.json')

if __name__=='__main__':main()
