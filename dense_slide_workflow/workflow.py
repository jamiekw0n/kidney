"""New immutable shorter-stride experiment: per-sample preparation/extraction, parallel folds."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from audit import write_json,snapshot
from data import read_meta
from validate import validate
from normalization import UNITS,target_groups
from source_paths import resolve_sources
from io_utils import independent_output
from raw_safety import source_stats
from spatial import select
from encoder_runtime import sha


def run_config(run):return json.loads((Path(run)/'experiment.json').read_text())

def task_config(run,index):
    c=run_config(run)
    if not 0<=index<len(c['tasks']):raise ValueError('Invalid task index')
    return c,c['tasks'][index]


def plan(a):
    norm=Path(a.normalization_plan).resolve();roots=sorted((norm/'cache').glob('*/manifest.json'))
    if not roots:raise ValueError('No completed normalized caches; finish normalization first')
    from cross_validate import reviewed_folds
    reviewed_folds(norm/'folds.json',[p.parent for p in roots])
    if a.stride<8 or a.stride%8 or a.cap<1:raise ValueError('Stride must be multiple of 8; cap positive')
    if a.he_budget_gib<=0 or not np.isfinite(a.he_budget_gib):raise ValueError('Positive H&E memory budget required')
    tasks=[];raw=[];signature=None
    for p in roots:
        meta=read_meta(p.parent);validate(p.parent)
        if meta.get('target_units')!=UNITS:raise ValueError('Need newly normalized reference targets')
        paths,provenance=resolve_sources(p.parent)
        norm_report=p.parent/'normalization.json'
        report=json.loads(norm_report.read_text())
        if report['recipe']!=meta['normalization_recipe']:raise ValueError('Normalization recipe mismatch')
        if [r['marker'] for r in report['markers']]!=meta['panel']:raise ValueError('Normalization marker order mismatch')
        sig=[meta[k] for k in ('panel','revision','checkpoint_sha256','transform','normalization_recipe','he_mpp')]
        if signature is not None and sig!=signature:raise ValueError('Cohort signatures differ')
        signature=sig
        if Path(meta['sample_id']).name!=meta['sample_id']:raise ValueError('Unsafe sample ID')
        raw.extend(Path(paths[k]).parent for k in ('he','mif'))
        tasks.append({'sample_id':meta['sample_id'],'patient_id':meta['patient_id'],'normalized_cache':str(p.parent.resolve()),
                      'normalization_sha256':sha(norm_report),'raw_paths':paths,'provenance':provenance,
                      'reference_manifest_sha256':sha(p)})
    checkpoint=Path(a.checkpoint).resolve()
    if sha(checkpoint)!=signature[2]:raise ValueError('Checkpoint differs from normalized cache encoder')
    out=independent_output(a.out,[norm,*raw]);out.mkdir(parents=True);snapshot(out,vars(a))
    cfg={'tasks':tasks,'normalization_plan':str(norm),'checkpoint':str(checkpoint),'stride':a.stride,'cap':a.cap,
         'seed':42,'he_budget_gib':a.he_budget_gib,'checkpoint_sha256':signature[2]}
    write_json(out/'experiment.json',cfg);write_json(out/'folds.json',json.loads((norm/'folds.json').read_text()))
    write_json(out/'inventory.json',[{'sample_id':t['sample_id'],'patient_id':t['patient_id']} for t in tasks])
    code=out/'source';code.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md','.sbatch'):(code/p.name).write_bytes(p.read_bytes())
    write_json(out/'source_hashes.json',{p.name:sha(p) for p in code.iterdir() if p.is_file()})
    for name in ('prepared','cache','cv','slides'):(out/name).mkdir()
    print(json.dumps({'plan':str(out),'samples':len(tasks),'stride':a.stride,'cap':a.cap,
        'max_feature_bytes':len(tasks)*a.cap*1536*4,'next':'submit prepare/extract arrays using submit.py'},indent=2))


def prepare_task(run,index):
    from image_io import TiffRegionReader,CachedRegionReader
    from prepare import header,check_scale
    c,t=task_config(run,index);run=Path(run);source=Path(t['normalized_cache']);meta=read_meta(source)
    if sha(source/'manifest.json')!=t['reference_manifest_sha256']:raise ValueError('Reference manifest changed since planning')
    if sha(source/'normalization.json')!=t['normalization_sha256']:raise ValueError('Normalization factors changed')
    paths,_=resolve_sources(source)
    if paths!=t['raw_paths']:raise ValueError('Raw source paths changed')
    row={'folder':str(Path(paths['mif']).parent),**paths};before=source_stats(row)
    oldcheck=json.loads((source/'raw_input_check.json').read_text())
    if before!=oldcheck['after']:raise ValueError('Raw inputs changed since normalization')
    out=independent_output(run/'prepared'/t['sample_id'],[source,*[Path(v).parent for v in paths.values()]])
    norm_info=json.loads((source/'normalization.json').read_text())
    hi,mi=header(paths['he']),header(paths['mif'],norm_info.get('series',0),norm_info.get('level',0));check_scale(hi,meta['he_mpp']);check_scale(mi,meta['he_mpp'])
    labels=mi['markers']
    if len(labels)!=len(set(labels)) or set(labels)!=set(meta['panel']):raise ValueError('mIF panel mismatch')
    order=[labels.index(m) for m in meta['panel']]
    axes=norm_info['source_axes']
    report=json.loads((source/'normalization.json').read_text());divisors=[r['effective_divisor'] for r in report['markers']]
    started=time.monotonic();out.mkdir();snapshot(out,{'run':str(run),'task_id':index})
    with ExitStack() as stack:
        he=TiffRegionReader(paths['he'],3);stack.callback(he.close)
        if he.dtype!=np.uint8 or [he.height,he.width]!=meta['shape']:raise ValueError('H&E geometry/dtype mismatch')
        nbytes=he.height*he.width*3
        if nbytes>c['he_budget_gib']*1024**3:raise MemoryError(f'H&E needs {nbytes} bytes; increase planned he budget and allocation')
        print(json.dumps({'stage':'decode_he_once','bytes':nbytes}),flush=True)
        pixels=he.read(0,he.height,0,he.width)
        np.save(out/'he.npy',pixels)
        with (out/'candidate_decisions.csv').open('w',newline='') as f:
            log=csv.writer(f);log.writerow(['x','y','context_fraction','center_fraction','decision'])
            xy,counts=select(pixels,c['stride'],c['cap'],c['seed']+index,log,lambda counts:print(json.dumps({'stage':'filter_progress','counts':counts}),flush=True))
        del pixels
        print(json.dumps({'stage':'selection_complete','sample':t['sample_id'],'counts':counts}),flush=True)
        if not len(xy):raise ValueError('No retained locations')
        mif=TiffRegionReader(paths['mif'],len(meta['panel']),axes,report.get('series',0),report.get('level',0));stack.callback(mif.close)
        if [mif.height,mif.width]!=meta['shape']:raise ValueError('mIF geometry mismatch')
        raw,target=target_groups(mif,xy,order,divisors,512,
             lambda done,total:print(json.dumps({'stage':'targets','done':done,'total':total}),flush=True))
    shards=[]
    for i,start in enumerate(range(0,len(xy),1000)):
        name=f'targets_{i:05d}.npz';np.savez(out/name,xy=xy[start:start+1000],measured=target[start:start+1000],raw_target=raw[start:start+1000]);shards.append(name)
    after=source_stats(row)
    if before!=after:raise ValueError('Raw input metadata changed')
    write_json(out/'raw_input_check.json',{'before':before,'after':after,'unchanged':True})
    write_json(out/'filtering.json',{'counts':counts,'stride':c['stride'],'cap':c['cap'],'context_min':.5,'center_min':.5,
        'bright':235,'dark':20,'chroma':15,'seed':c['seed']+index,'support':'registered all-valid assumption; no validity mask',
        'selection':'seeded reservoir sampling of eligible positions','coordinates_file':'candidate_decisions.csv'})
    write_json(out/'normalization.json',report)
    saved={k:meta[k] for k in ('sample_id','patient_id','panel','shape','he_mpp','input_patch','target_block','target_units','normalization_recipe')}
    saved['retained']=len(xy)
    saved.update(complete=True,**paths,he_axes=hi['axes'],he_series=0,he_level=0,shards=shards,
        source_normalized_cache=str(source),normalization_report='normalization.json',
        selection_policy={'stride':c['stride'],'max_per_sample':c['cap'],'min_context':.5,'min_center':.5,
                          'bright':235,'dark':20,'chroma':15,'seed':c['seed']},elapsed_seconds=time.monotonic()-started)
    saved['normalization_reference_sha256']=t['normalization_sha256']
    saved['he_array_stat']={'size':(out/'he.npy').stat().st_size,'mtime_ns':(out/'he.npy').stat().st_mtime_ns}
    write_json(out/'manifest.json',saved);print(json.dumps(validate(out)),flush=True)


def extract_task(run,index,batch,device):
    from encoder_runtime import load_encoder,encode
    c,t=task_config(run,index);run=Path(run);source=run/'prepared'/t['sample_id'];validate(source);meta=read_meta(source)
    out=independent_output(run/'cache'/t['sample_id'],[source,*[Path(t['raw_paths'][k]).parent for k in ('he','mif')]])
    ref=read_meta(t['normalized_cache']);model,transform=load_encoder(c['checkpoint'],ref['checkpoint_sha256'],device)
    he=np.load(source/'he.npy',mmap_mode='r',allow_pickle=False)
    out.mkdir();snapshot(out,{'run':str(run),'task_id':index,'batch_size':batch,'device':device})
    for name in meta['shards']:
        with np.load(source/name,allow_pickle=False) as z:xy=z['xy'];target=z['measured']
        features=np.empty((len(xy),1536),np.float32)
        for start in range(0,len(xy),batch):
            features[start:start+batch]=encode(model,transform,he,xy[start:start+batch],device)
        np.savez(out/name,xy=xy,features=features,target=target)
        print(json.dumps({'stage':'features','sample':t['sample_id'],'shard':name,'rows':len(xy)}),flush=True)
    saved={**meta,**{k:ref[k] for k in ('encoder','revision','checkpoint_sha256','transform')},'source':str(source.resolve()),'complete':True}
    write_json(out/'manifest.json',saved);write_json(out/'normalization.json',json.loads((source/'normalization.json').read_text()))
    print(json.dumps(validate(out)),flush=True)


def collect_cache(run,persist=True):
    c=run_config(run);results=[]
    for t in c['tasks']:results.append(validate(Path(run)/'cache'/t['sample_id']))
    from cross_validate import reviewed_folds
    reviewed_folds(Path(run)/'folds.json',[Path(run)/'cache'/t['sample_id'] for t in c['tasks']])
    if persist:write_json(Path(run)/'cache_collection.json',{'complete':True,'samples':results,'rows':sum(r['rows'] for r in results)})
    print(json.dumps({'complete':len(results),'rows':sum(r['rows'] for r in results)},indent=2))


def fold_task(run,index,mode,device):
    run=Path(run);c=run_config(run);folds=json.loads((run/'folds.json').read_text())
    if not 0<=index<len(folds):raise ValueError('Fold outside plan')
    collect_cache(run,persist=False)
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('cross_validate.py')),'--cache-root',str(run/'cache'),
         '--reference-folds',str(run/'folds.json'),'--out',str(run/'cv'/f'part_{index:02d}'),
         '--fold-index',str(index),'--mode',mode,'--device',device,'--max-numeric-gb','24']
    subprocess.run(cmd,check=True)


def collect_cv(run):
    from primary_evaluation import save_primary
    from expression import summarize_subgroups
    run=Path(run);c=run_config(run);folds=json.loads((run/'folds.json').read_text());patients={};expression={};modes=set()
    for i,split in enumerate(folds):
        part=run/'cv'/f'part_{i:02d}';m=read_meta_no_patient(part/'manifest.json')
        if not m.get('complete') or m.get('fold_index')!=i:raise ValueError('Incomplete fold part')
        modes.add(m['mode']);r=json.loads((part/'oof_patient_metrics.json').read_text())
        if set(r)!=set(split['test']) or set(r)&set(patients):raise ValueError('OOF patient mismatch')
        patients.update(r);expression.update(json.loads((part/'oof_expression_patients.json').read_text()))
    if len(modes)!=1:raise ValueError('Mixed experiment modes')
    panel=read_meta(Path(c['tasks'][0]['normalized_cache']))['panel']
    dest=independent_output(run/'cv'/'aggregate',[run/'cache']);dest.mkdir()
    merged={m:{'patients':{pid:r[m]['patients'][pid] for pid,r in patients.items()}} for m in panel}
    save_primary(dest,merged,'out_of_fold',2000,42)
    write_json(dest/'oof_expression_summary.json',summarize_subgroups(expression,panel))
    write_json(dest/'manifest.json',{'complete':True,'patients':len(patients),'folds':len(folds),'mode':next(iter(modes))})
    print(f'OOF report: {dest}/primary_summary.csv')


def read_meta_no_patient(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--normalization-plan',required=True);q.add_argument('--out',required=True);q.add_argument('--checkpoint',required=True)
    q.add_argument('--stride',type=int,default=32);q.add_argument('--cap',type=int,default=20000);q.add_argument('--he-budget-gib',type=float,default=8)
    for cmd in ('prepare','extract','train-fold','collect-cache','collect-cv'):
        q=sub.add_parser(cmd);q.add_argument('--run',required=True)
        if cmd in ('prepare','extract','train-fold'):q.add_argument('--task-id',type=int,required=True)
        if cmd in ('extract','train-fold'):q.add_argument('--device',default='cuda')
        if cmd=='extract':q.add_argument('--batch-size',type=int,default=16)
        if cmd=='train-fold':q.add_argument('--mode',choices=['weighted','weighted_fds'],default='weighted')
    a=p.parse_args()
    if a.command=='plan':plan(a)
    elif a.command=='prepare':prepare_task(a.run,a.task_id)
    elif a.command=='extract':
        if a.batch_size<1:raise ValueError('Positive batch size required')
        extract_task(a.run,a.task_id,a.batch_size,a.device)
    elif a.command=='train-fold':fold_task(a.run,a.task_id,a.mode,a.device)
    elif a.command=='collect-cache':collect_cache(a.run)
    else:collect_cv(a.run)

if __name__=='__main__':main()
