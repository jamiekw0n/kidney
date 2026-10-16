"""Plot recorded training/inner-validation Huber histories without retraining."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from audit import snapshot,write_json


def read_history(training):
    history=json.loads((training/'history.json').read_text())
    run=json.loads((training/'run.json').read_text())
    if not history or [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError(f'Empty or nonsequential history: {training}')
    values=np.array([[r['train_huber'],r['val_huber']] for r in history],float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid Huber values')
    best=int(run['best_epoch'])
    if best!=int(np.argmin(values[:,1]))+1:raise ValueError('Saved selected epoch differs from lowest validation loss')
    return history,run


def draw(ax,history,best,title):
    epochs=[r['epoch'] for r in history]
    ax.plot(epochs,[r['train_huber'] for r in history],label='Training',color='tab:blue')
    ax.plot(epochs,[r['val_huber'] for r in history],label='Inner validation',color='tab:orange')
    ax.axvline(best,color='gray',linestyle='--',label=f'Selected epoch {best}')
    ax.scatter([best],[history[best-1]['val_huber']],color='tab:orange',zorder=3)
    ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Huber loss (training-standardized targets)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-roots',nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    if len(a.cv_roots)!=len(a.labels) or len(set(a.labels))!=len(a.labels):p.error('One unique label per root required')
    roots=list(map(Path,a.cv_roots));folds=[json.loads((r/'folds.json').read_text()) for r in roots]
    if any(f!=folds[0] for f in folds[1:]):raise ValueError('Fold assignments differ')
    records=[]
    for root,label in zip(roots,a.labels):
        for i in range(len(folds[0])):
            training=root/f'fold_{i:02d}'/'training';history,run=read_history(training)
            # Axis label is specific to the completed z-score experiments.
            if run['args'].get('intensity_max') is not None:raise ValueError('This plot label assumes z-scored targets, not fixed range training')
            records.append((label,i,history,run,training))
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);snapshot(out,vars(a))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    overview,axes=plt.subplots(len(roots),len(folds[0]),figsize=(5*len(folds[0]),4*len(roots)),squeeze=False,constrained_layout=True)
    summary=[];links=[]
    for n,(label,i,history,run,training) in enumerate(records):
        best=run['best_epoch'];title=f'{label} — fold {i}'
        draw(axes[n//len(folds[0]),i],history,best,title)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True);draw(ax,history,best,title)
        name=f'run_{n//len(folds[0]):02d}_fold_{i:02d}.png';fig.savefig(out/name,dpi=160);plt.close(fig)
        links.append(f'<h2>{html.escape(title)}</h2><img style="max-width:100%" src="{name}">')
        args=run['args']
        summary.append({'model':label,'fold':i,'completed_epochs':len(history),'selected_epoch':best,
            'best_validation_huber':history[best-1]['val_huber'],'training_huber_at_selected_epoch':history[best-1]['train_huber'],
            'last_training_huber':history[-1]['train_huber'],'last_validation_huber':history[-1]['val_huber'],
            'max_epochs':args.get('epochs'),'patience':args.get('patience'),'history_path':str((training/'history.json').resolve())})
    overview.savefig(out/'overview.png',dpi=140);plt.close(overview)
    write_json(out/'loss_summary.json',summary)
    notes='Training loss is measured during weight updates using patient-balanced sampling with replacement; MLP dropout is active. Validation is computed after the epoch with dropout disabled, averaged equally over validation patients. These losses are not normalized reporting MAE and not outer-test scores. Each fold has its own training-only target scaler; avoid interpreting differences across folds as a common raw-intensity scale. No fold averaging, smoothing or rerunning inference is performed.'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Training versus inner-validation loss</h1><p>'+notes+'</p><img style="max-width:100%" src="overview.png">'+''.join(links))
    for row in summary:print(json.dumps(row),flush=True)
    print(f'Open {out}/index.html after copying the whole directory locally')

if __name__=='__main__':main()
