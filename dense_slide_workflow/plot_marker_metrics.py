"""Presentation plots from saved OOF metrics. Optional real marker-by-patient heatmaps."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def read_summary(path):
    data={}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            key=(row['marker'],row['metric'])
            if key in data:raise ValueError('Duplicate marker/metric')
            data[key]={k:(float(row[k]) if row[k] else np.nan) for k in ('patient_mean','ci95_low','ci95_high','defined_patients','total_patients')}
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--out',required=True)
    p.add_argument('--heatmap-only',action='store_true',help='Render only heatmap PNG/PDF; preserve other figures')
    p.add_argument('--overwrite-heatmap',action='store_true',help='With --heatmap-only, replace heatmap PNG/PDF in the same experiment output')
    p.add_argument('--patient-csv');p.add_argument('--label',default='Per-slide P99 normalization; stride-32 training; expression-weighted MLP')
    a=p.parse_args();out=Path(a.out).resolve()
    if a.heatmap_only and not a.patient_csv:p.error('--heatmap-only requires --patient-csv')
    if a.overwrite_heatmap and not a.heatmap_only:p.error('--overwrite-heatmap requires --heatmap-only')
    if out.exists() and not a.overwrite_heatmap:raise FileExistsError('Use a NEW output directory or --heatmap-only --overwrite-heatmap')
    if out.exists() and a.overwrite_heatmap:
        previous=json.loads((out/'manifest.json').read_text())
        if previous.get('source_summary')!=str(Path(a.summary).resolve()) or previous.get('source_patient_csv')!=str(Path(a.patient_csv).resolve()):
            raise ValueError('Existing figure inputs differ; refusing overwrite of another experiment')
    data=read_summary(a.summary);markers=list(dict.fromkeys(m for m,k in data))
    required=['pearson','spearman','mae','mean_bias']
    if any((m,k) not in data for m in markers for k in required):raise ValueError('Missing required metrics')
    markers.sort(key=lambda m:data[m,'pearson']['patient_mean'],reverse=True)
    patients={int(data[m,'pearson']['total_patients']) for m in markers}
    if len(patients)!=1:raise ValueError('Inconsistent total patient count')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out.mkdir(parents=True,exist_ok=a.overwrite_heatmap)
    caption=(f'{len(markers)} markers; {next(iter(patients))} patients. Equal-patient means of within-patient OOF metrics.\n'
             'Bars: reported 95% patient-bootstrap CIs, conditional on trained predictions; shared-fold training uncertainty is not fully captured.')
    y=np.arange(len(markers))
    if not a.heatmap_only:
        fig,axes=plt.subplots(1,2,figsize=(14,16),sharey=True)
        low=min(data[m,k]['ci95_low'] for m in markers for k in ('pearson','spearman'))
        for ax,metric,title,color in zip(axes,['pearson','spearman'],['Pearson correlation','Spearman rank correlation'],['#14648a','#a34c18']):
            for i,m in enumerate(markers):
                r=data[m,metric];v=r['patient_mean'];lo=r['ci95_low'];hi=r['ci95_high']
                if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[i,i],color=color,lw=2)
                ax.plot(v,i,'o',color=color,ms=5)
                ax.text(.985,i,f'{v:.3f}',ha='right',va='center',fontsize=9,transform=ax.get_yaxis_transform())
            ax.set_xlim(-1 if low<0 else 0,1);ax.set_xticks(np.arange(0,1.01,.2) if low>=0 else np.arange(-1,1.01,.4))
            ax.set_title(title+' (higher is better)',fontsize=13,pad=12);ax.set_xlabel('Patient-mean correlation')
            ax.grid(axis='x',alpha=.2);ax.axvline(0,color='#777777',lw=.7)
            ax.set_yticks(y);ax.set_yticklabels(markers);ax.tick_params(axis='y',length=0)
        axes[0].invert_yaxis();axes[1].tick_params(labelleft=True)
        fig.suptitle('Held-out prediction agreement by marker',fontsize=18,y=.986)
        fig.text(.5,.957,a.label,ha='center',fontsize=11)
        fig.text(.09,.018,caption,fontsize=9,va='bottom')
        fig.subplots_adjust(top=.93,bottom=.08,left=.09,right=.98,wspace=.3)
        for ext in ('png','pdf'):fig.savefig(out/f'marker_correlations.{ext}',dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,16),sharey=True)
        for ax,metric,title,color in zip(axes,['mae','mean_bias'],['Normalized MAE (lower is better)','Bias: predicted − measured (ideal = 0)'],['#14648a','#a34c18']):
            for i,m in enumerate(markers):
                r=data[m,metric];v=r['patient_mean'];lo=r['ci95_low'];hi=r['ci95_high']
                if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[i,i],color=color,lw=2)
                ax.plot(v,i,'o',color=color,ms=5)
            ax.set_title(title,fontsize=13,pad=12);ax.set_xlabel('Per-slide P99 normalized intensity units')
            ax.set_yticks(y);ax.set_yticklabels(markers);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.2)
            ax.axvline(0,color='#777777',lw=1,ls='--')
        axes[0].invert_yaxis();axes[1].tick_params(labelleft=True)
        fig.suptitle('Held-out prediction error and calibration by marker',fontsize=18,y=.986)
        fig.text(.5,.957,a.label,ha='center',fontsize=11)
        fig.text(.09,.018,caption+'\nMarker order matches the correlation figure. MAE units differ from historical raw-intensity experiments.',fontsize=9,va='bottom')
        fig.subplots_adjust(top=.93,bottom=.09,left=.09,right=.98,wspace=.3)
        for ext in ('png','pdf'):fig.savefig(out/f'marker_error_bias.{ext}',dpi=180)
        plt.close(fig)
    heatmap=False
    if a.patient_csv:
        with Path(a.patient_csv).open(newline='') as f:rows=list(csv.DictReader(f))
        ids=sorted({r['patient_id'] for r in rows});lookup={}
        if set(r['marker'] for r in rows)!=set(markers):raise ValueError('Patient and summary marker inventories differ')
        if len(ids)!=next(iter(patients)):raise ValueError('Patient inventory differs from summary')
        for r in rows:
            key=(r['marker'],r['patient_id'])
            if key in lookup:raise ValueError('Duplicate marker/patient')
            lookup[key]=r
        fig,axes=plt.subplots(1,2,figsize=(24,17))
        for ax,metric in zip(axes,['pearson','spearman']):
            matrix=np.array([[float(lookup[m,p][metric]) if lookup[m,p][metric] else np.nan for p in ids] for m in markers])
            for m,v in zip(markers,matrix):
                if np.isfinite(v).sum()!=int(data[m,metric]['defined_patients']):raise ValueError('Defined patient count mismatch')
                if np.isfinite(v).any() and not np.isclose(np.nanmean(v),data[m,metric]['patient_mean'],rtol=1e-6,atol=1e-8):raise ValueError('Patient metrics do not reproduce summary means')
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
            im=ax.imshow(matrix,cmap=cmap,vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
            ax.set_yticks(y);ax.set_yticklabels(markers);ax.set_xticks(np.arange(len(ids)));ax.set_xticklabels(ids,rotation=90,fontsize=8)
            ax.set_title(metric.title()+' within each patient');ax.set_xlabel('Patient / slide');ax.set_ylabel('Marker')
        fig.suptitle('Marker × patient: held-out prediction agreement',fontsize=18)
        fig.subplots_adjust(top=.95,bottom=.17,left=.06,right=.88,wspace=.3)
        # Fixed dedicated axis outside BOTH panels; later subplot movement cannot overlap it.
        color_axis=fig.add_axes([.92,.28,.014,.48])
        fig.colorbar(im,cax=color_axis,label='Correlation; gray = undefined')
        for ext in ('png','pdf'):fig.savefig(out/f'marker_patient_heatmap.{ext}',dpi=180)
        plt.close(fig);heatmap=True
    if a.heatmap_only:
        print(f'Heatmap PNG/PDF updated: {out}; other files unchanged',flush=True)
        return
    cards=['<h2>Per-marker correlation</h2><img style="max-width:100%" src="marker_correlations.png">',
           '<h2>Normalized MAE and bias</h2><img style="max-width:100%" src="marker_error_bias.png">']
    if heatmap:cards.append('<h2>Marker × patient heatmap</h2><img style="max-width:100%" src="marker_patient_heatmap.png">')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>OOF marker metrics</h1>'+''.join(cards))
    (out/'manifest.json').write_text(json.dumps({'complete':True,'source_summary':str(Path(a.summary).resolve()),'source_patient_csv':str(Path(a.patient_csv).resolve()) if a.patient_csv else None,
        'markers':len(markers),'patients':next(iter(patients)),'heatmap_generated':heatmap,'marker_order':markers},indent=2))
    print(f'Figures: {out}',flush=True)

if __name__=='__main__':main()
