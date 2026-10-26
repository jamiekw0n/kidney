"""Plot the three matched cohort summaries; no raw images or model inference."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);a=p.parse_args();root=Path(a.plan)
    with (root/'comparison/all_marker_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    variants=['control','geometry','geometry_color'];data={(r['marker'],r['variant']):r for r in rows}
    def number(value):return float(value) if value else np.nan
    markers=sorted({r['marker'] for r in rows},key=lambda m:number(data[m,'control']['pearson']) if data[m,'control']['pearson'] else -np.inf,reverse=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'comparison';dest=out/'matched_metrics.png'
    if dest.exists():raise FileExistsError(dest)
    fig,axes=plt.subplots(1,3,figsize=(19,16),sharey=True);colors=['#555555','#14739c','#ba531f']
    for ax,metric,title in zip(axes,['pearson','spearman','mae'],['Pearson (higher better)','Spearman (higher better)','Normalized MAE (lower better)']):
        for j,(v,color) in enumerate(zip(variants,colors)):
            values=[number(data[m,v][metric]) for m in markers]
            ax.scatter(values,np.arange(len(markers))+(j-1)*.19,color=color,label=v.replace('_',' + '),s=18)
        ax.set_yticks(np.arange(len(markers)));ax.set_yticklabels(markers);ax.grid(axis='x',alpha=.2);ax.set_title(title)
    axes[0].invert_yaxis();axes[2].legend(loc='lower right');fig.suptitle('Matched augmentation experiments: original-view out-of-fold evaluation')
    fig.text(.08,.015,'Each point is the equal-patient mean. Original targets/locations/folds and loss fixed. See paired_differences.csv for paired conditional bootstrap intervals.',fontsize=9)
    fig.subplots_adjust(top=.95,bottom=.06,left=.07,right=.98,wspace=.18)
    fig.savefig(dest,dpi=180);fig.savefig(out/'matched_metrics.pdf');plt.close(fig)
    print(dest,flush=True)

if __name__=='__main__':main()
