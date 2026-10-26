"""Recolor saved UMAP points by annotation mixtures without refitting UMAP or the encoder."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from audit import snapshot,write_json


def composition_matrix(tiles,column):
    records=[t['annotations'][column] for t in tiles]
    categories=sorted({label for record in records for label in record['composition']})
    counts=np.zeros((len(tiles),len(categories)),dtype=np.int64);index={k:i for i,k in enumerate(categories)}
    for i,record in enumerate(records):
        for label,n in record['composition'].items():
            if type(n) is not int or n<0:raise ValueError('Invalid annotation count')
            counts[i,index[label]]=n
        if counts[i].sum()!=record['labeled_cells']:raise ValueError('Composition/support mismatch')
    support=counts.sum(1)
    fractions=np.divide(counts,support[:,None],out=np.full(counts.shape,np.nan,dtype=float),where=support[:,None]>0)
    entropy=np.full(len(tiles),np.nan)
    if categories:
        safe=np.where(np.isfinite(fractions)&(fractions>0),fractions,1)
        h=-np.where(np.isfinite(fractions),fractions*np.log2(safe),0).sum(1)
        entropy[support>0]=h[support>0]
    return categories,counts,fractions,support,entropy


def continuous_plot(plt,xy,values,title,path,label,vmin=None,vmax=None):
    values=np.asarray(values);valid=np.isfinite(values)
    fig,ax=plt.subplots(figsize=(10,8),constrained_layout=True)
    ax.scatter(xy[~valid,0],xy[~valid,1],s=7,color='#bdbdbd',label='No labeled centroids',rasterized=True)
    if valid.any():
        ids=np.flatnonzero(valid);ids=ids[np.argsort(values[ids],kind='stable')]
        im=ax.scatter(xy[ids,0],xy[ids,1],c=values[ids],s=7,cmap='viridis',vmin=vmin,vmax=vmax,rasterized=True)
        fig.colorbar(im,ax=ax,label=label)
    else:ax.text(.5,.95,'No defined values',ha='center',transform=ax.transAxes)
    if (~valid).any():ax.legend(loc='best',fontsize=8)
    ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--umap-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--label-columns',nargs='+',default=['cell_type_merged','cell_type_new','compartment_refined','cell_state_primary'])
    a=p.parse_args();root=Path(a.umap_root).resolve();out=Path(a.out).resolve()
    if out==root or root in out.parents or out in root.parents:raise ValueError('Use an independent new output directory')
    if out.exists():raise FileExistsError(out)
    if len(set(a.label_columns))!=len(a.label_columns):raise ValueError('Duplicate annotation fields')
    meta=json.loads((root/'manifest.json').read_text())
    if not meta.get('complete') or meta.get('labels_used_to_fit') is not False:raise ValueError('Require completed unsupervised embedding run')
    tiles=json.loads((root/'joined_tiles.json').read_text())
    xy=np.array([t['umap'] for t in tiles],dtype=float)
    if xy.shape!=(len(tiles),2) or not len(tiles) or not np.isfinite(xy).all():raise ValueError('Invalid/missing saved UMAP coordinates')
    with np.load(root/'umap_points.npz',allow_pickle=False) as z:
        if not np.allclose(z['umap'],xy) or not np.array_equal(z['sample_id'],[t['sample_id'] for t in tiles]) or not np.array_equal(z['cache_row'],[t['cache_row'] for t in tiles]):raise ValueError('UMAP point order/provenance mismatch')
    if any(set(a.label_columns)-set(t['annotations']) for t in tiles):raise ValueError('Requested field missing from joined tiles')
    out.mkdir(parents=True);snapshot(out,vars(a))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[];links=[]
    for k,column in enumerate(a.label_columns):
        categories,counts,fractions,support,entropy=composition_matrix(tiles,column)
        directory=out/f'field_{k:02d}';directory.mkdir();cards=[]
        np.savez(directory/'composition.npz',umap=xy,categories=np.array(categories,dtype=str),counts=counts,fractions=fractions,labeled_cells=support,entropy_bits=entropy)
        table=[]
        for j,category in enumerate(categories):
            name=f'fraction_{j:03d}.png'
            continuous_plot(plt,xy,fractions[:,j]*100,f'{column}: {category} fraction',directory/name,'Percent of labeled cell centroids in context',0,100)
            n=int((counts[:,j]>0).sum());mean=float(np.nanmean(fractions[:,j])) if (support>0).any() else None
            cards.append(f'<h2>{html.escape(category)}</h2><p>Present in {n}/{len(tiles)} contexts. No dominant-label cutoff applied.</p><img style="max-width:100%" src="{name}">')
            table.append({'category':category,'contexts_with_category':n,'cell_centroid_assignments':int(counts[:,j].sum()),'mean_fraction_across_contexts_with_any_labels':mean})
        continuous_plot(plt,xy,support,f'{column}: labeled cell count',directory/'support.png','Number of labeled centroids')
        continuous_plot(plt,xy,entropy,f'{column}: mixture diversity',directory/'entropy.png','Shannon entropy (bits); 0 = one category',0)
        write_json(directory/'category_summary.json',table)
        notes='Fractions use all labeled centroids within each context, including mixed/tied contexts. Gray means no labels. Low-count contexts remain plotted, but their fractions are uncertain. Values describe centroid counts, not tissue-area coverage or predicted probabilities. Category plots share a 0–100% scale. Higher values draw last to remain visible; overlapping points can hide lower values.'
        (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>'+html.escape(column)+'</h1><p>'+notes+'</p><h2>Annotation support</h2><img style="max-width:100%" src="support.png"><h2>Mixture diversity</h2><img style="max-width:100%" src="entropy.png">'+''.join(cards))
        item={'column':column,'contexts':len(tiles),'with_labels':int((support>0).sum()),'without_labels':int((support==0).sum()),'categories':categories}
        summaries.append(item);links.append(f'<li><a href="field_{k:02d}/index.html">{html.escape(column)}</a>: {len(categories)} categories</li>')
        print(json.dumps(item),flush=True)
    write_json(out/'manifest.json',{'complete':True,'source_umap_root':str(root),'encoder_rerun':False,'umap_refit':False,'fields':summaries})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Annotation mixture views on existing UMAP</h1><p>Every original UMAP point is preserved. Annotation colors change; feature extraction and UMAP are not rerun. Inspect category fractions together with cell-count support. This is a descriptive projection, not a biological validation score.</p><ul>'+''.join(links)+'</ul>')
    print(f'Open {out}/index.html',flush=True)

if __name__=='__main__':main()
