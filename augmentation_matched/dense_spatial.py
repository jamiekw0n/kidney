"""Native-pixel grid and exact tissue-fraction operations shared by train/inference."""
import numpy as np

def integral(mask):
    return np.pad(mask.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))

def tissue_mask(rgb):
    v=rgb.astype(np.float32)
    return (v.mean(-1)>20)&(v.mean(-1)<235)&((v.max(-1)-v.min(-1))>15)

def fractions(ii,xy,half):
    x=xy[:,0];y=xy[:,1]
    return (ii[y+half,x+half]-ii[y-half,x+half]-ii[y+half,x-half]+ii[y-half,x-half])/(2*half)**2

def select(he,stride,cap,seed,log,progress=None):
    h,w=he.shape[:2];rng=np.random.default_rng(seed);items=[];counts={'candidates':0,'low_context_tissue':0,'low_center_tissue':0,'eligible':0}
    # Compute bounded strips rather than a full-slide int64 integral image.
    for y in range(68,h-64,stride):
        xs=np.arange(68,w-64,stride,dtype=np.int64)
        if not len(xs):continue
        ii=integral(tissue_mask(he[y-64:y+64]))
        coords=np.column_stack((xs,np.full(len(xs),64,dtype=np.int64)))
        c=fractions(ii,coords,64);t=fractions(ii,coords,4)
        for x,context,center in zip(xs,c,t):
            reason='low_context_tissue' if context<.5 else 'low_center_tissue' if center<.5 else 'eligible'
            counts['candidates']+=1;counts[reason]+=1
            log.writerow([int(x),y,float(context),float(center),reason])
            if reason=='eligible':
                item=(int(x),y)
                if len(items)<cap:items.append(item)
                else:
                    j=int(rng.integers(counts['eligible']))
                    if j<cap:items[j]=item
        if progress and (y==68 or (y-68)//stride%20==0):progress(dict(counts))
    counts['retained']=len(items);counts['eligible_not_sampled']=counts['eligible']-len(items)
    return np.asarray(sorted(items,key=lambda p:(p[1],p[0])),dtype=np.int64).reshape(-1,2),counts

def band_geometry(h,w,band_rows,index):
    ny,nx=h//8,w//8;n=(ny+band_rows-1)//band_rows
    if not 0<=index<n:raise ValueError('Invalid band index')
    start=index*band_rows;stop=min(ny,start+band_rows)
    return ny,nx,start,stop,n

def inference_mask(he,start,stop):
    h,w=he.shape[:2];nx=w//8
    xs=np.arange(nx)*8+4;ys=np.arange(start,stop)*8+4
    xx,yy=np.meshgrid(xs,ys);xy=np.column_stack((xx.ravel(),yy.ravel()))
    context=(xy[:,0]>=64)&(xy[:,0]+64<=w)&(xy[:,1]>=64)&(xy[:,1]+64<=h)
    supported=xy[context]
    valid=np.zeros(len(xy),bool)
    if len(supported):
        y0=max(0,int(ys[0])-64);y1=min(h,int(ys[-1])+64)
        ii=integral(tissue_mask(he[y0:y1]));local=supported-[0,y0]
        keep=(fractions(ii,local,64)>=.5)&(fractions(ii,local,4)>=.5)
        valid[np.flatnonzero(context)[keep]]=True
    return xy,valid
