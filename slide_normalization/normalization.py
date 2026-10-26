"""Exact unsigned-integer whole-slide percentiles and pixel-before-block scaling."""
import numpy as np


UNITS='slide_channel_percentile_pixel_clipped_0_1'


def recipe(percentile=99.):
    if not np.isfinite(percentile) or not 0<percentile<100:raise ValueError('Percentile must be in (0,100)')
    return {'version':1,'method':'per_slide_per_marker_exact_histogram_percentile',
            'percentile':float(percentile),'percentile_estimator':'linear (NumPy default)',
            'support':'ALL native-resolution image pixels, including zero/background; no tissue mask',
            'operation':'mean_8x8(clip(pixel / effective_divisor, 0, 1))',
            'zero_percentile_policy':'use unsigned image dtype maximum; retain and flag channel',
            'target_units':UNITS}


def histogram_percentile(hist,percentile=99.):
    hist=np.asarray(hist)
    if hist.ndim!=1 or not np.issubdtype(hist.dtype,np.integer) or (hist<0).any() or hist.sum()==0:
        raise ValueError('Need nonempty nonnegative integer histogram')
    if not 0<=percentile<=100:raise ValueError('Invalid percentile')
    cumulative=np.cumsum(hist,dtype=np.int64);n=int(cumulative[-1]);rank=(n-1)*percentile/100
    lo=int(np.floor(rank));hi=int(np.ceil(rank))
    a=int(np.searchsorted(cumulative,lo+1));b=int(np.searchsorted(cumulative,hi+1))
    return a+(b-a)*(rank-lo)


def divisors_from_histograms(hist,panel,percentile=99.):
    hist=np.asarray(hist);levels=hist.shape[1];dtype_max=levels-1
    if levels not in (256,65536) or hist.shape[0]!=len(panel):raise ValueError('Expected uint8 or uint16 histograms in panel order')
    records=[];divisors=[]
    for marker,counts in zip(panel,hist):
        n=int(counts.sum());q=histogram_percentile(counts,percentile)
        fallback=q<=0;divisor=float(dtype_max) if fallback else q
        last=int(np.flatnonzero(counts)[-1]);first=int(np.flatnonzero(counts)[0])
        clipped=int(counts[int(np.floor(divisor))+1:].sum()) if divisor<dtype_max else 0
        records.append({'marker':marker,'percentile_value':q,'effective_divisor':divisor,
            'fallback_dtype_max':bool(fallback),'zero_fraction':float(counts[0]/n),
            'clipped_fraction':clipped/n,'dtype_max_fraction':float(counts[-1]/n),
            'observed_min':first,'observed_max':last,'pixels':n,
            'very_low_percentile_warning':bool(q<=1)})
        divisors.append(divisor)
    return np.asarray(divisors,float),records


def normalize_pixels(pixels,divisors):
    pixels=np.asarray(pixels);d=np.asarray(divisors,float)
    if pixels.shape[-1]!=len(d) or not np.isfinite(pixels).all() or not np.isfinite(d).all() or (d<=0).any():
        raise ValueError('Invalid pixels/divisors')
    if (pixels<0).any():raise ValueError('Negative raw intensity')
    return np.clip(pixels.astype(np.float64)/d,0,1)


def target_groups(reader,xy,order,divisors,tile_size=512,progress=None):
    """Read selected targets grouped by TIFF tile; return raw and normalized means."""
    xy=np.asarray(xy);n=len(xy);m=len(order)
    if xy.shape!=(n,2) or not np.issubdtype(xy.dtype,np.integer) or len(np.unique(xy,axis=0))!=n:
        raise ValueError('Unique integer target centers required')
    starts=xy-4
    if (starts<0).any() or (starts[:,0]+8>reader.width).any() or (starts[:,1]+8>reader.height).any():
        raise ValueError('Target block outside registered image')
    groups={}
    for i,(x,y) in enumerate(starts):groups.setdefault((int(x)//tile_size,int(y)//tile_size),[]).append(i)
    raw_means=np.empty((n,m),np.float32);normalized=np.empty((n,m),np.float32)
    for step,indices in enumerate(groups.values()):
        coords=starts[indices];x0,y0=coords.min(0);x1,y1=coords.max(0)+8
        raw=reader.read(int(y0),int(y1),int(x0),int(x1))[:,:,order]
        for i in indices:
            x,y=starts[i]-[x0,y0];block=raw[y:y+8,x:x+8]
            raw_means[i]=block.mean((0,1),dtype=np.float64)
            normalized[i]=normalize_pixels(block,divisors).mean((0,1))
        if progress and (step%50==0 or step+1==len(groups)):progress(step+1,len(groups))
    return raw_means,normalized
