"""Marker-conditioned feature moment smoothing with training-only state updates.

Follows HEX's per-marker, per-bin diagonal feature calibration. Differences for
empty bins, initialization and stable accumulation are explicit in HEX_CROSSCHECK.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureSmoothing(nn.Module):
    def __init__(self,markers,dimension,config):
        super().__init__()
        self.config=config
        shape=(markers,config.bins,dimension)
        for name,value in [('running_mean',0.),('running_var',1.),('smooth_mean',0.),('smooth_var',1.)]:
            self.register_buffer(name,torch.full(shape,value))
        self.register_buffer('seen',torch.zeros(markers,config.bins,dtype=torch.long))
        self.register_buffer('last_epoch',torch.tensor(-1,dtype=torch.long))
        k=config.kernel_size//2
        # Match scipy.ndimage.gaussian_filter1d on HEX's length-kernel delta,
        # including its default reflect boundary mode, without requiring SciPy.
        pos=torch.arange(-k,k+1,dtype=torch.float64)
        radius=int(4*config.sigma+.5)
        offsets=torch.arange(-radius,radius+1,dtype=torch.long)
        weights=torch.exp(-offsets.double().square()/(2*config.sigma**2));weights/=weights.sum()
        idx=torch.arange(config.kernel_size)[:,None]+offsets[None,:]
        folded=idx.remainder(2*config.kernel_size)
        reflected=torch.where(folded<config.kernel_size,folded,2*config.kernel_size-1-folded)
        kernel=((reflected==k)*weights).sum(1);kernel/=kernel.sum()
        self.register_buffer('kernel',kernel.float())

    def buckets(self,targets):
        if targets.ndim!=2 or targets.shape[1]!=self.seen.shape[0]: raise ValueError('FDS target shape mismatch')
        if not torch.isfinite(targets).all() or (targets<0).any() or (targets>1).any():
            raise ValueError('FDS requires finite, fixed-range targets in [0,1]')
        return (targets*self.config.bins).long().clamp(max=self.config.bins-1)

    def active(self,epoch):
        return self.training and self.config.use_fds and self.config.fds_start<=epoch<self.config.fds_off

    def new_moments(self):
        return {'count':torch.zeros_like(self.seen),
                'sum':torch.zeros_like(self.running_mean,dtype=torch.float64),
                'sumsq':torch.zeros_like(self.running_mean,dtype=torch.float64)}

    @torch.no_grad()
    def accumulate(self,moments,features,targets):
        if not self.training: raise RuntimeError('Never collect FDS statistics during evaluation')
        if features.shape!=(len(targets),self.running_mean.shape[2]) or not torch.isfinite(features).all():
            raise ValueError('Invalid FDS features')
        b=self.buckets(targets);h=features.detach().double()
        for j in range(targets.shape[1]):
            moments['count'][j].index_add_(0,b[:,j],torch.ones(len(h),dtype=torch.long,device=h.device))
            moments['sum'][j].index_add_(0,b[:,j],h)
            moments['sumsq'][j].index_add_(0,b[:,j],h.square())

    @torch.no_grad()
    def finish_epoch(self,moments,epoch):
        if not self.training: raise RuntimeError('Never update FDS during evaluation')
        if epoch<=int(self.last_epoch): raise ValueError('FDS epochs must increase monotonically')
        n=moments['count'];valid=n>=self.config.min_bin_count
        mean=moments['sum']/n.clamp_min(1)[...,None]
        var=((moments['sumsq']-moments['sum'].square()/n.clamp_min(1)[...,None])/
             (n-1).clamp_min(1)[...,None]).clamp_min(0)
        if not torch.isfinite(mean[valid]).all() or not torch.isfinite(var[valid]).all():
            raise ValueError('Nonfinite FDS moments')
        first=(self.seen==0)&valid;continuing=(self.seen>0)&valid
        self.running_mean[first]=mean[first].float();self.running_var[first]=var[first].float()
        m=self.config.momentum
        self.running_mean[continuing]=m*self.running_mean[continuing]+(1-m)*mean[continuing].float()
        self.running_var[continuing]=m*self.running_var[continuing]+(1-m)*var[continuing].float()
        self.seen+=torch.where(valid,n,torch.zeros_like(n))
        # Smooth along the intensity-bin axis, never marker or feature axes.
        # Exclude never-supported bins from both numerator and denominator.
        occupancy=(self.seen>0).float()
        def smooth(values):
            markers,bins,dimension=values.shape
            v=(values*occupancy[...,None]).permute(0,2,1).reshape(markers*dimension,1,bins)
            pad=self.config.kernel_size//2
            numerator=F.conv1d(F.pad(v,(pad,pad),mode='reflect'),self.kernel[None,None,:])
            denominator=F.conv1d(F.pad(occupancy[:,None,:],(pad,pad),mode='reflect'),self.kernel[None,None,:])
            result=numerator.reshape(markers,dimension,bins).permute(0,2,1)/denominator[:,0,:,None].clamp_min(1e-12)
            return torch.where((denominator[:,0,:]>0)[...,None],result,values)
        self.smooth_mean.copy_(smooth(self.running_mean))
        self.smooth_var.copy_(smooth(self.running_var).clamp_min(0))
        self.last_epoch.fill_(epoch)

    def calibrate(self,features,targets,marker,epoch):
        if not self.active(epoch): return features
        if int(self.last_epoch)<0: raise RuntimeError('FDS smoothing needs completed training moments')
        idx=self.buckets(targets)[:,marker]
        mean=self.running_mean[marker,idx];var=self.running_var[marker,idx]
        sm=self.smooth_mean[marker,idx];sv=self.smooth_var[marker,idx]
        good=(self.seen[marker,idx]>0)[:,None] & (var>1e-12)
        ratio=(sv/var.clamp_min(1e-12)).clamp(self.config.ratio_min,self.config.ratio_max)
        adjusted=(features-mean)*ratio.sqrt()+sm
        # Same zero-variance bypass intent as HEX; no in-place changes to shared h.
        return torch.where(good,adjusted,features)

    @torch.no_grad()
    def diagnostics(self):
        return {'supported_bins_per_marker':(self.seen>0).sum(1).cpu().tolist(),
                'accepted_sample_draws_per_bin':self.seen.cpu().tolist(),
                'last_updated_zero_based_epoch':int(self.last_epoch)}
