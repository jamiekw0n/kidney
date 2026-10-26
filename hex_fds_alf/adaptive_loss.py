"""Per-marker adaptive Barron NLL, including log(scale) AND log Z(alpha).

Uses differentiable float64 Gauss-Legendre quadrature on r=tan(pi*u/2), u in (0,1).
This replaces the external reference package's spline resource, not its objective.
"""
import math
import numpy as np
import torch
from torch import nn
from alf_math import quadrature_rule


def rho(residual,alpha,scale):
    beta=2-alpha
    return beta/alpha*torch.expm1(alpha/2*torch.log1p((residual/scale).square()/beta))


class AdaptiveLoss(nn.Module):
    def __init__(self,markers,config):
        super().__init__()
        self.lo=config.alpha_lo; self.hi=config.alpha_hi; self.floor=config.scale_floor
        initial=(config.alpha_init-self.lo)/(self.hi-self.lo)
        self.raw_alpha=nn.Parameter(torch.full((markers,),math.log(initial/(1-initial))))
        # Reference util.affine_softplus: floor + (initial-floor)*softplus(raw+inv_softplus(1)).
        self.initial_scale=config.scale_init
        self.raw_scale=nn.Parameter(torch.zeros(markers))
        r,logw=quadrature_rule(config.quadrature_nodes)
        self.register_buffer('quadrature_r',torch.tensor(r,dtype=torch.float64))
        self.register_buffer('quadrature_logw',torch.tensor(logw,dtype=torch.float64))

    def alpha(self): return self.lo+(self.hi-self.lo)*self.raw_alpha.sigmoid()

    def scale(self):
        return self.floor+(self.initial_scale-self.floor)*torch.nn.functional.softplus(self.raw_scale+math.log(math.expm1(1.)))

    def log_partition(self,alpha):
        # [Q,M], no residual-dependent normalization or stop-gradient approximation.
        a=alpha.to(torch.float64)[None,:]
        values=self.quadrature_logw[:,None]-rho(self.quadrature_r[:,None],a,torch.ones_like(a))
        return torch.logsumexp(values,dim=0)

    def forward(self,residual):
        if residual.ndim!=2 or residual.shape[1]!=len(self.raw_alpha):
            raise ValueError('ALF expects [batch, markers] residuals')
        # FP64 prevents loss of precision for small alpha and the normalization tail.
        alpha=self.alpha().double();scale=self.scale().double()
        return rho(residual.double(),alpha,scale)+scale.log()+self.log_partition(alpha)

    @torch.no_grad()
    def diagnostics(self):
        return {'alpha':self.alpha().cpu().tolist(),'scale':self.scale().cpu().tolist(),
                'log_partition':self.log_partition(self.alpha()).cpu().tolist()}
