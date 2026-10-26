"""Independent NumPy reference for Barron robust density normalization.

No external robust-loss source or spline asset is needed. Torch implementation
uses the same integral, with autograd through its integrand. See HEX_CROSSCHECK.md.
"""
import numpy as np


def quadrature_rule(nodes=256):
    if nodes<64: raise ValueError('At least 64 quadrature nodes required')
    t,w=np.polynomial.legendre.leggauss(nodes)
    u=(t+1)/2
    angle=np.pi*u/2
    r=np.tan(angle)
    # Z = 2 integral_0^inf exp(-rho(r)) dr; factor 2 cancels du weight /2.
    log_weights=np.log(w)+np.log(np.pi/2)-2*np.log(np.cos(angle))
    return r,log_weights


def rho_numpy(residual,alpha,scale=1.):
    if not 0<alpha<2 or scale<=0: raise ValueError('Interior alpha and positive scale required')
    beta=2-alpha
    return beta/alpha*np.expm1(alpha/2*np.log1p((np.asarray(residual)/scale)**2/beta))


def log_partition_numpy(alpha,nodes=256):
    r,logw=quadrature_rule(nodes)
    value=logw-rho_numpy(r,alpha)
    maximum=np.max(value)
    return float(maximum+np.log(np.exp(value-maximum).sum()))
