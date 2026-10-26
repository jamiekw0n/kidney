"""Single serialized configuration contract shared by training, CV and inference."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Config:
    mode: str = 'fds_alf'
    input_dim: int = 1536
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = .5
    intensity_max: float = 255.
    bins: int = 50
    kernel_size: int = 9
    sigma: float = 2.
    momentum: float = .9
    fds_start: int = 10
    fds_off: int = 50
    min_bin_count: int = 2
    ratio_min: float = .1
    ratio_max: float = 10.
    quadrature_nodes: int = 512
    alpha_lo: float = .001
    alpha_hi: float = 1.999
    alpha_init: float = 1.
    scale_floor: float = 1e-5
    scale_init: float = 1.

    def validate(self):
        if self.mode not in ('control', 'fds', 'alf', 'fds_alf'):
            raise ValueError('Invalid ablation mode')
        if min(self.input_dim,self.hidden1,self.hidden2,self.bins,self.min_bin_count)<1:
            raise ValueError('Dimensions/counts must be positive')
        vals=[self.dropout,self.intensity_max,self.sigma,self.momentum,self.ratio_min,
              self.ratio_max,self.alpha_lo,self.alpha_hi,self.alpha_init,self.scale_floor,self.scale_init]
        if not all(math.isfinite(v) for v in vals): raise ValueError('Nonfinite configuration')
        if not 0<=self.dropout<1 or self.intensity_max<=0 or self.sigma<=0 or not 0<=self.momentum<1:
            raise ValueError('Invalid dropout/range/sigma/momentum')
        if self.kernel_size<1 or self.kernel_size%2!=1 or self.kernel_size//2>=self.bins:
            raise ValueError('Odd kernel and reflect padding smaller than bin count required')
        if not 0<=self.fds_start<self.fds_off or self.min_bin_count<2:
            raise ValueError('Need 0 <= FDS start < off and count >=2')
        if not 0<self.ratio_min<=1<=self.ratio_max:
            raise ValueError('Invalid variance-ratio limits')
        if not 0<self.alpha_lo<self.alpha_init<self.alpha_hi<2:
            raise ValueError('ALF alpha must remain strictly between 0 and 2')
        if not 0<self.scale_floor<self.scale_init or self.quadrature_nodes<64:
            raise ValueError('Invalid ALF scale or quadrature order')
        return self

    @property
    def use_fds(self): return self.mode in ('fds','fds_alf')

    @property
    def use_alf(self): return self.mode in ('alf','fds_alf')

    def to_dict(self): return asdict(self)


def add_config_arguments(parser):
    for name,value in Config().to_dict().items():
        if name=='input_dim': continue
        parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)


def config_from_args(args):
    return Config(**{key:getattr(args,key) for key in Config().to_dict() if key!='input_dim'}).validate()
