"""Matched one-hidden-layer MLP experiment; regression scale and bin scale differ."""
from dataclasses import dataclass,asdict
import math


@dataclass
class Config:
    mode: str = 'weighted_fds'
    input_dim: int = 1536
    hidden: int = 256
    dropout: float = .1
    intensity_max: float = 1.
    bins: int = 50
    kernel_size: int = 9
    sigma: float = 2.
    momentum: float = .9
    fds_start: int = 10
    fds_off: int = 50
    min_bin_count: int = 2
    ratio_min: float = .1
    ratio_max: float = 10.

    def validate(self):
        if self.mode not in ('weighted','weighted_fds'):raise ValueError('Mode must be weighted or weighted_fds')
        if self.input_dim!=1536 or self.hidden<1 or self.bins<2:raise ValueError('Invalid head/bin dimensions')
        if not all(math.isfinite(v) for v in (self.dropout,self.intensity_max,self.sigma,self.momentum,self.ratio_min,self.ratio_max)):
            raise ValueError('Nonfinite configuration')
        if not 0<=self.dropout<1 or self.intensity_max<=0 or self.sigma<=0 or not 0<=self.momentum<1:
            raise ValueError('Invalid dropout/intensity/sigma/momentum')
        if self.kernel_size<1 or self.kernel_size%2!=1 or self.kernel_size//2>=self.bins:
            raise ValueError('Need odd kernel and reflect padding smaller than bin count')
        if not 0<=self.fds_start<self.fds_off or self.min_bin_count<2:raise ValueError('Invalid FDS schedule/minimum support')
        if not 0<self.ratio_min<=1<=self.ratio_max:raise ValueError('Invalid variance-ratio caps')
        return self

    @property
    def use_fds(self):return self.mode=='weighted_fds'

    def to_dict(self):return asdict(self)


def add_config_arguments(parser):
    for key,value in Config().to_dict().items():
        if key!='input_dim':parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)


def config_from_args(args):
    return Config(**{key:getattr(args,key) for key in Config().to_dict() if key!='input_dim'}).validate()
