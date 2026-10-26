"""Original 1536->256 GELU/dropout0.1->markers head with optional training-only FDS."""
import torch
from torch import nn
from torch.nn import functional as F
from config import Config
from fds import FeatureSmoothing


class Head(nn.Module):
    def __init__(self,markers,config):
        super().__init__();self.config=config.validate()
        self.hidden=nn.Sequential(nn.Linear(config.input_dim,config.hidden),nn.GELU(),nn.Dropout(config.dropout))
        self.output=nn.Linear(config.hidden,markers)
        self.fds=FeatureSmoothing(markers,config.hidden,config)

    def forward(self,x,bin_targets=None,epoch=0,return_features=False):
        h=self.hidden(x)
        if self.fds.active(epoch):
            if bin_targets is None:raise ValueError('FDS training requires raw/intensity_max bin targets, NOT z scores')
            pred=torch.cat([F.linear(self.fds.calibrate(h,bin_targets,j,epoch),self.output.weight[j:j+1],
                                     self.output.bias[j:j+1]) for j in range(self.output.out_features)],dim=1)
        else:pred=self.output(h)
        return (pred,h) if return_features else pred


def load_checkpoint(path,device='cpu'):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if ckpt.get('format')!='kidney_expression_weighted_fds_v1':raise ValueError('Wrong checkpoint format for this head')
    model=Head(len(ckpt['panel']),Config(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    model.requires_grad_(False).eval()
    return model,ckpt
