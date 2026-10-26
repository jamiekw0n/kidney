"""Frozen-cache regression head; targets are never required at inference."""
import torch
from torch import nn
from torch.nn import functional as F
from fds import FeatureSmoothing
from config import Config


class Head(nn.Module):
    def __init__(self,markers,config):
        super().__init__();self.config=config.validate()
        self.hidden=nn.Sequential(nn.Linear(config.input_dim,config.hidden1),nn.ReLU(),nn.Dropout(config.dropout),
                                  nn.Linear(config.hidden1,config.hidden2),nn.ReLU(),nn.Dropout(config.dropout))
        self.output=nn.Linear(config.hidden2,markers)
        self.fds=FeatureSmoothing(markers,config.hidden2,config)

    def forward(self,x,targets=None,epoch=0,return_features=False):
        h=self.hidden(x)
        if self.fds.active(epoch):
            if targets is None: raise ValueError('Training FDS requires TRAINING targets')
            pred=torch.cat([F.linear(self.fds.calibrate(h,targets,j,epoch),self.output.weight[j:j+1],
                                      self.output.bias[j:j+1]) for j in range(self.output.out_features)],dim=1)
        else:
            pred=self.output(h)
        return (pred,h) if return_features else pred


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='kidney_hex_fds_alf_v1':raise ValueError('Unsupported checkpoint format')
    model=Head(len(checkpoint['panel']),Config(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.requires_grad_(False).eval()
    return model,checkpoint
