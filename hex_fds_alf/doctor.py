"""Read-only dependency and normalized-loss preflight; never installs or downloads."""
import argparse
import json
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');a=p.parse_args()
    import numpy as np
    import scipy
    import torch
    from config import Config
    from adaptive_loss import AdaptiveLoss
    from alf_math import log_partition_numpy
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    loss=AdaptiveLoss(2,Config()).to(a.device)
    residual=torch.tensor([[.1,.5],[1.,2.]],device=a.device,requires_grad=True)
    value=loss(residual).mean();value.backward()
    assert torch.isfinite(loss.raw_alpha.grad).all() and torch.isfinite(loss.raw_scale.grad).all()
    assert abs(float(loss.log_partition(torch.tensor([1.],device=a.device)))-log_partition_numpy(1.,512))<1e-6
    print(json.dumps({'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
        'scipy':scipy.__version__,'torch':str(torch.__version__),'device':a.device,
        'gpu':torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,
        'normalization_and_backward':'passed','network_or_model_downloads':False},indent=2))

if __name__=='__main__':main()
