"""Read-only environment and matched-loss preflight; no downloads or installation."""
import argparse,json,sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');a=p.parse_args()
    import numpy as np
    import torch
    from config import Config
    from model import Head
    from loss import weighted_huber,bin_targets
    from data import fit_scaler,scale
    from expression import fit_expression,element_weights
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    raw=np.array([[0.,10.],[20.,30.],[255.,255.]],np.float32)
    patient=np.array(['synthetic_a']*3);panel=['synthetic_A','synthetic_B']
    z=torch.from_numpy(scale(raw,fit_scaler(raw,patient))).to(a.device)
    weights=torch.from_numpy(element_weights(raw,fit_expression(raw,patient,panel))).to(a.device)
    ny=torch.from_numpy(bin_targets(raw,255.)).to(a.device)
    model=Head(2,Config()).to(a.device)
    loss=weighted_huber(model(torch.zeros(3,1536,device=a.device),ny,0),z,weights)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();assert torch.isfinite(model(torch.zeros(1,1536,device=a.device))).all()
    print(json.dumps({'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
        'torch':str(torch.__version__),'device':a.device,'loss_and_backward':'passed',
        'head':'1536 -> 256 GELU dropout0.1 -> markers','loss_scale':'training-only z score',
        'bin_scale':'raw / 255','encoder_loaded':False},indent=2))

if __name__=='__main__':main()
