"""Expression-weighted Huber in TRAINING-FITTED Z-SCORE units, identical in both modes."""
import numpy as np


def bin_targets(raw,intensity_max):
    raw=np.asarray(raw)
    if not np.isfinite(intensity_max) or intensity_max<=0 or not np.isfinite(raw).all():
        raise ValueError('Invalid bin intensity scale or targets')
    if (raw<0).any() or (raw>intensity_max).any():raise ValueError('Targets outside declared binning range')
    return (raw/intensity_max).astype(np.float32)


def weighted_huber(prediction,standardized_target,weights):
    import torch
    if prediction.shape!=standardized_target.shape or prediction.shape!=weights.shape:
        raise ValueError('Predictions, z-score targets and per-element weights must match')
    return (torch.nn.functional.huber_loss(prediction,standardized_target,reduction='none',delta=1.)*weights).mean()
