"""Numpy-only cache validation, patient partitions, and training target scaling."""
import json
from pathlib import Path
import numpy as np


def read_meta(root):
    meta = json.loads((Path(root) / 'manifest.json').read_text())
    if not meta.get('complete'):
        raise ValueError(f'Incomplete dataset: {root}')
    if not meta.get('patient_id') or meta['patient_id'].lower() == 'unknown':
        raise ValueError(f'Actual patient_id required: {root}')
    return meta


def partitions(caches, split):
    if set(split) != {'train', 'val', 'test'}:
        raise ValueError('Split JSON must contain train, val, test patient lists')
    seen = set()
    for group, patients in split.items():
        if len(patients) != len(set(patients)) or seen.intersection(patients):
            raise ValueError('Patient leakage or duplicate patient in split')
        seen.update(patients)
    if not split['train'] or not split['val']:
        raise ValueError('Independent training and validation patients required')
    result = {k: [] for k in split}
    reference = None
    available = set()
    samples = set()
    for root in caches:
        meta = read_meta(root)
        signature = (meta['panel'], meta['encoder'], meta['revision'], meta['checkpoint_sha256'], meta['transform'],
                     meta['input_patch'], meta['he_mpp'], meta['target_block'], meta.get('selection_policy'), meta.get('target_units'), meta.get('normalization_recipe'))
        if reference is not None and signature != reference:
            raise ValueError('Caches differ in panel, encoder, preprocessing, or physical scale')
        reference = signature
        if meta['sample_id'] in samples:
            raise ValueError('Duplicate sample cache')
        samples.add(meta['sample_id'])
        patient = meta['patient_id']
        if patient not in seen:
            raise ValueError(f'Patient absent from split: {patient}')
        available.add(patient)
        group = next(k for k, v in split.items() if patient in v)
        result[group].append(Path(root))
    if available != seen:
        raise ValueError(f'Split patients without caches: {seen - available}')
    return result


def fit_scaler(targets, patients, log1p=False, intensity_max=None):
    """Equal-patient moments; targets are transformed before standardization."""
    if intensity_max is not None:
        if log1p or not np.isfinite(intensity_max) or intensity_max <= 0:
            raise ValueError('Fixed 0-1 scaling needs positive intensity_max and no log1p')
        if not np.isfinite(targets).all() or (targets < 0).any() or (targets > intensity_max).any():
            raise ValueError('Measured targets outside declared 0..intensity_max range')
        return {'mean': [0.]*targets.shape[1], 'std': [float(intensity_max)]*targets.shape[1],
                'log1p': False, 'mode': 'fixed_0_1', 'intensity_max': float(intensity_max)}
    means, seconds = [], []
    for patient in np.unique(patients):
        y = targets[patients == patient].astype(np.float64)
        if not np.isfinite(y).all() or (log1p and (y < 0).any()):
            raise ValueError('Targets must be finite; log1p requires nonnegative intensities')
        if log1p:
            y = np.log1p(y)
        means.append(y.mean(0))
        seconds.append((y * y).mean(0))
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds, axis=0) - mean * mean, 0))
    return {'mean': mean.tolist(), 'std': np.maximum(std, 1e-6).tolist(), 'log1p': log1p}


def scale(y, state, inverse=False):
    mean, std = np.asarray(state['mean']), np.asarray(state['std'])
    if inverse:
        y = y * std + mean
        return np.expm1(y) if state['log1p'] else y
    if state.get('mode') == 'fixed_0_1' and (not np.isfinite(y).all() or (y < 0).any() or (y > state['intensity_max']).any()):
        raise ValueError('Measured targets outside declared fixed intensity range')
    if state['log1p']:
        if (y < 0).any():
            raise ValueError('Negative targets with log1p')
        y = np.log1p(y)
    return ((y - mean) / std).astype(np.float32)
