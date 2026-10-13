"""Train only a regression head, with patient-balanced sampling and validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import partitions, read_meta, fit_scaler, scale
from metrics import metrics


def load_group(roots):
    xs, ys, patients = [], [], []
    for root in roots:
        meta = read_meta(root)
        for name in meta['shards']:
            with np.load(root/name, allow_pickle=False) as z:
                x, y = z['features'].copy(), z['target'].copy()
            if x.shape != (len(y), 1536) or y.shape[1] != len(meta['panel']):
                raise ValueError('Invalid cache shapes')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite cache values')
            xs.append(x); ys.append(y)
            patients.extend([meta['patient_id']] * len(y))
    if not xs or not sum(map(len, xs)):
        raise ValueError('Empty partition')
    return np.concatenate(xs), np.concatenate(ys), np.asarray(patients)


def make_head(torch, markers, hidden):
    if hidden == 0:
        return torch.nn.Linear(1536, markers)
    return torch.nn.Sequential(torch.nn.Linear(1536, hidden), torch.nn.GELU(),
                               torch.nn.Dropout(0.1), torch.nn.Linear(hidden, markers))


def predict(torch, model, x, device, batch_size):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size]).to(device)).cpu().numpy()
                               for i in range(0, len(x), batch_size)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--caches', nargs='+', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--hidden', type=int, default=256, help='0 for linear baseline')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log1p', action='store_true', help='Only for nonnegative measured intensities')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.epochs, a.patience, a.batch_size) < 1 or a.hidden < 0 or a.lr <= 0:
        p.error('Invalid training settings')
    split = json.loads(Path(a.split).read_text())
    from validate import validate
    for root in a.caches:
        validate(root)
    groups = partitions(a.caches, split)
    meta = read_meta(groups['train'][0])
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    x, y, patients = load_group(groups['train'])
    vx, vy, vp = load_group(groups['val'])
    scaler = fit_scaler(y, patients, a.log1p)
    sy, svy = scale(y, scaler), scale(vy, scaler)
    model = make_head(torch, len(meta['panel']), a.hidden).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    counts = {patient: int((patients == patient).sum()) for patient in np.unique(patients)}
    weights = torch.as_tensor([1/counts[patient] for patient in patients], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(x), replacement=True)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(sy))
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch_size, sampler=sampler)
    out.mkdir(parents=True)
    from audit import snapshot, event, write_json
    snapshot(out,vars(a))
    write_json(out/'target_scaler.json',scaler)
    event(out,'training_start',split=split,train_rows=len(x),validation_rows=len(vx),
          head_hidden=a.hidden,encoder_frozen=True,optimizer='AdamW',lr=a.lr,weight_decay=1e-4,
          loss='Huber delta=1, equal standardized marker weights',sampling='equal expected patient sampling',
          torch_version=torch.__version__,device=a.device)
    history, best, stale = [], float('inf'), 0
    for epoch in range(a.epochs):
        model.train()
        total, count = 0., 0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.huber_loss(model(bx.to(a.device)), by.to(a.device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx); count += len(bx)
        valpred = predict(torch, model, vx, a.device, a.batch_size)
        # Huber loss averaged within each patient, then across patients and markers.
        err = np.abs(valpred - svy)
        losses = np.where(err <= 1, 0.5*err**2, err-0.5)
        val = float(np.mean([losses[vp == patient].mean() for patient in np.unique(vp)]))
        if not np.isfinite(val):
            raise ValueError('Nonfinite validation loss')
        history.append({'epoch':epoch+1, 'train_huber':total/count, 'val_huber':val})
        print(json.dumps(history[-1]), flush=True)
        event(out,'epoch',**history[-1])
        (out/'history.json').write_text(json.dumps(history, indent=2))
        if val < best:
            best, stale = val, 0
            torch.save({'state_dict':model.state_dict(), 'hidden':a.hidden, 'panel':meta['panel'],
                        'scaler':scaler, 'split':split, 'args':vars(a), 'epoch':epoch+1,
                        'val_huber':val, 'selection_policy':meta.get('selection_policy'), 'cache_signature':{k:meta[k] for k in
                        ('encoder','revision','checkpoint_sha256','transform','input_patch','he_mpp','target_block')}},
                       out/'best.pt')
        else:
            stale += 1
            if stale >= a.patience:
                break
    checkpoint = torch.load(out/'best.pt', map_location=a.device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    raw = scale(predict(torch, model, vx, a.device, a.batch_size), scaler, inverse=True)
    baseline_mean = np.mean([y[patients == patient].mean(0) for patient in np.unique(patients)], axis=0)
    baseline = np.tile(baseline_mean, (len(vy),1))
    report = {'model':metrics(vy,raw,vp,meta['panel']),
              'training_mean_baseline':metrics(vy,baseline,vp,meta['panel'])}
    (out/'validation_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    event(out,'training_complete',best_epoch=checkpoint['epoch'],best_validation_loss=checkpoint['val_huber'])
    (out/'run.json').write_text(json.dumps({'args':vars(a), 'split':split,
        'torch_version':torch.__version__, 'best_epoch':checkpoint['epoch'],
        'train_patients':len(counts), 'val_patients':len(np.unique(vp)),
        'note':'Test partition is not loaded or scored during training.'}, indent=2))

if __name__ == '__main__':
    main()
