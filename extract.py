"""Cache frozen UNI2-h CLS embeddings for the existing prepared kidney shards."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from data import read_meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', required=True, help='Local approved UNI2-h pytorch_model.bin')
    p.add_argument('--revision', required=True, help='Exact Hugging Face checkpoint commit SHA')
    p.add_argument('--he', help='Optional replacement H&E path on the SAME registered grid')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--he-cache-gb', type=float, default=0, help='Decode single-chunk H&E once if raw image fits this GiB budget; 0 disables')
    a = p.parse_args()
    if a.batch_size < 1 or a.he_cache_gb < 0:
        p.error('batch-size must be positive')
    if not re.fullmatch(r'[0-9a-f]{40}', a.revision):
        p.error('revision must be a full 40-character checkpoint commit SHA')
    root, out = Path(a.prepared), Path(a.out)
    from validate import validate
    validate(root)
    meta = read_meta(root)
    if meta['input_patch'] != 128 or meta['target_block'] != 8:
        raise ValueError('This baseline requires 128-pixel context and 8-pixel targets')
    if out.exists():
        raise FileExistsError(out)
    import torch
    import timm
    from PIL import Image
    from image_io import TiffRegionReader, CachedRegionReader, read_he_patch
    started=time.monotonic()
    print('Initializing UNI2-h architecture',flush=True)
    model = timm.create_model('vit_giant_patch14_224', pretrained=False,
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667*2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True)
    print('Loading local checkpoint on CPU',flush=True)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True), strict=True)
    print(f'Moving frozen model to {a.device}',flush=True)
    model.requires_grad_(False).eval().to(a.device)
    # Explicit square resize preserves the full central context without center cropping.
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    digest = hashlib.sha256()
    with open(a.checkpoint, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    out.mkdir(parents=True)
    print('Opening registered H&E',flush=True)
    reader = TiffRegionReader(a.he or meta['he'], 3, meta['he_axes'], meta['he_series'], meta['he_level'])
    shards = []
    try:
        if reader.dtype != np.uint8:
            raise ValueError('H&E must be uint8 RGB')
        if [reader.height, reader.width] != meta['shape']:
            raise ValueError('Replacement H&E dimensions do not match prepared grid')
        raw_bytes=reader.height*reader.width*3*reader.dtype.itemsize
        cached=False
        if a.he_cache_gb and tuple(reader.array.chunks)==tuple(reader.array.shape) and raw_bytes <= a.he_cache_gb*1024**3:
            print(f'Decoding single-chunk H&E once ({raw_bytes} raw bytes)',flush=True)
            before=time.monotonic()
            reader=CachedRegionReader(reader); cached=True
            print(f'H&E cache ready in {time.monotonic()-before:.2f} seconds',flush=True)
        else:
            print(f'Using regional H&E reads; chunks={reader.array.chunks}, raw_bytes={raw_bytes}',flush=True)
        for name in meta['shards']:
            with np.load(root/name, allow_pickle=False) as z:
                xy, y = z['xy'].copy(), z['measured'].copy()
            if (xy.ndim != 2 or xy.shape[1] != 2 or not np.issubdtype(xy.dtype, np.integer)
                    or len(xy) == 0):
                raise ValueError('Coordinates must be a nonempty integer [N,2] array')
            if y.shape != (len(xy), len(meta['panel'])) or not np.isfinite(y).all():
                raise ValueError('Invalid measured kidney targets')
            features = np.empty((len(xy), 1536), dtype=np.float32)
            for start in range(0, len(xy), a.batch_size):
                batch = torch.stack([transform(Image.fromarray(read_he_patch(reader, int(x), int(y))))
                                     for x,y in xy[start:start+a.batch_size]]).to(a.device)
                with torch.inference_mode():
                    emb = model(batch)
                if emb.shape != (len(batch), 1536) or not torch.isfinite(emb).all():
                    raise ValueError('Unexpected UNI2-h embedding')
                features[start:start+len(batch)] = emb.cpu().float().numpy()
                if start == 0 or (start // a.batch_size) % 25 == 0 or start+len(batch) == len(xy):
                    print(f'{name}: {start+len(batch)}/{len(xy)} features', flush=True)
            np.savez(out/name, features=features, target=y, xy=xy)
            shards.append(name)
            print(f'Cached {name}: {len(xy)} crops', flush=True)
    finally:
        reader.close()
    saved = {k: meta[k] for k in ('sample_id','patient_id','panel','shape','he_mpp','input_patch','target_block')}
    import shutil
    if (root/'filtering.json').exists():
        shutil.copy2(root/'filtering.json',out/'filtering.json')
    saved.update(he_read_cache={'enabled':cached,'budget_gib':a.he_cache_gb,'raw_bytes':raw_bytes},elapsed_seconds=time.monotonic()-started,diagnostic_only=meta.get('diagnostic_only',False),selection_policy=meta.get('selection_policy'),encoder='UNI2-h', revision=a.revision, checkpoint_sha256=digest.hexdigest(),
        transform='PIL bilinear square resize 224; ToTensor; ImageNet normalization',
        he=str(Path(a.he or meta['he']).resolve()), source=str(root.resolve()),
        torch_version=torch.__version__, timm_version=timm.__version__, shards=shards, complete=True)
    (out/'manifest.json').write_text(json.dumps(saved, indent=2))

if __name__ == '__main__':
    main()
