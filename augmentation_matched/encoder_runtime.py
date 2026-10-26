"""Exact frozen UNI2-h architecture and original full-context transform (FP32)."""
import hashlib
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_encoder(checkpoint,expected_sha,device):
    import torch,timm
    from torchvision import transforms
    actual=sha(checkpoint)
    if actual!=expected_sha:raise ValueError('UNI2-h checkpoint hash differs from reference cache')
    model=timm.create_model('vit_giant_patch14_224',pretrained=False,img_size=224,patch_size=14,depth=24,num_heads=24,
        init_values=1e-5,embed_dim=1536,mlp_ratio=2.66667*2,num_classes=0,no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,act_layer=torch.nn.SiLU,reg_tokens=8,dynamic_img_size=True)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False).eval().to(device)
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return model,transform

def encode(model,transform,he,xy,device):
    import torch
    from PIL import Image
    batch=torch.stack([transform(Image.fromarray(he[int(y)-64:int(y)+64,int(x)-64:int(x)+64])) for x,y in xy]).to(device)
    with torch.inference_mode():features=model(batch)
    if features.shape!=(len(xy),1536) or not torch.isfinite(features).all():raise ValueError('Invalid UNI2-h output')
    return features.cpu().float().numpy()
