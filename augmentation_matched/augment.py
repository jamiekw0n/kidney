"""Deterministic center-preserving D4 and mild RGB augmentation. No image/target translation."""
import hashlib
import numpy as np
from PIL import Image,ImageEnhance

RECIPE={'parameter_columns':['D4_code','brightness','contrast','saturation','hue'],'version':1,'geometry':'one non-identity of 8 unique D4 transforms; same geometry for both views',
 'brightness':[.9,1.1],'contrast':[.9,1.1],'saturation':[.9,1.1],'hue':[-.01,.01],
 'order':'128native crop -> D4 -> optional brightness,contrast,saturation,hue -> existing 224 resize/ImageNet transform',
 'targets':'unchanged normalized central8x8 means','identity_probability':.5}

def parameters(sample,x,y,seed=42):
    token=f'{seed}|{sample}|{int(x)}|{int(y)}'.encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
    # Codes1..7 exclude identity; 0..3 rotate,4..7 reflect then rotate.
    return np.array([rng.integers(1,8),rng.uniform(.9,1.1),rng.uniform(.9,1.1),rng.uniform(.9,1.1),rng.uniform(-.01,.01)],np.float64)

def geometry(rgb,code):
    code=int(code)
    if not 0<=code<8:raise ValueError('D4 code outside 0..7')
    image=np.fliplr(rgb) if code>=4 else rgb
    return np.ascontiguousarray(np.rot90(image,code%4))

def views(rgb,params):
    if rgb.shape!=(128,128,3) or rgb.dtype!=np.uint8:raise ValueError('Expected native128 uint8 RGB')
    g=geometry(rgb,params[0]);image=Image.fromarray(g)
    image=ImageEnhance.Brightness(image).enhance(float(params[1]))
    image=ImageEnhance.Contrast(image).enhance(float(params[2]))
    image=ImageEnhance.Color(image).enhance(float(params[3]))
    hsv=np.array(image.convert('HSV'),copy=True)
    hsv[:,:,0]=((hsv[:,:,0].astype(np.int16)+int(round(float(params[4])*255)))%256).astype(np.uint8)
    image=Image.frombytes('HSV',(hsv.shape[1],hsv.shape[0]),hsv.tobytes()).convert('RGB')
    return Image.fromarray(g),image
