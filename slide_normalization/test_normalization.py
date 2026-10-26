import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from normalization import histogram_percentile,divisors_from_histograms,normalize_pixels,target_groups,recipe,UNITS
from normalize_sample import normalize


class ArrayReader:
    def __init__(self,array):
        self.pixels=array;self.height,self.width=array.shape[:2];self.dtype=array.dtype
        self.axes='YXC';self.array=SimpleNamespace(chunks=array.shape)
    def read(self,y0,y1,x0,x1):return self.pixels[y0:y1,x0:x1]
    def close(self):pass


class NormalizationTests(unittest.TestCase):
    def test_exact_histogram_percentiles(self):
        rng=np.random.default_rng(42)
        for dtype in [np.uint8,np.uint16]:
            values=rng.integers(0,200,size=1003,dtype=dtype)
            counts=np.bincount(values,minlength=np.iinfo(dtype).max+1)
            for p in [0,50,95,99,100]:
                self.assertAlmostEqual(histogram_percentile(counts,p),np.percentile(values,p),places=10)

    def test_zero_percentile_fallback_and_per_channel_divisors(self):
        values=np.zeros((1000,2),np.uint8);values[:5,0]=200;values[:,1]=50
        hist=np.stack([np.bincount(values[:,j],minlength=256) for j in range(2)])
        d,records=divisors_from_histograms(hist,['sparse','constant'])
        np.testing.assert_array_equal(d,[255.,50.])
        self.assertTrue(records[0]['fallback_dtype_max'])
        self.assertEqual(records[0]['zero_fraction'],.995)
        self.assertFalse(records[1]['fallback_dtype_max'])
        self.assertGreater(normalize_pixels(np.array([[200,50]],np.uint8),d)[0,0],0)

    def test_pixel_clipping_must_precede_block_mean(self):
        pixels=np.zeros((8,8,1),np.uint8);pixels[0,0,0]=255
        target=normalize_pixels(pixels,[10.]).mean()
        self.assertEqual(target,1/64)
        wrong=np.clip(pixels.mean()/10,0,1)
        self.assertNotAlmostEqual(target,wrong)

    def test_grouped_targets_keep_center_order_and_reorder_channels(self):
        array=np.zeros((16,16,2),np.uint8);array[:,:,0]=10;array[:,:,1]=40
        xy=np.array([[12,12],[4,4],[12,4],[4,12]])
        raw,y=target_groups(ArrayReader(array),xy,[1,0],[80,20],tile_size=8)
        np.testing.assert_array_equal(raw,np.array([[40.,10.]]*4))
        np.testing.assert_array_equal(y,np.full((4,2),.5))

    def test_rewrite_unchanged_features_and_raw_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();rawdir=root/'raw';rawdir.mkdir()
            he=rawdir/'example_HandE.tiff';mif=rawdir/'example.qptiff'
            he.write_bytes(b'SYNTHETIC HEADER ONLY');mif.write_bytes(b'SYNTHETIC HEADER ONLY')
            array=np.zeros((16,16,2),np.uint8);array[:,:,0]=10;array[0,0,0]=255
            array[:8,:,1]=20;array[8:,:,1]=40
            xy=np.array([[4,4],[12,4],[4,12],[12,12]])
            raw,_=target_groups(ArrayReader(array),xy,[0,1],[1,1],8)
            x=np.random.default_rng(3).normal(size=(4,1536)).astype('float32')
            np.savez(source/'p.npz',features=x,xy=xy,target=raw)
            meta=dict(complete=True,sample_id='synthetic_sample',patient_id='synthetic_patient',panel=['A','B'],
                      encoder='UNI2-h',input_patch=128,target_block=8,he_mpp=.5,shape=[16,16],shards=['p.npz'],
                      he=str(he),mif=str(mif),revision='synthetic',checkpoint_sha256='synthetic',transform='synthetic')
            (source/'manifest.json').write_text(json.dumps(meta));out=root/'derived'
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [he,mif,source/'p.npz']}
            info={'markers':['A','B'],'shape':[16,16,2],'axes':'YXC','mpp_xy':[.5,.5]}
            with patch('normalize_sample.header',return_value=info),patch('normalize_sample.TiffRegionReader',side_effect=lambda *a,**kw:ArrayReader(array)):
                normalize(source,out,tile_size=8)
                with self.assertRaises(FileExistsError):normalize(source,out,tile_size=8)
            m=json.loads((out/'manifest.json').read_text());self.assertEqual(m['target_units'],UNITS)
            with self.assertRaisesRegex(ValueError,'double normalization'):normalize(out,root/'double',tile_size=8)
            with np.load(out/'p.npz') as z:
                np.testing.assert_array_equal(z['features'],x);np.testing.assert_array_equal(z['xy'],xy)
                np.testing.assert_array_equal(z['raw_target'],raw)
                self.assertTrue(((z['target']>=0)&(z['target']<=1)).all())
                # Pixel 255 is capped to1 before averaging, unlike simple block rescaling.
                self.assertAlmostEqual(z['target'][0,0],1.)
            self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
            np.savez(source/'p.npz',features=x,xy=xy,target=raw+1)
            with patch('normalize_sample.header',return_value=info),patch('normalize_sample.TiffRegionReader',side_effect=lambda *a,**kw:ArrayReader(array)):
                with self.assertRaisesRegex(ValueError,'cached means differ'):normalize(source,root/'bad',tile_size=8)
            self.assertFalse((root/'bad/manifest.json').exists())

if __name__=='__main__':unittest.main()
