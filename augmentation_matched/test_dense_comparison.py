"""Synthetic plotting/mask checks; real UNI2 inference requires workstation validation."""
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import dense_comparison as d
from dense_spatial import inference_mask

class DenseComparisonTests(unittest.TestCase):
    def test_legacy_reference_encoder_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp);(source/'bands/band_00000').mkdir(parents=True)
            slide={'sample_id':'S','fold':0,'bands':1,'markers':['CAV'],'grid_shape':[2,3]}
            d.save(source/'slide.json',slide)
            reference={**slide,'source_slide_plan':str(source),'normalization_sha256':'norm'}
            band={**slide,'complete':True,'band':0,'encoder_sha256':'encoder',
                  'normalization_sha256':'norm','slide_plan_sha256':d.sha(source/'slide.json')}
            d.save(source/'bands/band_00000/manifest.json',band)
            d.verify_reference_encoder(reference,'encoder')
            with self.assertRaisesRegex(ValueError,'encoder hash differs'):d.verify_reference_encoder(reference,'wrong')
            with self.assertRaisesRegex(ValueError,'encoder hash differs'):d.verify_reference_encoder({'encoder_sha256':'wrong'},'encoder')

    def test_original_support_and_centers(self):
        rgb=np.full((256,256,3),[150,70,110],np.uint8)
        xy,valid=inference_mask(rgb,0,32)
        self.assertTrue(np.all(xy%8==4));self.assertEqual(int(valid.sum()),16*16)
        self.assertTrue(np.all(xy[valid]>=64));self.assertTrue(np.all(xy[valid]+64<=256))
    def test_five_panel_gallery_and_preserved_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=root/'reference';ref.mkdir();out=root/'new';(out/'bands/00000').mkdir(parents=True)
            np.save(ref/'measured.npy',np.full((2,3,1),.2,np.float32));np.save(ref/'he.npy',np.zeros((2,3,3),np.uint8));np.save(ref/'valid.npy',np.ones((2,3),bool))
            d.save(out/'comparison_plan.json',{'bands':1,'markers':['CAV']})
            band=out/'bands/00000';d.save(band/'complete.json',{'start':0,'stop':2,'plan_sha256':d.sha(out/'comparison_plan.json')})
            for v in d.VARIANTS:np.save(band/(v+'.npy'),np.full((2,3,1),1.2,np.float32))
            a=SimpleNamespace(plan='unused',reference=str(ref),out=str(out),sample='S')
            with patch.object(d,'inputs',return_value=({}, {},{'panel':['CAV']},{'markers':['CAV']},0)):d.plot(a)
            self.assertTrue((out/'figures/marker_00.png').exists());self.assertTrue(d.read(out/'complete.json')['complete'])
            self.assertAlmostEqual(float(np.load(band/'control.npy')[0,0,0]),1.2,places=6)
            self.assertIn('no test-time augmentation',(out/'figures/index.html').read_text())
if __name__=='__main__':unittest.main()
