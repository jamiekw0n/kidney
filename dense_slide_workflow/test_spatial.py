import csv,io,json,tempfile,unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
from spatial import integral,fractions,tissue_mask,select,inference_mask,band_geometry
from whole_slide import assemble
from encoder_runtime import sha

class SpatialTest(unittest.TestCase):
    def test_exact_tissue_fractions_and_reservoir(self):
        rgb=np.full((384,384,3),[190,80,150],dtype=np.uint8);rgb[120:190,100:230]=255
        xy,counts=select(rgb,32,10,42,csv.writer(io.StringIO()))
        again,_=select(rgb,32,10,42,csv.writer(io.StringIO()))
        np.testing.assert_array_equal(xy,again);self.assertEqual(len(xy),10)
        self.assertEqual(counts['retained']+counts['eligible_not_sampled'],counts['eligible'])
        ii=integral(tissue_mask(rgb));coords=np.array([[128,128],[256,256]])
        for half in (4,64):
            got=fractions(ii,coords,half)
            for (x,y),v in zip(coords,got):self.assertAlmostEqual(v,tissue_mask(rgb[y-half:y+half,x-half:x+half]).mean())
    def test_band_grid_mask_and_complete_coverage(self):
        he=np.full((268,280,3),[180,70,140],dtype=np.uint8)
        ny,nx,start,stop,n=band_geometry(268,280,8,0);seen=[]
        for i in range(n):
            _,_,start,stop,_=band_geometry(268,280,8,i)
            xy,valid=inference_mask(he,start,stop)
            expected=(xy[:,0]>=64)&(xy[:,0]+64<=280)&(xy[:,1]>=64)&(xy[:,1]+64<=268)
            np.testing.assert_array_equal(valid,expected);seen.extend(map(tuple,xy))
        self.assertEqual(len(seen),ny*nx);self.assertEqual(len(set(seen)),ny*nx)
    def test_synthetic_whole_slide_assembly(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'plan';root.mkdir();(root/'bands').mkdir()
            s={'bands':2,'grid_shape':[4,3],'shape':[32,24],'rows_per_band':2,'sample_id':'synthetic',
               'markers':['CAV','CD8'],'fold':0}
            (root/'slide.json').write_text(json.dumps(s))
            for i in range(2):
                band=root/'bands'/f'band_{i:05d}';band.mkdir()
                valid=np.ones((2,3),bool);valid[0,0]=False
                pred=np.ones((2,3,2),np.float32)*.2;pred[~valid]=np.nan
                np.savez(band/'maps.npz',measured=np.ones((2,3,2),np.float32)*.4,prediction=pred,
                         valid=valid,he=np.full((2,3,3),180,np.uint8))
                meta={'complete':True,'band':i,'row_start':i*2,'row_stop':i*2+2,'slide_plan_sha256':sha(root/'slide.json'),
                      'markers':s['markers'],'checkpoint_sha256':'same','normalization_sha256':'same_norm'}
                (band/'manifest.json').write_text(json.dumps(meta))
            out=Path(d)/'figures';assemble(Namespace(slide_plan=str(root),out=str(out),max_display_side=2500))
            self.assertTrue(json.loads((out/'manifest.json').read_text())['complete'])
            self.assertEqual(np.load(out/'valid.npy').sum(),10)
            self.assertTrue(np.isnan(np.load(out/'prediction.npy')[0,0]).all())
            self.assertTrue((out/'marker_01.png').is_file())
            with self.assertRaises(FileExistsError):assemble(Namespace(slide_plan=str(root),out=str(out),max_display_side=2500))

if __name__=='__main__':unittest.main()
