import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dense_roi import roi_grid,fold_head
from prepare import block_targets

class DenseRoiTests(unittest.TestCase):
    def test_requested_roi_count_and_margin(self):
        xy=roi_grid([3800,27500,6200,31500],[40000,10000])
        self.assertEqual(len(xy),150000)
        np.testing.assert_array_equal(xy[0],[3804,27504])
        np.testing.assert_array_equal(xy[-1],[6196,31496])
        with self.assertRaises(ValueError):roi_grid([0,0,16,16],[100,100])
        with self.assertRaises(ValueError):roi_grid([64,64,81,80],[200,200])
    def test_targets_and_context_alignment(self):
        roi=[100,100,116,124];xy=roi_grid(roi,[300,300])
        pixels=np.arange(144*136).reshape(144,136)
        for k,(x,y) in enumerate(xy):
            patch=pixels[(k//2)*8:(k//2)*8+128,(k%2)*8:(k%2)*8+128]
            self.assertEqual(patch.shape,(128,128))
            np.testing.assert_array_equal(patch[60:68,60:68],pixels[y-100+56:y-100+64,x-100+56:x-100+64])
        image=np.arange(24*16*2).reshape(24,16,2)
        actual,y=block_targets(image,np.ones((24,16),bool),[1,0],100,100)
        np.testing.assert_array_equal(xy,actual)
        np.testing.assert_allclose(y[0],image[:8,:8].mean((0,1))[[1,0]])
    def test_only_outer_fold(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=[{'train':['a'],'val':['b'],'test':['c']}]
            (root/'folds.json').write_text(json.dumps(f))
            self.assertEqual(fold_head(root,'c')[0],0)
            with self.assertRaises(ValueError):fold_head(root,'a')
            f[0]['train'].append('c');(root/'folds.json').write_text(json.dumps(f))
            with self.assertRaises(ValueError):fold_head(root,'c')
