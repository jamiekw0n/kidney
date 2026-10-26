import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from render import render,clipping

class RenderTest(unittest.TestCase):
    def test_cached_render_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';root.mkdir();out=Path(d)/'new'
            meta={'complete':True,'stride':8,'target_block':8,'panel':['CAV','CD8'],
                  'sample_id':'synthetic','roi':[0,0,16,16],'grid_shape':[2,2]}
            (root/'manifest.json').write_text(json.dumps(meta))
            (root/'invocation.json').write_text(json.dumps({'args':{'intensity_max':255}}))
            Image.fromarray(np.full((16,16,3),200,dtype=np.uint8)).save(root/'he_roi.png')
            pixels=np.full((16,16),64,dtype=np.uint8)
            np.savez(root/'measured_mif_native_roi.npz',CAV=pixels)
            xy=np.array([[4,4],[12,4],[4,12],[12,12]])
            target=np.column_stack([np.full(4,64),np.zeros(4)])
            pred=target.copy();pred[0,0]=-1;pred[1,0]=300
            np.savez(root/'dense_predictions.npz',xy=xy,target=target,linear=pred,mlp=target)
            digest=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            before=digest()
            render(root,out,['CAV','CD8'],255,0,1,35)
            self.assertEqual(before,digest())
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual((m['vmin'],m['vmax']),(0,1))
            self.assertTrue((out/'marker_00.png').is_file())
            self.assertTrue((out/'marker_01.png').is_file())
            self.assertIn('unavailable',(out/'index.html').read_text())
            stats=json.loads((out/'display_statistics.json').read_text())
            s=next(s for s in stats if s['marker']=='CAV' and s['map'].startswith('linear'))
            self.assertEqual(s['below_vmin_fraction'],.25)
            self.assertEqual(s['above_vmax_fraction'],.25)
            with self.assertRaises(FileExistsError):render(root,out,['CAV'],255,0,1,35)

if __name__=='__main__':unittest.main()
