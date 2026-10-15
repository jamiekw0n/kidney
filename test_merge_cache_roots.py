import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from merge_cache_roots import inventory

class MergeTests(unittest.TestCase):
    def make_sample(self,root,name,revision='sha'):
        sample=root/name;sample.mkdir(parents=True)
        meta=dict(complete=True,sample_id=name,patient_id=name,panel=['DAPI'],shape=[256,256],
            shards=['cache.npz'],encoder='UNI2-h',revision=revision,checkpoint_sha256='hash',transform='resize',
            input_patch=128,he_mpp=.507,target_block=8,selection_policy={'stride':128},he='/raw/'+name+'/he.tiff')
        (sample/'manifest.json').write_text(json.dumps(meta))
        np.savez(sample/'cache.npz',xy=np.array([[68,68]]),target=np.array([[3.]],np.float32),features=np.ones((1,1536),np.float32))
    def test_merge_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_sample(root/'old','A');self.make_sample(root/'new','B')
            self.assertEqual(set(inventory([root/'old',root/'new'])),{'A','B'})
            self.make_sample(root/'new','A')
            with self.assertRaises(ValueError):inventory([root/'old',root/'new'])
    def test_revision_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_sample(root/'old','A');self.make_sample(root/'new','B','other')
            with self.assertRaises(ValueError):inventory([root/'old',root/'new'])

if __name__=='__main__':unittest.main()
