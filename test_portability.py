import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from image_io import read_he_patch
from validate import validate


class Reader:
    height=10
    width=12
    def __init__(self):
        self.image=np.arange(10*12*3,dtype=np.uint8).reshape(10,12,3)
    def read(self,y0,y1,x0,x1):
        return self.image[y0:y1,x0:x1]


class PortabilityTests(unittest.TestCase):
    def test_crop_padding_and_spatial_center(self):
        reader=Reader()
        crop=read_he_patch(reader,1,2,size=8)
        self.assertEqual(crop.shape,(8,8,3))
        np.testing.assert_array_equal(crop[4,4],reader.image[2,1])
        self.assertTrue((crop[:2]==0).all())
        self.assertTrue((crop[:,:3]==0).all())
        with self.assertRaises(ValueError):
            read_he_patch(reader,12,2)

    def test_standalone_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            meta=dict(complete=True,patient_id='patient',sample_id='sample',panel=['X'],
                      shape=[20,20],target_block=8,shards=['targets.npz'])
            (root/'manifest.json').write_text(json.dumps(meta))
            np.savez(root/'targets.npz',xy=np.array([[4,4]]),measured=np.array([[2.]]))
            self.assertEqual(validate(root)['rows'],1)
            np.savez(root/'targets.npz',xy=np.array([[40,4]]),measured=np.array([[2.]]))
            with self.assertRaises(ValueError):
                validate(root)

    def test_target_block_support_and_target_free_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta = dict(complete=True, patient_id='patient', sample_id='sample', panel=['X'],
                        shape=[20,20], target_block=8, encoder='UNI2-h', shards=['cache.npz'])
            (root/'manifest.json').write_text(json.dumps(meta))
            features = np.zeros((1,1536), dtype=np.float32)
            np.savez(root/'cache.npz', xy=np.array([[4,4]]), features=features)
            self.assertEqual(validate(root, require_targets=False)['rows'], 1)
            with self.assertRaises(KeyError):
                validate(root)
            np.savez(root/'cache.npz', xy=np.array([[1,4]]), features=features)
            with self.assertRaisesRegex(ValueError, 'Target block'):
                validate(root, require_targets=False)

    def test_no_external_source_imports(self):
        import ast
        for path in Path(__file__).parent.glob('*.py'):
            if path.name.startswith('test_'):
                continue
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertNotEqual(node.module,'preprocess')
            self.assertNotIn('sys.path.insert',path.read_text())

if __name__=='__main__':
    unittest.main()
