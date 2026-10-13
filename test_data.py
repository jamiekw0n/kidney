import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from data import partitions, fit_scaler, scale

class DataTests(unittest.TestCase):
    def test_patient_balanced_scaling_and_roundtrip(self):
        y = np.array([[0.],[0.],[10.]], dtype=np.float32)
        patients = np.array(['a','a','b'])
        state = fit_scaler(y,patients)
        self.assertEqual(state['mean'],[5.])
        np.testing.assert_allclose(scale(scale(y,state),state,inverse=True),y)
        state = fit_scaler(y,patients,True)
        np.testing.assert_allclose(scale(scale(y,state),state,inverse=True),y,atol=1e-5)

    def test_negative_log_rejected(self):
        with self.assertRaises(ValueError):
            fit_scaler(np.array([[-1.]]),np.array(['a']),True)

    def test_leakage_rejected(self):
        with self.assertRaises(ValueError):
            partitions([],{'train':['a'],'val':['a'],'test':[]})

    def test_unknown_patient_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            roots=[]
            for patient in ('a','b'):
                root=Path(d)/patient; root.mkdir(); roots.append(root)
                meta=dict(complete=True,patient_id=patient,sample_id=patient,panel=['X'],
                          encoder='UNI2-h',revision='sha',checkpoint_sha256='hash',transform='resize',
                          input_patch=128,he_mpp=0.5,target_block=8)
                (root/'manifest.json').write_text(json.dumps(meta))
            split={'train':['a'],'val':['b'],'test':[]}
            self.assertEqual(len(partitions(roots,split)['train']),1)
            meta['he_mpp']=0.25
            (roots[1]/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                partitions(roots,split)
            meta['patient_id']='unknown'
            (roots[1]/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                partitions(roots,split)

if __name__=='__main__':
    unittest.main()
