import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from expression import fit_expression, element_weights, subgroup_metrics, average_precision


class ExpressionTests(unittest.TestCase):
    def test_patient_weighted_quantile(self):
        # The larger patient's row count must not dominate the threshold.
        y=np.array([[0.]]*100+[[10.]]*2)
        state=fit_expression(y,np.array(['a']*100+['b']*2),['m'],quantile=.75)
        self.assertEqual(state['thresholds'],[10.])
        self.assertEqual(state['positive_weight'],[1.])

    def test_balance_and_cap(self):
        y=np.arange(100.)[:,None]; patients=np.array(['a']*100)
        s=fit_expression(y,patients,['m'])
        w=element_weights(y,s)[:,0]; high=y[:,0]>s['thresholds'][0]
        self.assertAlmostEqual(float(w[high].sum()),float(w[~high].sum()),places=4)
        self.assertLessEqual(float(w.max()),5.)
        s=fit_expression(y,patients,['m'],quantile=.99,max_weight=2)
        self.assertLessEqual(float(element_weights(y,s).max()),2.)

    def test_constant_and_invalid(self):
        s=fit_expression(np.zeros((4,1)),np.array(['a']*4),['m'])
        np.testing.assert_array_equal(element_weights(np.zeros((4,1)),s),1.)
        with self.assertRaises(ValueError): fit_expression(np.zeros((4,1)),np.array(['a']*4),['m'],max_weight=0)

    def test_assay_threshold_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'thresholds.json'; path.write_text('{"m": 2}')
            y=np.array([[0.],[1.],[3.],[4.]])
            s=fit_expression(y,np.array(['a']*4),['m'],str(path))
            r=subgroup_metrics(y,np.array([[0.],[3.],[3.],[0.]]),np.array(['a']*4),['m'],s)['m']['a']
            self.assertEqual(r['precision'],.5); self.assertEqual(r['recall'],.5)
            self.assertEqual(r['false_positive_rate'],.5)
            self.assertEqual(r['positive_mae'],2.)
            json.dumps(r,allow_nan=False)

    def test_tied_ap(self):
        self.assertEqual(average_precision(np.array([True,False]),np.array([1.,1.])),.5)
        self.assertIsNone(average_precision(np.array([False,False]),np.array([1.,2.])))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; real trainer smoke requires workstation environment')
    def test_full_synthetic_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache'; cache.mkdir()
            rng=np.random.default_rng(7); ids=['synthetic_'+str(i) for i in range(4)]
            folds=[]
            for i,pid in enumerate(ids):
                folder=cache/pid; folder.mkdir()
                np.savez(folder/'a.npz',xy=np.array([[4,4]]*12),features=rng.normal(size=(12,1536)).astype('float32'),target=np.arange(12,dtype='float32')[:,None])
                meta=dict(complete=True,sample_id=pid,patient_id=pid,shape=[128,128],panel=['synthetic_marker'],shards=['a.npz'],encoder='UNI2-h',revision='SYNTHETIC',checkpoint_sha256='SYNTHETIC',transform='SYNTHETIC',input_patch=128,he_mpp=.5,target_block=8)
                (folder/'manifest.json').write_text(json.dumps(meta))
                val=ids[(i+1)%4]; folds.append(dict(train=[p for p in ids if p not in [pid,val]],val=[val],test=[pid]))
            plan=root/'folds.json'; plan.write_text(json.dumps(folds)); out=root/'run'
            cmd=[sys.executable,str(Path(__file__).with_name('cross_validate.py')),'--cache-root',str(cache),'--reference-folds',str(plan),'--out',str(out),'--folds','4','--device','cpu','--hidden','4','--epochs','2','--patience','1','--bootstrap-replicates','5']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads((out/'oof_expression_summary.json').read_text())
            self.assertEqual(report['metrics']['synthetic_marker']['positive_mae']['total_patients'],4)
            self.assertTrue((out/'primary_summary.json').exists())

if __name__=='__main__': unittest.main()
