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


if __name__=='__main__': unittest.main()
