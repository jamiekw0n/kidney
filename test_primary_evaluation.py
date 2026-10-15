import json
import unittest
import numpy as np
from metrics import metrics,average_ranks
from primary_evaluation import summarize

class PrimaryTests(unittest.TestCase):
    def test_rank_ties_and_monotonic(self):
        np.testing.assert_array_equal(average_ranks([3,1,1,2]),[4,1.5,1.5,3])
        report=metrics(np.array([[1.],[2.],[3.]]),np.array([[1.],[4.],[9.]]),np.array(['a']*3),['X'])
        self.assertAlmostEqual(report['X']['patient_mean_spearman'],1)
    def test_calibration_and_signed_bias(self):
        report=metrics(np.array([[1.],[2.],[3.]]),np.array([[5.],[7.],[9.]]),np.array(['a']*3),['X'])
        item=report['X']['patients']['a']
        self.assertAlmostEqual(item['prediction_on_measured_slope'],2)
        self.assertAlmostEqual(item['prediction_on_measured_intercept'],3)
        self.assertAlmostEqual(item['mean_bias'],5)
    def test_equal_patient_weight_and_reproducible_ci(self):
        y=np.zeros((101,1));pred=np.r_[np.zeros(100),10][:,None];ids=np.array(['a']*100+['b'])
        report=metrics(y,pred,ids,['X'])
        a=summarize(report,200,42);self.assertEqual(a,summarize(report,200,42))
        self.assertEqual(a['X']['mae']['patient_mean'],5)
        self.assertIsNone(a['X']['pearson']['ci95_low']);json.dumps(a,allow_nan=False)
    def test_single_patient_no_ci(self):
        report=metrics(np.array([[1.],[2.]]),np.array([[1.],[2.]]),np.array(['a','a']),['X'])
        self.assertIsNone(summarize(report)['X']['mae']['ci95_low'])

if __name__=='__main__':unittest.main()
