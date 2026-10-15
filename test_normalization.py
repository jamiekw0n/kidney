import unittest
import numpy as np
from data import fit_scaler,scale
from metrics import metrics

class NormalizationTests(unittest.TestCase):
    def test_fixed_scale_roundtrip_and_old_scaler(self):
        y=np.array([[0.,255.],[127.5,0.]],np.float32);ids=np.array(['a','b'])
        state=fit_scaler(y,ids,intensity_max=255)
        np.testing.assert_allclose(scale(y,state),y/255)
        np.testing.assert_allclose(scale(scale(y,state),state,inverse=True),y)
        old=fit_scaler(y,ids)
        np.testing.assert_allclose(scale(scale(y,old),old,inverse=True),y,atol=1e-5)
        with self.assertRaises(ValueError):scale(np.array([[256.,0.]]),state)
        with self.assertRaises(ValueError):fit_scaler(y,ids,True,255)
        with self.assertRaises(ValueError):fit_scaler(y,ids,intensity_max=1)
    def test_normalized_metrics_preserve_correlations(self):
        y=np.array([[0.],[10.],[100.]]);pred=np.array([[2.],[15.],[70.]])
        ids=np.array(['a']*3)
        raw=metrics(y,pred,ids,['X'])['X']['patients']['a']
        norm=metrics(y/255,pred/255,ids,['X'])['X']['patients']['a']
        for key in ('mae','rmse','mean_bias','prediction_on_measured_intercept'):
            self.assertAlmostEqual(norm[key],raw[key]/255)
        for key in ('pearson','spearman','prediction_on_measured_slope'):
            self.assertAlmostEqual(norm[key],raw[key])
