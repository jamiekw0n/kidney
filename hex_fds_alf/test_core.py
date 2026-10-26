"""Independent numerical, gradient, schedule and leakage checks for FDS/ALF."""
import math
import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d
from config import Config
from fds import FeatureSmoothing
from model import Head
from adaptive_loss import AdaptiveLoss,rho
from alf_math import rho_numpy,log_partition_numpy


class AdaptiveTests(unittest.TestCase):
    def test_partition_against_independent_integration(self):
        for alpha in [.001,.003,.01,.05,.1,.5,1.,1.5,1.99,1.999]:
            exact=np.log(2*quad(lambda r:np.exp(-rho_numpy(r,alpha)),0,np.inf,
                                  epsabs=1e-10,epsrel=1e-10,limit=300)[0])
            self.assertAlmostEqual(log_partition_numpy(alpha,512),exact,delta=3e-7)
        # Analytic endpoints, approached from the interior.
        self.assertAlmostEqual(log_partition_numpy(1e-8,512),math.log(math.pi*math.sqrt(2)),delta=1e-7)
        self.assertAlmostEqual(log_partition_numpy(2-1e-8,512),.5*math.log(2*math.pi),delta=3e-7)

    def test_rho_closed_form_alpha_one(self):
        x=torch.tensor([[-2.,0.,3.]],dtype=torch.float64)
        got=rho(x,torch.ones_like(x),torch.full_like(x,2.))
        torch.testing.assert_close(got,torch.sqrt(1+(x/2).square())-1)

    def test_nll_includes_scale_and_partition(self):
        c=Config();loss=AdaptiveLoss(2,c)
        self.assertTrue(torch.allclose(loss.alpha(),torch.ones(2)))
        self.assertTrue(torch.allclose(loss.scale(),torch.ones(2)))
        x=torch.zeros(3,2)
        want=log_partition_numpy(1.,512)
        np.testing.assert_allclose(loss(x).detach().numpy(),want,rtol=1e-7)
        with torch.no_grad():loss.raw_scale.fill_(2.)
        got=loss(x)
        torch.testing.assert_close(got[0],loss.scale().double().log()+loss.log_partition(loss.alpha()))

    def test_autograd_and_large_residuals(self):
        loss=AdaptiveLoss(2,Config())
        residual=torch.tensor([[.1,-.3],[3.,100000.]],requires_grad=True)
        value=loss(residual).mean();value.backward()
        for grad in [loss.raw_alpha.grad,loss.raw_scale.grad,residual.grad]:
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(loss(torch.tensor([[0.,1e12]]))).all())

    def test_log_partition_gradient_independent_quad(self):
        loss=AdaptiveLoss(1,Config())
        for a in [.003,.1,1.,1.9]:
            alpha=torch.tensor([a],dtype=torch.float64,requires_grad=True)
            actual=torch.autograd.grad(loss.log_partition(alpha).sum(),alpha)[0].item()
            def ref(v):
                return np.log(2*quad(lambda r:np.exp(-rho_numpy(r,v)),0,np.inf,epsabs=1e-10,limit=300)[0])
            eps=1e-5;expected=(ref(a+eps)-ref(a-eps))/(2*eps)
            self.assertAlmostEqual(actual,expected,delta=2e-4)

    def test_scale_parameterization_matches_reference_util(self):
        loss=AdaptiveLoss(1,Config())
        for value in [-10.,-1.,0.,2.,10.]:
            with torch.no_grad():loss.raw_scale.fill_(value)
            expected=1e-5+(1-1e-5)*np.logaddexp(0,value+np.log(np.expm1(1.)))
            self.assertAlmostEqual(loss.scale().item(),expected,delta=2e-6)

    def test_scale_gradient_finite_difference(self):
        loss=AdaptiveLoss(1,Config());residual=torch.tensor([[.2],[2.]])
        got=torch.autograd.grad(loss(residual).mean(),loss.raw_scale)[0].item()
        with torch.no_grad():
            loss.raw_scale.fill_(1e-3);plus=loss(residual).mean().item()
            loss.raw_scale.fill_(-1e-3);minus=loss(residual).mean().item()
        self.assertAlmostEqual(got,(plus-minus)/.002,delta=1e-4)


class FDSTests(unittest.TestCase):
    def setUp(self):
        self.c=Config(bins=5,kernel_size=3,sigma=1.,hidden2=2,fds_start=1,fds_off=3)

    def test_kernel_matches_hex_scipy_delta(self):
        for c in [self.c,Config()]:
            f=FeatureSmoothing(1,2,c)
            delta=np.zeros(c.kernel_size);delta[c.kernel_size//2]=1
            want=gaussian_filter1d(delta,c.sigma);want/=want.sum()
            np.testing.assert_allclose(f.kernel.numpy(),want,rtol=1e-7)

    def test_bins_and_running_moments(self):
        f=FeatureSmoothing(1,2,self.c)
        self.assertEqual(f.buckets(torch.tensor([[0.],[.2],[1.]])).flatten().tolist(),[0,1,4])
        h=torch.tensor([[1.,2.],[3.,6.],[5.,8.],[7.,12.]])
        y=torch.tensor([[.01],[.01],[.99],[.99]])
        moment=f.new_moments();f.accumulate(moment,h,y);f.finish_epoch(moment,0)
        torch.testing.assert_close(f.running_mean[0,0],torch.tensor([2.,4.]))
        torch.testing.assert_close(f.running_var[0,0],torch.tensor([2.,8.]))
        # Unsupported neighbors excluded: isolated first bin is not shrunk toward zero.
        torch.testing.assert_close(f.smooth_mean[0,0],f.running_mean[0,0])
        fds_before=f.running_mean.clone()
        moment=f.new_moments();f.accumulate(moment,h+10,y);f.finish_epoch(moment,1)
        torch.testing.assert_close(f.running_mean[0,0],fds_before[0,0]+1.)
        with self.assertRaises(ValueError):f.finish_epoch(moment,1)

    def test_first_late_observation_not_shrunk(self):
        f=FeatureSmoothing(1,2,self.c)
        f.finish_epoch(f.new_moments(),0)
        moment=f.new_moments();f.accumulate(moment,torch.tensor([[4.,8.],[6.,10.]]),torch.tensor([[.5],[.5]]))
        f.finish_epoch(moment,1)
        torch.testing.assert_close(f.running_mean[0,2],torch.tensor([5.,9.]))

    def test_calibration_formula_and_zero_variance(self):
        f=FeatureSmoothing(1,2,self.c)
        f.running_mean.fill_(2);f.running_var[...,:]=torch.tensor([4.,0.])
        f.smooth_mean.fill_(3);f.smooth_var.fill_(16);f.seen.fill_(5);f.last_epoch.fill_(0)
        x=torch.tensor([[4.,99.]],requires_grad=True);y=torch.tensor([[.1]])
        got=f.calibrate(x,y,0,1)
        torch.testing.assert_close(got,torch.tensor([[7.,99.]]))
        got.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([[2.,1.]]))
        torch.testing.assert_close(x,torch.tensor([[4.,99.]]))
        self.assertTrue(torch.equal(f.calibrate(x,y,0,3),x))

    def test_marker_specific_adjustment(self):
        m=Head(2,self.c);m.train()
        with torch.no_grad():
            m.fds.running_mean.zero_();m.fds.running_var.fill_(1);m.fds.smooth_var.fill_(1)
            m.fds.smooth_mean[0].fill_(1);m.fds.smooth_mean[1].fill_(5)
            m.fds.seen.fill_(2);m.fds.last_epoch.fill_(0)
            m.output.weight.fill_(1);m.output.bias.zero_()
        pred,h=m(torch.zeros(3,1536),torch.tensor([[.1,.5]]*3),1,True)
        torch.testing.assert_close(pred[:,1]-pred[:,0],torch.full((3,),8.))
        pred.sum().backward();self.assertIsNotNone(m.output.weight.grad)

    def test_eval_bypasses_labels_and_state(self):
        m=Head(2,self.c).eval();x=torch.randn(5,1536)
        before={k:v.clone() for k,v in m.state_dict().items()}
        a=m(x,epoch=1);b=m(x,torch.full((5,2),float('nan')),epoch=1)
        torch.testing.assert_close(a,b)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        with self.assertRaises(RuntimeError):m.fds.finish_epoch(m.fds.new_moments(),0)
        with self.assertRaises(RuntimeError):m.fds.accumulate(m.fds.new_moments(),torch.ones(5,2),torch.ones(5,2))

    def test_empty_and_singleton_bins_safe(self):
        f=FeatureSmoothing(1,2,self.c)
        moment=f.new_moments();f.accumulate(moment,torch.tensor([[5.,9.]]),torch.tensor([[.5]]))
        f.finish_epoch(moment,0)
        self.assertEqual(int(f.seen.sum()),0)
        x=torch.tensor([[8.,9.]])
        torch.testing.assert_close(f.calibrate(x,torch.tensor([[.5]]),0,1),x)

if __name__=='__main__':unittest.main()
