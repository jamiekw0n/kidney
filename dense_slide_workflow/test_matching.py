"""Verify the isolated treatment and the separate loss/bin target scales."""
import unittest
import numpy as np
import torch
from model import Head
from config import Config
from loss import bin_targets,weighted_huber
from data import fit_scaler,scale
from expression import fit_expression,element_weights


class MatchedTests(unittest.TestCase):
    def test_architecture_matches_earlier_expression_mlp(self):
        torch.manual_seed(42)
        ref=torch.nn.Sequential(torch.nn.Linear(1536,256),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(256,42)).eval()
        torch.manual_seed(42)
        head=Head(42,Config(mode='weighted')).eval()
        torch.testing.assert_close(head.hidden[0].weight,ref[0].weight)
        torch.testing.assert_close(head.output.weight,ref[3].weight)
        self.assertEqual(sum(p.numel() for p in head.parameters()),404266)
        x=torch.randn(3,1536)
        torch.testing.assert_close(head(x),ref(x))

    def test_modes_have_identical_initial_weights_and_warmup_predictions(self):
        torch.manual_seed(3);a=Head(2,Config(mode='weighted'))
        torch.manual_seed(3);b=Head(2,Config(mode='weighted_fds'))
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
        x=torch.randn(4,1536);ny=torch.rand(4,2)
        torch.manual_seed(5);p=a(x,ny,0)
        torch.manual_seed(5);q=b(x,ny,0)
        torch.testing.assert_close(p,q)

    def test_bin_targets_are_not_regression_zscores(self):
        raw=np.array([[0.,10.],[10.,20.],[255.,255.]],np.float32)
        patients=np.array(['synthetic_a']*3)
        scaler=fit_scaler(raw,patients)
        z=scale(raw,scaler);ny=bin_targets(raw,255.)
        self.assertLess(float(z.min()),0)
        self.assertGreater(float(z.max()),1.)
        self.assertGreaterEqual(float(ny.min()),0);self.assertLessEqual(float(ny.max()),1)
        self.assertFalse(np.allclose(z,ny))
        head=Head(2,Config())
        # Accidentally passing z scores to FDS should fail, not silently clamp.
        with self.assertRaises(ValueError):head.fds.buckets(torch.from_numpy(z))
        np.testing.assert_allclose(scale(z,scaler,inverse=True),raw,atol=2e-5)
        with self.assertRaises(ValueError):bin_targets(np.array([[-1.]]),255.)
        with self.assertRaises(ValueError):bin_targets(np.array([[256.]]),255.)

    def test_loss_is_exact_previous_weighted_huber(self):
        raw=np.arange(100,dtype=np.float32)[:,None]
        patient=np.array(['synthetic']*100)
        state=fit_expression(raw,patient,['synthetic_marker'])
        z=scale(raw,fit_scaler(raw,patient))
        target=torch.from_numpy(z);pred=torch.zeros_like(target,requires_grad=True)
        weights=torch.from_numpy(element_weights(raw,state))
        got=weighted_huber(pred,target,weights)
        error=np.abs(z)
        expected=np.mean(np.where(error<=1,.5*error**2,error-.5)*weights.numpy())
        self.assertAlmostEqual(got.item(),float(expected),places=6)
        got.backward();self.assertTrue(torch.isfinite(pred.grad).all())
        with self.assertRaises(ValueError):weighted_huber(pred,target,weights[:1])

if __name__=='__main__':unittest.main()
