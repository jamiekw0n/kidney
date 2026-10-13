import unittest
import numpy as np
from cohort_prepare import eligible,reservoir_add
from cross_validate import make_folds

class CohortTests(unittest.TestCase):
    def test_tissue_and_center_checks(self):
        rgb=np.full((128,128,3),255,np.uint8); valid=np.ones((128,128),bool)
        args=(0.5,0.5,235,20,15)
        self.assertEqual(eligible(rgb,valid,*args)[0],'low_context_tissue')
        rgb[:]=[180,80,150]
        self.assertEqual(eligible(rgb,valid,*args)[0],'eligible')
        rgb[60:68,60:68]=255
        self.assertEqual(eligible(rgb,valid,*args)[0],'low_center_tissue')
        valid[0,0]=False
        self.assertEqual(eligible(rgb,valid,*args)[0],'invalid_support')
        rgb[:]=0;valid[:]=True
        self.assertEqual(eligible(rgb,valid,*args)[0],'low_context_tissue')
    def test_sampling_is_reproducible_and_bounded(self):
        samples=[]
        for repeat in range(2):
            selected=[];rng=np.random.default_rng(42)
            for i in range(100): reservoir_add(selected,i,i+1,10,rng)
            samples.append(selected)
        self.assertEqual(samples[0],samples[1]);self.assertEqual(len(samples[0]),10)
        self.assertEqual(len(set(samples[0])),10)
    def test_folds_no_leakage_and_each_patient_test_once(self):
        patients=[f'p{i}' for i in range(37)]
        folds=make_folds(patients,5,42,0.2)
        self.assertEqual(folds,make_folds(patients,5,42,0.2))
        tested=[]
        for split in folds:
            self.assertFalse(set(split['train'])&set(split['val']))
            self.assertFalse(set(split['test'])&(set(split['train'])|set(split['val'])))
            self.assertEqual(set().union(*map(set,split.values())),set(patients))
            tested.extend(split['test'])
        self.assertEqual(sorted(tested),sorted(patients))
    def test_small_cohort_rejected(self):
        with self.assertRaises(ValueError): make_folds(['a','b','c'],3,42,0.2)

if __name__=='__main__': unittest.main()
