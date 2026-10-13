import unittest
import numpy as np
from evaluate_head import sparse_grid,check_role

class EvaluateTests(unittest.TestCase):
    def test_unsampled_stays_masked_and_values_placed(self):
        grid=sparse_grid(np.array([[68,68],[196,196]]),np.array([2.,5.]),[384,384],128)
        self.assertEqual(grid[0,0],2);self.assertEqual(grid[1,1],5)
        self.assertTrue(grid.mask[2,2]);self.assertEqual(grid.count(),2)
    def test_role_leakage_and_empty_test_rejected(self):
        split={'train':['a'],'val':['b'],'test':[]}
        check_role('b',split,'val')
        with self.assertRaises(ValueError):check_role('b',split,'test')
        with self.assertRaises(ValueError):check_role('a',split,'val')
    def test_duplicate_display_cell_rejected(self):
        with self.assertRaises(ValueError):sparse_grid(np.array([[68,68],[69,69]]),[1,2],[384,384],128)

if __name__=='__main__':unittest.main()
