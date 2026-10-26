import json
from pathlib import Path
import tempfile
import unittest
from dense_from_review import resolve_reference
from dense_compare import roi_grid, fold_head

class DenseReferenceTests(unittest.TestCase):
    def test_review_uses_exact_original_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dense=root/'dense';review=root/'review';dense.mkdir();review.mkdir()
            roi=[3800,27500,6200,31500]
            (dense/'manifest.json').write_text(json.dumps(dict(complete=True,sample_id='F03600',stride=8,target_block=8,roi=roi)))
            (dense/'invocation.json').write_text(json.dumps({'args':{'roi':roi}}))
            (review/'manifest.json').write_text(json.dumps({'complete':True,'source_dense_root':str(dense)}))
            resolved,meta,args=resolve_reference(review)
            self.assertEqual(resolved,dense.resolve());self.assertEqual(meta['roi'],roi)
            xy=roi_grid(roi,[40000,10000])
            self.assertEqual(xy.shape,(150000,2));self.assertEqual(xy[0].tolist(),[3804,27504])

    def test_reject_cycle_and_wrong_patient_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'complete':True,'source_dense_root':str(root)}))
            with self.assertRaises(ValueError):resolve_reference(root)
            (root/'folds.json').write_text(json.dumps([{'train':['p'],'val':[],'test':['p']}]))
            with self.assertRaises(ValueError):fold_head(root,'p')

if __name__=='__main__':unittest.main()
