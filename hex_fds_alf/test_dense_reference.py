import json
from pathlib import Path
import tempfile
import unittest
from dense_from_review import resolve_reference
from dense_compare import roi_grid, fold_head, load_heldout_models

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

class DenseHeadTests(unittest.TestCase):
    def test_new_format_heads_are_held_out_eval_and_mode_checked(self):
        import torch
        from config import Config
        from model import Head
        from io_utils import SIGNATURE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta=dict(panel=['synthetic_marker'],patient_id='synthetic_heldout',encoder='UNI2-h',revision='synthetic',
                      checkpoint_sha256='synthetic',transform='synthetic',input_patch=128,target_block=8,he_mpp=.5,selection_policy='synthetic')
            paths=[]
            for mode in ['control','fds_alf']:
                config=Config(mode=mode,hidden1=4,hidden2=2)
                head=Head(1,config)
                path=root/(mode+'.pt')
                torch.save(dict(format='kidney_hex_fds_alf_v1',config=config.to_dict(),state_dict=head.state_dict(),
                    panel=meta['panel'],scaler={'mean':[0.],'std':[255.],'log1p':False},
                    cache_signature={k:meta[k] for k in SIGNATURE},
                    split={'train':['synthetic_train'],'val':['synthetic_val'],'test':['synthetic_heldout']}),path)
                paths.append(path)
            models=load_heldout_models(paths,meta,'cpu')
            for model,state in models:
                self.assertFalse(model.training)
                self.assertFalse(model.fds.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertTrue(torch.isfinite(model(torch.zeros(2,1536))).all())
            with self.assertRaises(ValueError):load_heldout_models(paths[::-1],meta,'cpu')
            with self.assertRaises(ValueError):load_heldout_models(paths,{**meta,'patient_id':'synthetic_train'},'cpu')
            with self.assertRaises(ValueError):load_heldout_models(paths,{**meta,'checkpoint_sha256':'wrong'},'cpu')

if __name__=='__main__':unittest.main()
