import json
from pathlib import Path
import tempfile
import unittest
from verify_pair import verify


class PairTests(unittest.TestCase):
    def test_matched_pair_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[Path(tmp)/'base',Path(tmp)/'fds']
            for root,mode in zip(roots,['weighted','weighted_fds']):
                root.mkdir();train=root/'fold_00/training';train.mkdir(parents=True)
                values={'manifest.json':{'complete':True,'mode':mode},
                        'config.json':{'mode':mode,'hidden':256},
                        'folds.json':[{'train':['a'],'val':['b'],'test':['c']}],
                        'inventory.json':[{'sample_id':'c','patient_id':'c','rows':4}],
                        'fold_00/training/expression_state.json':{'thresholds':[3.],'positive_weight':[5.]},
                        'fold_00/training/target_scaler.json':{'mean':[2.],'std':[4.]},
                        'fold_00/training/invocation.json':{'args':{'mode':mode,'out':str(root),'split':str(root/'split.json'),'lr':.001}}}
                for name,value in values.items():(root/name).write_text(json.dumps(value))
            self.assertTrue(verify(*roots)['matched'])
            path=roots[1]/'fold_00/training/expression_state.json'
            path.write_text(json.dumps({'thresholds':[4.],'positive_weight':[5.]}))
            with self.assertRaises(ValueError):verify(*roots)

if __name__=='__main__':unittest.main()
