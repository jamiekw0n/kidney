import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from evaluate import bootstrap,roc_diagnostics
from plan import choose


class EvaluationTests(unittest.TestCase):
    def test_roc_ties_and_missing_class(self):
        truth=np.array([False,False,True,True])
        auc,recall=roc_diagnostics(truth,np.array([0.,1.,2.,3.]))
        self.assertEqual(auc,1);self.assertEqual(recall,1)
        auc,recall=roc_diagnostics(truth,np.ones(4));self.assertEqual(auc,.5);self.assertEqual(recall,0)
        self.assertEqual(roc_diagnostics(np.zeros(4,dtype=bool),np.ones(4)),(None,None))

    def test_paired_bootstrap_missing_and_fixed_difference(self):
        draws=np.random.default_rng(42).integers(0,4,size=(200,4))
        got=bootstrap([-2.,-2.,np.nan,-2.],draws)
        self.assertEqual(got['patient_mean'],-2)
        self.assertEqual(got['ci95_low'],-2);self.assertEqual(got['ci95_high'],-2)
        self.assertEqual(got['defined_patients'],3)
        self.assertIsNone(bootstrap([np.nan]*4,draws)['patient_mean'])

    def test_ambiguous_auto_selection_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['a','b']:
                r=root/name;r.mkdir();(r/'inventory.json').write_text('[]');(r/'folds.json').write_text('[]')
                (r/'primary_summary.json').write_text('{}')
                d=r/'fold_00/outer_predictions/s';d.mkdir(parents=True);(d/'manifest.json').write_text('{}')
                (r/'manifest.json').write_text('{"complete":true,"mode":"weighted"}')
            with self.assertRaises(ValueError):choose(root,mode='weighted')

    def test_full_evaluation_and_mismatched_ground_truth_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['synthetic_a','synthetic_b','synthetic_c','synthetic_d']
            folds=[{'train':[ids[0]],'val':[ids[1]],'test':ids[2:]},
                   {'train':[ids[2]],'val':[ids[3]],'test':ids[:2]}]
            paths={label:root/label for label in ['baseline','candidate']}
            for label,r in paths.items():
                r.mkdir();(r/'folds.json').write_text(json.dumps(folds))
                (r/'inventory.json').write_text(json.dumps([{'sample_id':pid,'patient_id':pid,'rows':4} for pid in ids]))
                (r/'config.json').write_text(json.dumps({'mode':label,'hidden':256}))
                for i,f in enumerate(folds):
                    fold=r/f'fold_{i:02d}';t=fold/'training';t.mkdir(parents=True)
                    (fold/'split.json').write_text(json.dumps(f))
                    (t/'invocation.json').write_text(json.dumps({'args':{'mode':label,'out':str(t),'split':str(fold/'split.json'),'seed':42+i}}))
                    (t/'target_scaler.json').write_text('{"mean":[0,0],"std":[1,1]}')
                    (t/'config.json').write_text(json.dumps({'mode':label,'hidden':256}))
                    (t/'expression_state.json').write_text('{"panel":["CAV","synthetic"],"thresholds":[2,2],"method":"synthetic_training_only"}')
                    for pid in f['test']:
                        d=fold/'outer_predictions'/pid;d.mkdir(parents=True)
                        y=np.array([[0.,0.],[1.,1.],[3.,3.],[4.,4.]])
                        pred=y+1 if label=='baseline' else y.copy()
                        xy=np.array([[4,4],[12,4],[20,4],[28,4]])
                        if label=='baseline':y=y[::-1];pred=pred[::-1];xy=xy[::-1]
                        np.savez(d/'p.npz',xy=xy,target=y,prediction=pred)
                        m={'complete':True,'sample_id':pid,'patient_id':pid,'outer_fold':i,'panel':['CAV','synthetic'],
                           'shards':['p.npz'],'shape':[64,64],'encoder':'UNI2-h','revision':'synthetic','checkpoint_sha256':'synthetic',
                           'transform':'synthetic','he_mpp':.5,'input_patch':128,'target_block':8,'selection_policy':'synthetic'}
                        (d/'manifest.json').write_text(json.dumps(m))
            plan={'runs':{k:str(v) for k,v in paths.items()},'threshold_reference':'baseline',
                  'pairs':[{'baseline':'baseline','candidate':'candidate','matched_settings':True,'interpretation':'synthetic matched pair'}]}
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan));out=root/'report'
            cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--runs-json',str(plan_path),'--out',str(out),'--bootstrap-replicates','20']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            s=json.loads((out/'summary.json').read_text())
            self.assertEqual(s['baseline']['CAV']['mae']['patient_mean'],1)
            self.assertEqual(s['candidate']['CAV']['mae']['patient_mean'],0)
            paired=json.loads((out/'paired_differences.json').read_text())
            delta=next(v for v in paired if v['marker']=='CAV' and v['metric']=='mae')
            self.assertEqual(delta['patient_mean'],-1);self.assertEqual(delta['patient_wins'],4)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['locations'],16)
            # Refuse changed measured labels, even though patient/fold metadata matches.
            shard=paths['candidate']/'fold_00/outer_predictions'/ids[2]/'p.npz'
            with np.load(shard) as z:xy=z['xy'].copy();y=z['target'].copy();p=z['prediction'].copy()
            y[0,0]=99;np.savez(shard,xy=xy,target=y,prediction=p)
            result=subprocess.run([*cmd[:cmd.index('--out')+1],str(root/'bad'),*cmd[cmd.index('--out')+2:]],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Unmatched targets',result.stderr)

if __name__=='__main__':unittest.main()
