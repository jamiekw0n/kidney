"""Tiny real CPU training/CV/inference run; explicit synthetic IDs and encoder provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from config import Config
from model import Head,load_checkpoint
from cross_validate import reviewed_folds
from train import check_arguments
from argparse import Namespace


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.cache=self.root/'cache';self.cache.mkdir()
        self.ids=['synthetic_patient_'+str(i) for i in range(4)]
        rng=np.random.default_rng(123)
        for i,pid in enumerate(self.ids):
            folder=self.cache/pid;folder.mkdir()
            # Distinct distributions for each patient, but several rows per supported bin.
            y=np.tile(np.array([[0.,0.],[10.,40.],[100.,180.],[255.,255.]],np.float32),(4,1))
            x=rng.normal(size=(len(y),1536)).astype('float32')
            np.savez(folder/'shard.npz',features=x,target=y,xy=np.column_stack((np.arange(len(y))*8+4,np.full(len(y),4))))
            m=dict(complete=True,sample_id=pid,patient_id=pid,shape=[256,256],panel=['synthetic_A','synthetic_B'],
                   shards=['shard.npz'],encoder='UNI2-h',revision='SYNTHETIC_NOT_REAL_UNI2',checkpoint_sha256='SYNTHETIC',
                   transform='SYNTHETIC',input_patch=128,target_block=8,he_mpp=.5,selection_policy='synthetic_only')
            (folder/'manifest.json').write_text(json.dumps(m))
        self.folds=[]
        for i,pid in enumerate(self.ids):
            val=self.ids[(i+1)%4]
            self.folds.append(dict(train=[p for p in self.ids if p not in [pid,val]],val=[val],test=[pid]))
        reference=self.root/'reference';reference.mkdir()
        self.plan=reference/'folds.json';self.plan.write_text(json.dumps(self.folds))
        self.source=Path(__file__).parent
        self.env=dict(os.environ)

    def tearDown(self):self.temp.cleanup()

    def run_command(self,command,success=True):
        result=subprocess.run([sys.executable,*command],capture_output=True,text=True,env=self.env)
        if success:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        else:self.assertNotEqual(result.returncode,0)
        return result

    def common(self):
        return ['--device','cpu','--epochs','4','--min-epochs','3','--patience','1','--batch-size','8',
                '--hidden','8','--dropout','0','--bins','5','--kernel-size','3',
                '--fds-start','1','--fds-off','2']

    def test_two_weighted_trainers_and_heldout_inference(self):
        # Both variants retain weighted loss and z scores; only FDS is toggled.
        split=self.root/'split.json';split.write_text(json.dumps(self.folds[0]))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.cache.glob('*/*')}
        for mode in ['weighted','weighted_fds']:
            out=self.root/mode
            self.run_command([str(self.source/'train.py'),'--caches',*[str(r) for r in sorted(self.cache.iterdir())],
                              '--split',str(split),'--out',str(out),'--mode',mode,*self.common()])
            history=json.loads((out/'history.json').read_text())
            self.assertGreaterEqual(len(history),3)
            self.assertFalse(history[0]['checkpoint_eligible'])
            if mode in ['weighted_fds']:
                self.assertTrue(history[1]['fds_applied']);self.assertFalse(history[2]['fds_applied'])
            model,ckpt=load_checkpoint(out/'best.pt')
            self.assertGreaterEqual(ckpt['epoch'],3)
            self.assertNotIn('criterion_state',ckpt)
            self.assertNotEqual(ckpt['scaler'].get('mode'),'fixed_0_1')
            model.eval();a=model(torch.zeros(2,1536));b=model(torch.zeros(2,1536))
            torch.testing.assert_close(a,b)
        test_cache=self.cache/self.folds[0]['test'][0]
        pred=self.root/'predictions'
        self.run_command([str(self.source/'predict.py'),'--checkpoint',str(self.root/'weighted_fds/best.pt'),
                          '--caches',str(test_cache),'--out',str(pred)])
        with np.load(pred/test_cache.name/'shard.npz') as z:self.assertEqual(z['prediction'].shape,(16,2))
        # Explicitly refuse to score a training patient as held-out.
        self.run_command([str(self.source/'predict.py'),'--checkpoint',str(self.root/'weighted_fds/best.pt'),
                          '--caches',str(self.cache/self.folds[0]['train'][0]),'--out',str(self.root/'bad')],False)
        self.assertEqual(hashes,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.cache.glob('*/*')})

    def test_full_cv_exports_metrics_and_predictions(self):
        out=self.root/'cv'
        self.run_command([str(self.source/'cross_validate.py'),'--cache-root',str(self.cache),
                          '--reference-folds',str(self.plan),'--out',str(out),'--bootstrap-replicates','8',*self.common()])
        self.assertTrue(json.loads((out/'manifest.json').read_text())['complete'])
        report=json.loads((out/'primary_summary.json').read_text())
        self.assertEqual(report['role'],'out_of_fold')
        self.assertEqual(len(list(out.glob('fold_*/outer_predictions/*/manifest.json'))),4)
        sub=json.loads((out/'oof_expression_summary.json').read_text())
        self.assertEqual(sub['metrics']['synthetic_A']['negative_mae']['total_patients'],4)
        # New outputs only, never silently overwrite a prior run.
        self.run_command([str(self.source/'cross_validate.py'),'--cache-root',str(self.cache),
                          '--reference-folds',str(self.plan),'--out',str(out),*self.common()],False)

    def test_folds_and_schedule_guards(self):
        invalid=[dict(f) for f in self.folds];invalid[0]={**invalid[0],'val':invalid[0]['train']}
        self.plan.write_text(json.dumps(invalid))
        with self.assertRaises(ValueError):reviewed_folds(self.plan,list(self.cache.iterdir()))
        with self.assertRaises(ValueError):check_arguments(Namespace(epochs=120,min_epochs=10,patience=10,batch_size=8,lr=.001,lr_gamma=.95,seed=42),Config())

if __name__=='__main__':unittest.main()
