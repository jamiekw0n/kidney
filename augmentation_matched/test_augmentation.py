import json,hashlib,tempfile,unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import numpy as np
import torch
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
torch.set_num_threads(1)
from augment import parameters,geometry,views
from feature_views import FeatureViews
from pipeline import plan,extract,verify_all,train_job,aggregate,config,paired_interval,sha

class AugmentationTests(unittest.TestCase):
    def test_geometry_keeps_center_target_and_pixels(self):
        rgb=np.random.default_rng(5).integers(0,256,(128,128,3),dtype=np.uint8);before=rgb.copy()
        for code in range(8):
            transformed=geometry(rgb,code)
            np.testing.assert_array_equal(transformed[60:68,60:68].sum((0,1),dtype=np.int64),rgb[60:68,60:68].sum((0,1),dtype=np.int64))
        p=parameters('S',68,68);np.testing.assert_array_equal(p,parameters('S',68,68))
        self.assertTrue(1<=p[0]<=7);self.assertTrue(np.all((p[1:4]>=.9)&(p[1:4]<=1.1)))
        a,b=views(rgb,p);self.assertEqual(a.size,(128,128));self.assertEqual(b.size,(128,128));np.testing.assert_array_equal(rgb,before)
    def test_paired_intervals(self):
        self.assertEqual(paired_interval([.1,.1,.1]),(.10000000000000002,.10000000000000002,.10000000000000002))
        self.assertEqual(paired_interval([]),(None,None,None))

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();source=self.root/'original';source.mkdir();self.source=source
        (source/'cache').mkdir();(source/'prepared').mkdir();self.ck=self.root/'SYNTHETIC_weights.bin';self.ck.write_bytes(b'synthetic')
        self.ids=['SYNTHETIC_'+str(i) for i in range(4)];rng=np.random.default_rng(123);tasks=[]
        for index,sample in enumerate(self.ids):
            cache=source/'cache'/sample;cache.mkdir();prep=source/'prepared'/sample;prep.mkdir()
            np.save(prep/'he.npy',rng.integers(40,220,(160,160,3),dtype=np.uint8))
            xy=np.array([[68+8*(j%4),68+8*(j//4)] for j in range(8)],np.int64);shards=[]
            for i in range(2):
                name=f'shard_{i}.npz';shards.append(name);sl=slice(i*4,i*4+4)
                he_pixels=np.load(prep/'he.npy');identity=[]
                for x,y in xy[sl]:identity.append(np.tile((he_pixels[y-64:y+64,x-64:x+64].astype(np.float32)/255).mean((0,1)),512))
                np.savez(cache/name,features=np.asarray(identity,dtype=np.float32),xy=xy[sl],
                    target=np.column_stack((np.linspace(0,1,8),np.linspace(1,0,8)))[sl].astype(np.float32))
            meta={'complete':True,'sample_id':sample,'patient_id':sample,'shape':[160,160],'panel':['A','B'],'shards':shards,
                'encoder':'UNI2-h','revision':'SYNTHETIC','checkpoint_sha256':sha(self.ck),
                'transform':'PIL bilinear square resize 224; ToTensor; ImageNet normalization','input_patch':128,'target_block':8,'he_mpp':.5,
                'target_units':'slide_channel_percentile_pixel_clipped_0_1','normalization_recipe':{'method':'synthetic'},'selection_policy':{'stride':32}}
            stat=(prep/'he.npy').stat();meta['he_array_stat']={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            (cache/'manifest.json').write_text(json.dumps(meta));tasks.append({'sample_id':sample,'patient_id':sample})
        folds=[{'train':[p for p in self.ids if p not in (test,self.ids[(i+1)%4])],'val':[self.ids[(i+1)%4]],'test':[test]} for i,test in enumerate(self.ids)]
        (source/'folds.json').write_text(json.dumps(folds));(source/'experiment.json').write_text(json.dumps({'tasks':tasks}))
        self.plan=self.root/'augmentation';plan(Namespace(source_run=str(source),checkpoint=str(self.ck),out=str(self.plan),epochs=51,min_epochs=51,patience=1))
        self.before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    def tearDown(self):self.temp.cleanup()
    def generate(self,ids):
        class Encoder:
            def __call__(self,b):return b.mean((2,3)).repeat(1,512)
        def transform(p):return torch.from_numpy(np.asarray(p).copy().transpose(2,0,1)).float()/255
        with patch('pipeline.load_encoder',return_value=(Encoder(),transform)):extract(self.plan,ids,'cpu',16,False)

class CacheTests(Fixture):
    def test_view_alignment_resume_and_original_preservation(self):
        self.generate([0]);task=config(self.plan)['tasks'][0]
        base=self.plan/'views'/task['sample_id'];manifest=json.loads((base/'manifest.json').read_text())
        hashes={str(p):sha(p) for p in base.iterdir() if p.is_file()}
        extract(self.plan,[0],'cpu',16,True)
        self.assertEqual(hashes,{str(p):sha(p) for p in base.iterdir() if p.is_file()})
        store=FeatureViews(self.plan,[Path(task['cache'])]);indices=np.array([7,0,4,3]);batch=np.zeros((4,1536),np.float32)
        chosen=np.array([True,False,True,False]);out=store.replace(batch,indices,chosen,'geometry')
        second=np.load(base/manifest['shards'][1]['geometry']);np.testing.assert_array_equal(out[0],second[3]);np.testing.assert_array_equal(out[2],second[0])
        self.assertTrue((out[1]==0).all());self.assertTrue((out[3]==0).all())
        self.assertEqual(self.before,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})
    def test_corrupt_view_is_rejected_without_touching_original(self):
        from pipeline import verify_views
        self.generate([0]);task=config(self.plan)['tasks'][0];base=self.plan/'views'/task['sample_id']
        manifest=json.loads((base/'manifest.json').read_text())
        with (base/manifest['shards'][0]['geometry']).open('ab') as f:f.write(b'SYNTHETIC corruption')
        with self.assertRaisesRegex(ValueError,'checksum'):verify_views(self.plan,task)
        self.assertEqual(self.before,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_partial_cache_reuses_committed_shard(self):
        self.generate([0]);base=self.plan/'views'/self.ids[0]
        # Simulate absence of final completion marker in a separate new partial artifact.
        # This is synthetic test metadata only; feature arrays are never removed.
        saved=json.loads((base/'manifest.json').read_text());(base/'manifest.json').rename(base/'SYNTHETIC_old_completion.json')
        with patch('pipeline.load_encoder',return_value=(None,None)):extract(self.plan,[0],'cpu',16,True)
        self.assertEqual(saved['shards'],json.loads((base/'manifest.json').read_text())['shards'])

class EndToEndTests(Fixture):
    def test_three_variants_train_and_export_one_fold(self):
        self.generate(range(4));verify_all(self.plan)
        for variant in ('control','geometry','geometry_color'):train_job(self.plan,variant,0,'cpu')
        records=[]
        for variant in ('control','geometry','geometry_color'):
            root=self.plan/'runs'/variant/'fold_00';self.assertTrue(json.loads((root/'manifest.json').read_text())['complete'])
            history=json.loads((root/'training/history.json').read_text());self.assertEqual(len(history),51)
            self.assertEqual(history[0]['augmented_draws']==0,variant=='control')
            records.append([json.loads((root/'training'/name).read_text()) for name in ('initial_state.json','target_scaler.json','expression_state.json')])
            self.assertTrue((root/'outer_predictions'/self.ids[0]/'shard_0.npz').is_file())
        self.assertEqual(records[0],records[1]);self.assertEqual(records[1],records[2])
        digests=[json.loads((self.plan/'runs'/v/'fold_00/training/history.json').read_text())[0]['sampler_draw_sha256'] for v in ('control','geometry','geometry_color')]
        self.assertEqual(len(set(digests)),1)
        self.assertEqual(self.before,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})


class AggregateTests(Fixture):
    def test_aggregate_paired_metrics_and_audit(self):
        from metrics import metrics
        from expression import subgroup_metrics,fit_expression
        c=config(self.plan);panel=['A','B']
        for variant,offset in [('control',.1),('geometry',.05),('geometry_color',.02)]:
            for i,split in enumerate(c['folds']):
                root=self.plan/'runs'/variant/f'fold_{i:02d}';(root/'training').mkdir(parents=True)
                pid=split['test'][0];y=np.array([[0.,.1],[.2,.4],[.5,.8],[.9,1.]],np.float32);ids=np.array([pid]*len(y))
                report={pid:metrics(y,y+offset,ids,panel)}
                state=fit_expression(y,ids,panel)
                (root/'patient_metrics.json').write_text(json.dumps(report))
                (root/'expression_patients.json').write_text(json.dumps({pid:subgroup_metrics(y,y+offset,ids,panel,state)}))
                (root/'manifest.json').write_text(json.dumps({'complete':True,'patients':[pid],'fold':i,'variant':variant,'plan_fingerprint':c['fingerprint']}))
                for name in ('initial_state.json','target_scaler.json','expression_state.json','config.json'):
                    (root/'training'/name).write_text(json.dumps({'SYNTHETIC':'identical across variants'}))
                (root/'training/history.json').write_text(json.dumps([{'sampler_draw_sha256':'SYNTHETIC_MATCHED','steps':1,'augmented_draws':0 if variant=='control' else 2,'view_draw_sha256':'SYNTHETIC_MATCHED_VIEWS'}]))
        aggregate(self.plan)
        import csv
        with (self.plan/'comparison/paired_differences.csv').open() as f:rows=list(csv.DictReader(f))
        r=next(r for r in rows if r['baseline']=='control' and r['candidate']=='geometry' and r['marker']=='A' and r['metric']=='mae')
        self.assertAlmostEqual(float(r['patient_mean_delta']),-.05,places=6)
        self.assertEqual(r['defined_pairs'],'4')

if __name__=='__main__':unittest.main()
