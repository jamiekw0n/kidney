"""Synthetic reader integration: new grid, normalized targets, unchanged raw inputs, copied cache provenance."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from workflow import prepare_task,extract_task
from normalization import UNITS,recipe
from raw_safety import source_stats
from encoder_runtime import sha

class PreparationTest(unittest.TestCase):
    def test_new_grid_targets_and_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();raw=root/'raw';raw.mkdir();hefile=raw/'s_HandE.tiff';miffile=raw/'s.qptiff'
            hefile.write_bytes(b'synthetic HE');miffile.write_bytes(b'synthetic mIF')
            paths={'he':str(hefile),'mif':str(miffile)};source=root/'normalized';source.mkdir()
            report={'recipe':recipe(),'markers':[{'marker':'A','effective_divisor':64},{'marker':'B','effective_divisor':128}],
                    'source_axes':'CYX','series':0,'level':0}
            (source/'normalization.json').write_text(json.dumps(report))
            meta={**paths,'complete':True,'sample_id':'s','patient_id':'p','panel':['A','B'],'shape':[384,384],
                  'he_mpp':.5,'input_patch':128,'target_block':8,'encoder':'UNI2-h','revision':'synthetic',
                  'checkpoint_sha256':'synthetic','transform':'synthetic','shards':['old.npz'],
                  'target_units':UNITS,'normalization_recipe':recipe()}
            (source/'manifest.json').write_text(json.dumps(meta))
            (source/'raw_input_check.json').write_text(json.dumps({'after':source_stats(paths)}))
            run=root/'new';run.mkdir();(run/'prepared').mkdir();(run/'cache').mkdir()
            task={'sample_id':'s','patient_id':'p','normalized_cache':str(source),'raw_paths':paths,
                  'reference_manifest_sha256':sha(source/'manifest.json'),'normalization_sha256':sha(source/'normalization.json')}
            (run/'experiment.json').write_text(json.dumps({'tasks':[task],'stride':32,'cap':20,'seed':42,'he_budget_gib':1,
                'checkpoint':'synthetic','checkpoint_sha256':'synthetic'}))
            he=np.full((384,384,3),[180,70,140],np.uint8);mif=np.full((384,384,2),[16,32],np.uint8)
            class Reader:
                def __init__(self,path,*args):
                    self.pixels=he if str(path)==str(hefile) else mif
                    self.dtype=self.pixels.dtype;self.height=384;self.width=384
                def read(self,y0,y1,x0,x1):return self.pixels[y0:y1,x0:x1]
                def close(self):pass
            header=lambda path,*args:{'axes':'YXS' if str(path)==str(hefile) else 'CYX',
                                       'shape':[384,384,3] if str(path)==str(hefile) else [2,384,384],
                                       'markers':['A','B'],'mpp_xy':[.5,.5]}
            before={str(p):sha(p) for p in (hefile,miffile)}
            with patch('image_io.TiffRegionReader',Reader),patch('prepare.header',header):prepare_task(run,0)
            prepared=run/'prepared/s';m=json.loads((prepared/'manifest.json').read_text())
            self.assertNotIn('encoder',m);self.assertEqual(m['selection_policy']['stride'],32)
            with np.load(prepared/m['shards'][0]) as z:
                self.assertEqual(len(z['xy']),20);np.testing.assert_allclose(z['measured'],.25)
            with patch('encoder_runtime.load_encoder',return_value=(None,None)),patch('encoder_runtime.encode',side_effect=lambda model,transform,he,xy,device:np.zeros((len(xy),1536),np.float32)):
                extract_task(run,0,16,'cpu')
            saved=json.loads((run/'cache/s/manifest.json').read_text())
            self.assertEqual(saved['target_units'],UNITS);self.assertIn('he_array_stat',saved)
            self.assertEqual(before,{str(p):sha(p) for p in (hefile,miffile)})
            with self.assertRaises(FileExistsError):prepare_task(run,0)

if __name__=='__main__':unittest.main()
