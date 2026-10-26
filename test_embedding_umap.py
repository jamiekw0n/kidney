import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from embedding_umap import summarize_counts,half_open_matches,join_annotations,load_caches

class AnnotationEmbeddingTests(unittest.TestCase):
    def test_mixed_labels_not_forced(self):
        self.assertEqual(summarize_counts({},3,.6)['label'],'UNANNOTATED')
        self.assertEqual(summarize_counts({'T':2,'E':2},1,.5)['label'],'MIXED_OR_LOW_SUPPORT')
        self.assertEqual(summarize_counts({'T':3,'E':1},3,.6)['label'],'T')
        self.assertEqual(summarize_counts({'T':1},3,.6)['label'],'MIXED_OR_LOW_SUPPORT')
    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Optional scipy not installed')
    def test_half_open_adjacent_windows(self):
        from scipy.spatial import cKDTree
        centers=np.array([[68.,68.],[196.,68.]])
        cells=np.array([[4.,4.],[132.,68.],[131.99,68.],[260.,68.]])
        result=half_open_matches(cKDTree(centers),centers,cells,64)
        self.assertEqual(result,[[0],[1],[0],[]])
    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Optional scipy not installed')
    def test_registered_units_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'cells.csv'
            file.write_text('sample_id,cell_id,x,y,cell_type_merged\nA,c1,34,34,T\nA,c2,35,35,E\n')
            tiles=[{'sample_id':'A','x':68,'y':68}];meta={'A':{'shape':[300,300],'he_mpp':.5}}
            out=root/'valid';out.mkdir()
            c,n,a=join_annotations([file],tiles,meta,['cell_type_merged'],'micrometers',4,out)
            self.assertEqual(n.tolist(),[2]);self.assertEqual(c['cell_type_merged'][0],{'T':1,'E':1})
            file.write_text('sample_id,cell_id,x,y,cell_type_merged\nA,c1,68,68,T\nA,c1,69,69,T\n')
            out=root/'duplicates';out.mkdir()
            with self.assertRaisesRegex(ValueError,'Duplicate'):join_annotations([file],tiles,meta,['cell_type_merged'],'native-pixels',64,out)
    def test_cache_sampling_does_not_use_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sample=root/'A';sample.mkdir()
            meta=dict(complete=True,sample_id='A',patient_id='P',encoder='UNI2-h',panel=['X'],revision='r',checkpoint_sha256='h',transform='t',input_patch=128,he_mpp=.5,target_block=8,shape=[500,500],shards=['a.npz'])
            (sample/'manifest.json').write_text(json.dumps(meta))
            np.savez(sample/'a.npz',features=np.ones((4,1536),np.float32),target=np.ones((4,1),np.float32),xy=np.array([[68,68],[196,68],[68,196],[196,196]]))
            first=load_caches(root,None,2,42);second=load_caches(root,None,2,42)
            self.assertEqual(first[2],second[2]);self.assertEqual(len(first[0]),2)

class CoordinateDiscoveryTests(unittest.TestCase):
    def test_float_does_not_establish_unit_and_ambiguous_csv(self):
        from annotation_coordinates import convert_xy,annotation_files
        xy=np.array([[5194.,1904.]])
        np.testing.assert_allclose(convert_xy(xy,'native-pixels',.5),xy)
        np.testing.assert_allclose(convert_xy(xy,'micrometers',.5),xy*2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'A';folder.mkdir();(folder/'combined.csv').write_text('x,y\n')
            self.assertEqual(len(annotation_files(root,['A'])),1)
            (folder/'extra.csv').write_text('x,y\n')
            with self.assertRaises(ValueError):annotation_files(root,['A'])
